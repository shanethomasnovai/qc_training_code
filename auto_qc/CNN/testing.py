"""Load saved architectures; evaluate patches and whole images separately."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import VOTES, extract_patches, load_workbook
from .losses import grade_probabilities, ordering_diagnostics
from .metrics import calculate_combined_metrics, calculate_validation_metrics
from .utils import load_checkpoint, write_json


def confusion_figure(matrix, path):
    from PIL import Image, ImageDraw
    n = len(matrix)
    image = Image.new('RGB', (100+n*70, 100+n*70), 'white')
    draw = ImageDraw.Draw(image)
    values = np.asarray(matrix)
    maximum = max(1, int(values.max()))
    for r in range(n):
        draw.text((20, 75+r*70), str(r+1), fill='black')
        draw.text((75+r*70, 30), str(r+1), fill='black')
        for c in range(n):
            shade = 255-int(170*values[r, c]/maximum)
            draw.rectangle((60+c*70, 60+r*70, 129+c*70, 129+r*70), fill=(shade, shade, 255), outline='white')
            draw.text((85+c*70, 85+r*70), str(values[r, c]), fill='black')
    draw.text((20, 10), 'Rows: reference; columns: predicted', fill='black')
    image.save(path)


def evaluate(model, frame, config, output_dir, split='validation', device='cpu'):
    output_dir = Path(output_dir)
    filename = output_dir / f'{split}_predictions.csv'
    if filename.exists():
        raise FileExistsError(filename)
    records = frame.loc[frame.split.eq(split) & frame.eligible & frame.evaluation_eligible].copy().reset_index(drop=True)
    if records.empty:
        raise ValueError(f'No eligible {split} images')
    head, batch_size = config['model']['head'], config['training']['batch_size']
    model.eval()
    images, patches, binary_values, raw_values = [], [], [], []
    with torch.inference_mode():
        for row in records.itertuples():
            crops = extract_patches(row.image_path)
            probabilities, thresholds, binary = [], [], []
            for start in range(0, 9, batch_size):
                outputs = model(crops[start:start+batch_size].to(device))
                logits, binary_logits = outputs if isinstance(outputs, tuple) else (outputs, None)
                probabilities.append(grade_probabilities(logits, head).cpu())
                if head != 'five_logits':
                    thresholds.append(logits.sigmoid().cpu())
                if binary_logits is not None:
                    binary.append(binary_logits.sigmoid().cpu())
            p = torch.cat(probabilities)
            images.append(p.mean(0).numpy())
            if binary:
                binary_values.append(float(torch.cat(binary).mean()))
            raw = torch.cat(thresholds) if thresholds else None
            if raw is not None:
                raw_values.append(raw)
            for patch_index in range(9):
                detail = {'image_id': row.image_id, 'patch_index': patch_index, 'split': split}
                detail.update({f'probability_grade_{k}': float(p[patch_index, k-1]) for k in range(1, 6)})
                if raw is not None:
                    detail.update({f'raw_above_grade_{k}': float(raw[patch_index, k-1]) for k in range(1, 5)})
                if binary:
                    detail['binary_head_p_unacceptable'] = float(torch.cat(binary)[patch_index])
                patches.append(detail)
    p = np.stack(images)
    metrics, details = calculate_combined_metrics(records, p, config['dataset']['graders'], .5,
                                                 np.asarray(binary_values) if binary_values else None)
    for name in details:
        records[name] = details[name].to_numpy()
    records[[f'probability_grade_{k}' for k in range(1, 6)]] = p
    if binary_values:
        records['binary_head_p_unacceptable'] = binary_values
    metrics.update({'split': split, 'evaluation_unit': 'whole_image', 'patch_aggregation': 'mean of reconstructed patch grade probabilities',
                    'primary_multiclass_metric': 'consensus_reference_quadratic_weighted_kappa',
                    'primary_binary_rule': 'grade_argmax', 'monitoring_population_size': len(config['cohort']['monitor_image_ids'])})
    consensus, per_grade, cm = calculate_validation_metrics(records.consensus_reference_grade.to_numpy(), records.predicted_grade.to_numpy())
    metrics['consensus'] = consensus
    metrics['per_grader'] = {}
    for grader in config['dataset']['graders']:
        available = records[grader].notna()
        if available.any():
            metrics['per_grader'][grader] = calculate_validation_metrics(records.loc[available, grader].to_numpy(), records.loc[available, 'predicted_grade'].to_numpy())[0]
    if raw_values:
        raw = torch.cat(raw_values)
        metrics['ordinal_ordering'] = ordering_diagnostics(raw)
        records['paper_threshold_count_grade'] = [1+int((r.mean(0) > .5).sum()) for r in raw_values]
        metrics['paper_threshold_count'] = {'rule': 'reference only: 1+count(mean raw patch sigmoid > .5); image averaging is our adaptation',
            'metrics': calculate_validation_metrics(records.consensus_reference_grade.to_numpy(), records.paper_threshold_count_grade.to_numpy())[0]}
        changed_images = sum(bool((r[:, 1:] > r[:, :-1]).any()) for r in raw_values)
        metrics['ordinal_ordering'].update({'projection_changed_image_count': changed_images, 'projection_changed_image_fraction': changed_images/len(records)})
    metrics['secondary_subsets'] = {}
    for label, ids in config['evaluation']['subsets'].items():
        selected = records.image_id.isin(ids)
        if selected.any():
            metrics['secondary_subsets'][label] = calculate_combined_metrics(records.loc[selected], p[selected], config['dataset']['graders'], .5,
                np.asarray(binary_values)[selected] if binary_values else None)[0]
    clear = ~records.binary_reference_tied.to_numpy()
    metrics['clear_majority'] = calculate_combined_metrics(records.loc[clear], p[clear], config['dataset']['graders'], .5,
                np.asarray(binary_values)[clear] if binary_values else None)[0] if clear.any() else {'image_count': 0}
    records.to_csv(filename, index=False)
    pd.DataFrame(patches).to_csv(output_dir / f'{split}_patch_predictions.csv', index=False)
    per_grade.to_csv(output_dir / f'{split}_per_grade_metrics.csv', index=False)
    cm.to_csv(output_dir / f'{split}_confusion_matrix.csv', index_label='reference_grade')
    confusion_figure(cm, output_dir / f'{split}_confusion_matrix.png')
    for method in ('binary', 'binary_grade_argmax', 'binary_dedicated_head'):
        if method in metrics:
            m = metrics[method]
            pd.DataFrame([[m['TN'], m['FP']], [m['FN'], m['TP']]], index=['acceptable', 'unacceptable'],
                columns=['predicted_acceptable', 'predicted_unacceptable']).to_csv(output_dir / f'{split}_{method}_confusion_matrix.csv')
    write_json(output_dir / f'{split}_metrics.json', metrics)
    return metrics, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--enable-test', action='store_true')
    args = parser.parse_args()
    if args.split == 'test' and not args.enable_test:
        parser.error('Test evaluation requires explicit --enable-test')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    model, config, _ = load_checkpoint(args.run_dir)
    frame, _ = load_workbook(args.run_dir / 'dataset.xlsx', check_files=True)
    if args.split == 'test':
        frame.loc[frame.split.eq('test'), 'evaluation_eligible'] = frame.loc[frame.split.eq('test'), 'eligible']
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'evaluation_request.json', {'source_run': str(args.run_dir), 'split': args.split,
                                                           'explicit_test_permission': args.enable_test})
    evaluate(model, frame, config, args.output_dir, args.split)


if __name__ == '__main__':
    main()
