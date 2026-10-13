"""Whole-image validation of a saved five-grade AutoQC checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from dataset import ImageQualityDataset
from model import IQAnet


def image_transform():
    return transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])


def calculate_validation_metrics(reference, predicted):
    """Zero undefined P/R/F1 explicitly; undefined kappa becomes JSON null."""
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    if reference.size == 0 or reference.shape != predicted.shape:
        raise ValueError('Expected matching, nonempty grade arrays')
    if not np.isin(reference, range(1, 6)).all() or not np.isin(predicted, range(1, 6)).all():
        raise ValueError('Grades must be 1 through 5')
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (reference.astype(int) - 1, predicted.astype(int) - 1), 1)
    support, predicted_count, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    precision = np.divide(tp, predicted_count, out=np.zeros(5), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    denominator = support + predicted_count
    f1 = np.divide(2 * tp, denominator, out=np.zeros(5), where=denominator != 0)
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
    expected = np.outer(support, predicted_count) / reference.size
    expected_disagreement = float((weights * expected).sum())
    kappa = 1 - float((weights * cm).sum()) / expected_disagreement if expected_disagreement > 0 else None
    error = np.abs(reference - predicted)
    metrics = {
        'image_count': int(reference.size),
        'exact_accuracy': float((error == 0).mean()),
        'within_one_grade_accuracy': float((error <= 1).mean()),
        'mean_absolute_grade_error': float(error.mean()),
        'quadratic_weighted_kappa': kappa,
        'macro_f1': float(f1.mean()),
        'weighted_f1': float(np.dot(f1, support) / support.sum()),
        'undefined_metric_policy': 'Undefined precision/recall/F1 are 0 with per-grade defined flags; macro F1 includes all five grades. Undefined kappa is null.',
        'quadratic_weighted_kappa_undefined_reason': 'Zero expected disagreement' if kappa is None else None,
    }
    per_grade = pd.DataFrame({
        'grade': range(1, 6), 'precision': precision, 'recall': recall,
        'f1': f1, 'support': support,
        'precision_defined': predicted_count != 0, 'recall_defined': support != 0,
        'f1_defined': denominator != 0,
    })
    return metrics, per_grade, pd.DataFrame(cm, index=range(1, 6), columns=range(1, 6))


def evaluate_validation(model, manifest, grader, output_dir, device='cpu', batch_size=16):
    """Average nine patch softmax probabilities, preserving each image's row ID."""
    if grader not in ('previousStudent', 'shane') or batch_size < 1:
        raise ValueError('Invalid grader or batch size')
    df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    required = {'image_id', 'image_path', 'patient_id', 'project', 'split', f'{grader}_grade'}
    if not required.issubset(df.columns) or df.empty or df[list(required)].isna().any().any():
        raise ValueError('Expected complete shared manifest')
    if df.image_id.duplicated().any() or df.image_path.duplicated().any():
        raise ValueError('Duplicate image IDs or paths')
    if not df.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown split')
    if df.groupby('patient_id').split.nunique().max() != 1:
        raise ValueError('Patient leakage')
    held_out = df.project.str.casefold() == 'visus-nhp'
    if (df.loc[held_out, 'split'] != 'test').any():
        raise ValueError('VISUS-NHP must remain entirely in test')
    # Filter before any image path is opened. Never decode train/test images.
    val = df[df.split == 'validation'].copy().reset_index(drop=True)
    if val.empty or not val[f'{grader}_grade'].isin(range(1, 6)).all():
        raise ValueError('Empty validation or invalid reference grades')
    output_dir = Path(output_dir)
    filenames = ['validation_metrics.json', 'validation_per_grade_metrics.csv',
                 'validation_confusion_matrix.csv', 'validation_predictions.csv']
    for filename in filenames:
        if (output_dir / filename).exists():
            raise FileExistsError(f'{output_dir / filename} already exists; preserving existing results')
    val['image_filename'] = val.image_path
    val['image_grading'] = val[f'{grader}_grade']
    dataset = ImageQualityDataset(val, image_transform(), label_mapping={grade: grade - 1 for grade in range(1, 6)})
    to_pil, to_tensor = transforms.ToPILImage(), transforms.ToTensor()
    model.to(device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for index, row in val.iterrows():
            image, target = dataset[index]
            if int(target) != int(row[f'{grader}_grade']) - 1:
                raise ValueError('Reference grade alignment failed')
            # Identical row-major 3x3 crops and PIL conversion to subsample_subimages.
            pil_image = to_pil(image)
            patches = torch.stack([
                to_tensor(pil_image.crop((col * 256, r * 256, (col + 1) * 256, (r + 1) * 256)))
                for r in range(3) for col in range(3)
            ])
            probabilities = []
            for start in range(0, 9, batch_size):
                logits = model(patches[start:start + batch_size].to(device))
                if logits.shape != (min(batch_size, 9 - start), 5):
                    raise ValueError('Expected five logits per patch')
                probabilities.append(torch.softmax(logits, dim=1).cpu())
            patch_probabilities = torch.cat(probabilities)
            if patch_probabilities.shape != (9, 5) or not torch.isfinite(patch_probabilities).all():
                raise ValueError('Expected nine finite patch probability vectors')
            mean_probability = patch_probabilities.mean(dim=0).numpy()
            predictions.append({
                'image_id': row.image_id,
                'reference_grade': int(row[f'{grader}_grade']),
                'predicted_grade': int(mean_probability.argmax()) + 1,
                **{f'probability_grade_{grade}': float(mean_probability[grade - 1]) for grade in range(1, 6)},
            })
            print(f'Validation images: {index + 1}/{len(val)}', end='\r')
    predictions = pd.DataFrame(predictions)
    if predictions.image_id.tolist() != val.image_id.tolist():
        raise ValueError('Image ID alignment failed')
    metrics, per_grade, cm = calculate_validation_metrics(predictions.reference_grade, predictions.predicted_grade)
    metrics.update({'grader': grader, 'split': 'validation', 'patches_per_image': 9,
                    'aggregation': 'mean of patch softmax probabilities',
                    'grade_to_class_index': {str(grade): grade - 1 for grade in range(1, 6)}})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / filenames[3], index=False)
    per_grade.to_csv(output_dir / filenames[1], index=False)
    cm.index.name = 'reference_grade'
    cm.columns = [f'predicted_grade_{grade}' for grade in range(1, 6)]
    cm.to_csv(output_dir / filenames[2])
    (output_dir / filenames[0]).write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    print('\nWhole-image validation (best saved checkpoint):')
    for key in ('exact_accuracy', 'within_one_grade_accuracy', 'mean_absolute_grade_error',
                'quadratic_weighted_kappa', 'macro_f1', 'weighted_f1'):
        value = metrics[key]
        print(f"  {key}: {value:.6f}" if value is not None else f'  {key}: undefined (zero expected disagreement)')
    print(f'Validation results: {output_dir}')
    return metrics, predictions


def evaluate_saved_run(run_dir, grader=None, device=None, batch_size=None):
    """Load the run's own manifest, model parameters and best saved state dict."""
    run_dir = Path(run_dir)
    params = pd.read_csv(run_dir / 'model_params.csv').iloc[0]
    if grader is not None and 'grader' in params and grader != params['grader']:
        raise ValueError('Requested grader differs from saved run metadata')
    grader = grader or params.get('grader')
    if grader not in ('previousStudent', 'shane'):
        raise ValueError('Specify --grader for older runs without saved grader metadata')
    def boolean(name):
        value = str(params[name]).lower()
        if value not in ('true', 'false'):
            raise ValueError(f'Invalid saved setting {name}: {params[name]}')
        return value == 'true'
    if boolean('output_sigmoid'):
        raise ValueError('Expected multiclass logits, not sigmoid output')
    device = device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = IQAnet(num_classes=5, batch_normalization=boolean('batch_normalization'), output_sigmoid=False)
    model.load_state_dict(torch.load(run_dir / 'model.pth', map_location='cpu', weights_only=True))
    print(f'Loading best saved checkpoint: {run_dir / "model.pth"}')
    return evaluate_validation(model, run_dir / params.get('dataset_manifest', 'manifest.csv'), grader, run_dir,
                               device=device, batch_size=int(params['batch_size']) if batch_size is None else batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--grader', choices=['previousStudent', 'shane'])
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    evaluate_saved_run(args.run_dir, args.grader, batch_size=args.batch_size)
