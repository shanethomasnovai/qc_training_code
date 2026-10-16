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
from soft_labels import TARGET_COLUMNS, construct_targets
from distribution_metrics import calculate_distribution_metrics


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


def calculate_combined_metrics(reference, probabilities, grader_columns):
    rated = construct_targets(reference.reset_index(drop=True), grader_columns)
    if not rated.has_ratings.all():
        raise ValueError('Combined metrics require at least one selected rating per image')
    targets = reference[TARGET_COLUMNS].to_numpy() if set(TARGET_COLUMNS).issubset(reference.columns) else rated[TARGET_COLUMNS].to_numpy()
    metrics, details = calculate_distribution_metrics(targets, probabilities)
    if not np.allclose(targets, rated[TARGET_COLUMNS].to_numpy(), rtol=0, atol=1e-8):
        raise ValueError('Soft targets disagree with selected original ratings')
    ratings = rated[grader_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    votes = np.stack([(ratings == grade).sum(axis=1) for grade in range(1, 6)], axis=1)
    winners = votes == votes.max(axis=1, keepdims=True)
    reference_grade = votes.argmax(axis=1) + 1
    predicted = np.asarray(probabilities).argmax(axis=1) + 1
    tied = winners.sum(axis=1) > 1
    matches_any = votes[np.arange(len(votes)), predicted - 1] > 0
    errors = np.abs(reference_grade - predicted)
    consensus, _, _ = calculate_validation_metrics(reference_grade, predicted)
    metrics.update({
        'matches_any_grader_accuracy': float(matches_any.mean()),
        'consensus_reference_exact_accuracy': consensus['exact_accuracy'],
        'consensus_reference_mae': consensus['mean_absolute_grade_error'],
        'consensus_reference_quadratic_weighted_kappa': consensus['quadratic_weighted_kappa'],
        'consensus_reference_quadratic_weighted_kappa_undefined_reason': consensus['quadratic_weighted_kappa_undefined_reason'],
        'reference_tie_count': int(tied.sum()),
        'reference_tie_percentage': float(tied.mean() * 100),
        'single_grade_reference_convention': 'Reference grades are the most frequent available grader rating; ties are resolved in favour of the lower grade.',
    })
    details['predicted_grade'] = predicted
    details['consensus_reference_grade'] = reference_grade
    details['reference_tied'] = tied
    details['rating_count'] = rated.rating_count.to_numpy()
    details['matches_any_grader'] = matches_any
    details['consensus_exact_match'] = errors == 0
    details['absolute_grade_error'] = errors
    return metrics, details


def evaluate_validation(model, manifest, grader, output_dir, device='cpu', batch_size=16, grader_columns=None, class_weights=None):
    """Average nine patch softmax probabilities, preserving each image's row ID."""
    soft = grader == 'soft'
    if grader not in ('previousStudent', 'shane', 'soft') or batch_size < 1:
        raise ValueError('Invalid grader or batch size')
    df = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    required = {'image_id', 'image_path', 'patient_id', 'project', 'split', f'{grader}_grade'}
    if soft:
        required.remove('soft_grade')
        df = construct_targets(df, grader_columns)
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
    if soft:
        val = val[val.has_ratings].reset_index(drop=True)
    if val.empty or (not soft and not val[f'{grader}_grade'].isin(range(1, 6)).all()):
        raise ValueError('Empty validation or invalid reference grades')
    output_dir = Path(output_dir)
    filenames = ['validation_metrics.json', 'validation_per_grade_metrics.csv',
                 'validation_confusion_matrix.csv', 'validation_predictions.csv']
    distribution_files = ['distribution_metrics.json', 'distribution_per_image.csv'] if soft else []
    for filename in filenames + distribution_files:
        if (output_dir / filename).exists():
            raise FileExistsError(f'{output_dir / filename} already exists; preserving existing results')
    val['image_filename'] = val.image_path
    if not soft:
        val['image_grading'] = val[f'{grader}_grade']
    dataset = ImageQualityDataset(val, image_transform(), label_mapping={grade: grade - 1 for grade in range(1, 6)},
                                  target_columns=TARGET_COLUMNS if soft else None)
    to_pil, to_tensor = transforms.ToPILImage(), transforms.ToTensor()
    model.to(device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for index, row in val.iterrows():
            image, target = dataset[index]
            if not soft and int(target) != int(row[f'{grader}_grade']) - 1:
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
            reference = ({**{column: row[column] for column in grader_columns},
                          **{column: float(row[column]) for column in TARGET_COLUMNS},
                          'rating_count': int(row.rating_count)} if soft else
                         {'reference_grade': int(row[f'{grader}_grade'])})
            predictions.append({
                'image_id': row.image_id,
                **reference,
                'predicted_grade': int(mean_probability.argmax()) + 1,
                **{f'probability_grade_{grade}': float(mean_probability[grade - 1]) for grade in range(1, 6)},
            })
            print(f'Validation images: {index + 1}/{len(val)}', end='\r')
    predictions = pd.DataFrame(predictions)
    if predictions.image_id.tolist() != val.image_id.tolist():
        raise ValueError('Image ID alignment failed')
    if soft:
        metrics = {'references': {}}
        grade_tables, matrices = [], []
        for column in grader_columns:
            reference = pd.to_numeric(predictions[column], errors='coerce')
            available = reference.notna()
            if not available.any():
                metrics['references'][column] = {'grader': column, 'image_count': 0, 'reason': 'No available validation ratings'}
                continue
            scores, grades, matrix = calculate_validation_metrics(reference[available], predictions.loc[available, 'predicted_grade'])
            scores['grader'] = column
            metrics['references'][column] = scores
            grades.insert(0, 'grader', column)
            matrix.index.name = 'reference_grade'
            matrix = matrix.reset_index()
            matrix.insert(0, 'grader', column)
            grade_tables.append(grades)
            matrices.append(matrix)
            print(f'\nReference {column}: {scores}')
        per_grade = pd.concat(grade_tables, ignore_index=True)
        cm = pd.concat(matrices, ignore_index=True)
        probabilities = predictions[[f'probability_grade_{i}' for i in range(1, 6)]].to_numpy()
        targets = predictions[TARGET_COLUMNS].to_numpy()
        distribution_metrics, distribution_details = calculate_combined_metrics(val, probabilities, grader_columns)
        distribution_metrics.update({'grader_columns': grader_columns, 'split': 'validation'})
        distribution_details.insert(0, 'image_id', predictions.image_id.to_numpy())
        for column in distribution_details.columns:
            if column != 'image_id':
                predictions[column] = distribution_details[column].to_numpy()
        for column in grader_columns + TARGET_COLUMNS + [f'probability_grade_{i}' for i in range(1, 6)]:
            distribution_details[column] = predictions[column].to_numpy()
        metrics.update(distribution_metrics)
        metrics.update({'image_count': len(predictions),
                        'soft_cross_entropy': distribution_metrics['soft_cross_entropy'],
                        'earth_movers_distance': distribution_metrics['earth_movers_distance'],
                        'earth_movers_distance_units': 'grade steps (0-4); lower is better',
                        'loss': 'unweighted cross-entropy of mean patch probabilities',
                        'grader_columns': grader_columns})
        if class_weights is not None:
            weights = np.asarray(class_weights, dtype=float)
            if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Expected five finite positive class weights')
            scored_targets = targets / targets.sum(1, keepdims=True)
            scored_probabilities = probabilities.astype(float) / probabilities.astype(float).sum(1, keepdims=True)
            metrics.update({'weighted_soft_cross_entropy': float(-(scored_targets * weights * np.log(np.maximum(scored_probabilities, np.finfo(np.float32).tiny))).sum(1).mean()),
                            'class_weights': weights.tolist(), 'weighted_loss_reduction': 'mean over images',
                            'weighted_loss': 'class-weighted cross-entropy of mean patch probabilities'})
    else:
        metrics, per_grade, cm = calculate_validation_metrics(predictions.reference_grade, predictions.predicted_grade)
    metrics.update({'grader': grader, 'split': 'validation', 'patches_per_image': 9,
                    'aggregation': 'mean of patch softmax probabilities',
                    'grade_to_class_index': {str(grade): grade - 1 for grade in range(1, 6)}})
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / filenames[3], index=False)
    per_grade.to_csv(output_dir / filenames[1], index=False)
    if soft:
        cm = cm.rename(columns={grade: f'predicted_grade_{grade}' for grade in range(1, 6)})
        cm.to_csv(output_dir / filenames[2], index=False)
    else:
        cm.index.name = 'reference_grade'
        cm.columns = [f'predicted_grade_{grade}' for grade in range(1, 6)]
        cm.to_csv(output_dir / filenames[2])
    (output_dir / filenames[0]).write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    if soft:
        (output_dir / distribution_files[0]).write_text(json.dumps(distribution_metrics, indent=2, allow_nan=False), encoding='utf-8')
        distribution_details.to_csv(output_dir / distribution_files[1], index=False)
    print('\nWhole-image validation (best saved checkpoint):')
    if soft:
        print(f"  earth_movers_distance (grade steps): {metrics['earth_movers_distance']:.6f}")
        print(f"  soft_cross_entropy (unweighted): {metrics['soft_cross_entropy']:.6f}")
        for key in ('matches_any_grader_accuracy', 'consensus_reference_exact_accuracy',
                    'consensus_reference_mae', 'consensus_reference_quadratic_weighted_kappa'):
            value = metrics[key]
            print(f'  {key}: {value:.6f}' if value is not None else f'  {key}: undefined (zero expected disagreement)')
        print(f"  Reference ties: {metrics['reference_tie_count']}/{metrics['image_count']} ({metrics['reference_tie_percentage']:.2f}%)")
    for key in ('exact_accuracy', 'within_one_grade_accuracy', 'mean_absolute_grade_error',
                'quadratic_weighted_kappa', 'macro_f1', 'weighted_f1'):
        if soft:
            continue
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
    if grader not in ('previousStudent', 'shane', 'soft'):
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
    columns = None
    class_weights = None
    manifest = run_dir / params.get('dataset_manifest', 'manifest.csv')
    if grader == 'soft':
        config = json.loads((run_dir / 'run_config.json').read_text(encoding='utf-8'))
        columns = config['grader_columns']
        class_weights = config.get('effective_class_weights_float32', config.get('class_weights'))
        manifest = run_dir / 'targets.csv'
    return evaluate_validation(model, manifest, grader, run_dir,
                               device=device, batch_size=int(params['batch_size']) if batch_size is None else batch_size,
                               grader_columns=columns, class_weights=class_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--grader', choices=['previousStudent', 'shane', 'soft'])
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    evaluate_saved_run(args.run_dir, args.grader, batch_size=args.batch_size)
