"""Score saved whole-image predictions against one immutable soft-label reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from distribution_metrics import PROBABILITY_COLUMNS, TARGET_COLUMNS
from soft_labels import construct_targets
from validation import calculate_combined_metrics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_ids(frame, label):
    if 'image_id' not in frame or frame.image_id.isna().any() or frame.image_id.astype(str).str.strip().eq('').any() or frame.image_id.duplicated().any():
        raise ValueError(f'{label}: missing or duplicate image IDs')


def same_manifest(reference, candidate):
    fields = ['image_id', 'patient_id', 'project', 'split']
    for name, frame in [('reference manifest', reference), ('candidate manifest', candidate)]:
        unique_ids(frame, name)
        if not set(fields).issubset(frame.columns) or frame[fields].isna().any().any():
            raise ValueError(f'{name}: missing split metadata')
        if not frame.split.isin(['train', 'validation', 'test']).all() or frame.groupby('patient_id').split.nunique().max() != 1:
            raise ValueError(f'{name}: invalid splits or patient leakage')
        held_out = frame.project.str.casefold().eq('visus-nhp')
        if not held_out.any() or frame.loc[held_out, 'split'].ne('test').any():
            raise ValueError(f'{name}: VISUS-NHP must remain test only')
    left = reference[fields].set_index('image_id').sort_index()
    right = candidate[fields].set_index('image_id').sort_index()
    if not left.equals(right):
        raise ValueError('Image population, patient IDs or split assignments differ from the reference')


def prepare_report(run_dir, reference_run):
    run_dir, reference_run = Path(run_dir), Path(reference_run)
    config_path = reference_run / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    columns = config['grader_columns']
    target_path = reference_run / 'targets.csv'
    targets = pd.read_csv(target_path)
    unique_ids(targets, 'reference targets')
    reconstructed = construct_targets(targets, columns)
    if not np.allclose(targets[TARGET_COLUMNS].to_numpy(dtype=float), reconstructed[TARGET_COLUMNS].to_numpy(dtype=float), rtol=0, atol=1e-8):
        raise ValueError('Saved targets disagree with their original grader ratings')
    if not np.array_equal(targets.rating_count.to_numpy(), reconstructed.rating_count.to_numpy()):
        raise ValueError('Saved rating counts disagree with original ratings')
    reference_manifest = pd.read_csv(reference_run / 'manifest.csv')
    run_manifest = pd.read_csv(run_dir / 'manifest.csv')
    same_manifest(reference_manifest, run_manifest)
    same_manifest(reference_manifest, targets)
    eligible = reconstructed.loc[reconstructed.split.eq('validation') & reconstructed.has_ratings].copy()
    prediction_path = run_dir / 'validation_predictions.csv'
    predictions = pd.read_csv(prediction_path)
    unique_ids(predictions, 'saved predictions')
    if not set(PROBABILITY_COLUMNS).issubset(predictions.columns):
        raise ValueError('Saved predictions lack five class probabilities')
    val_ids = set(reference_manifest.loc[reference_manifest.split.eq('validation'), 'image_id'])
    if not set(predictions.image_id).issubset(val_ids):
        raise ValueError('Saved predictions contain non-validation IDs')
    if 'split' in predictions and not predictions.split.eq('validation').all():
        raise ValueError('Saved prediction split must be validation')
    if not set(eligible.image_id).issubset(predictions.image_id):
        raise ValueError('Missing predictions for rated validation images; cannot compare on identical images')
    aligned = predictions.set_index('image_id').loc[eligible.image_id]
    metrics, per_image = calculate_combined_metrics(eligible, aligned[PROBABILITY_COLUMNS].to_numpy(), columns)
    if 'predicted_grade' in aligned and not np.array_equal(aligned.predicted_grade.to_numpy(), per_image.predicted_grade.to_numpy()):
        raise ValueError('Saved predicted grades disagree with argmax of whole-image probabilities')
    metrics.update({
        'run_name': run_dir.name, 'split': 'validation',
        'reference_run': str(reference_run.resolve()), 'grader_columns': columns,
        'reference_targets_sha256': file_hash(target_path),
        'reference_config_sha256': file_hash(config_path),
        'predictions_sha256': file_hash(prediction_path),
        'manifest_sha256': file_hash(run_dir / 'manifest.csv'),
        'reference_population': 'only rated validation images; train/test never scored',
    })
    details = eligible[['image_id', 'split'] + columns + TARGET_COLUMNS].reset_index(drop=True)
    details[PROBABILITY_COLUMNS] = aligned[PROBABILITY_COLUMNS].to_numpy()
    details = pd.concat([details, per_image], axis=1)
    return metrics, details


def score_runs(run_dirs, reference_run, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} exists; choose a new output directory')
    run_dirs = [Path(run) for run in run_dirs]
    if len({run.name for run in run_dirs}) != len(run_dirs) or len({run.resolve() for run in run_dirs}) != len(run_dirs):
        raise ValueError('Run names and paths must be unique')
    reports = [(run, *prepare_report(run, reference_run)) for run in run_dirs]
    reference_run = Path(reference_run)
    target_bytes = (reference_run / 'targets.csv').read_bytes()
    config_bytes = (reference_run / 'run_config.json').read_bytes()
    if not reports or any(metrics['reference_targets_sha256'] != hashlib.sha256(target_bytes).hexdigest()
                          or metrics['reference_config_sha256'] != hashlib.sha256(config_bytes).hexdigest()
                          for _, metrics, _ in reports):
        raise ValueError('Reference snapshot changed while preparing comparison')
    # Preflight every run before writing any reports; never change original run outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'reference_targets.csv').write_bytes(target_bytes)
    (output_dir / 'reference_config.json').write_bytes(config_bytes)
    comparison = []
    for run, metrics, details in reports:
        destination = output_dir / run.name
        destination.mkdir()
        (destination / 'distribution_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        details.to_csv(destination / 'distribution_per_image.csv', index=False)
        comparison.append({key: metrics[key] for key in [
            'run_name', 'image_count', 'earth_movers_distance', 'soft_cross_entropy',
            'matches_any_grader_accuracy', 'consensus_reference_exact_accuracy',
            'consensus_reference_mae', 'consensus_reference_quadratic_weighted_kappa',
            'reference_tie_count', 'reference_tie_percentage', 'reference_targets_sha256']})
    summary = pd.DataFrame(comparison)
    summary.to_csv(output_dir / 'comparison.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Reports: {output_dir}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--run-dir', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    score_runs(args.run_dir, args.reference_run, args.output_dir)
