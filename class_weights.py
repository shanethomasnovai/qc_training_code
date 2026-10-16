"""Reuse a hard-label run's weights without consulting new grader votes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soft_labels import read_table, sha256, validate_manifest


def baseline_class_weights(run_dir, reference_grader=None, current_manifest=None):
    run_dir = Path(run_dir).resolve()
    params_path = run_dir / 'model_params.csv'
    params = pd.read_csv(params_path).iloc[0]
    config_path = run_dir / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    saved_graders = [value for value in (config.get('grader'), params.get('grader')) if pd.notna(value)]
    if len(set(saved_graders)) > 1:
        raise ValueError('Baseline grader metadata conflicts')
    grader = reference_grader or (saved_graders[0] if saved_graders else None)
    if grader not in ('previousStudent', 'shane'):
        raise ValueError('Specify --weight_reference_grader previousStudent or shane for a hard-label baseline without grader metadata')
    if saved_graders and any(value != grader for value in saved_graders):
        raise ValueError('Requested weight reference grader conflicts with baseline metadata')
    manifest_name = params.get('dataset_manifest', 'manifest.csv')
    manifest_path = run_dir / manifest_name
    manifest = read_table(manifest_path)
    validate_manifest(manifest, check_files=False)
    if current_manifest is not None:
        current = read_table(current_manifest)
        identity = ['image_id', 'patient_id', 'project', 'split']
        if not manifest[identity].sort_values('image_id').reset_index(drop=True).equals(
                current[identity].sort_values('image_id').reset_index(drop=True)):
            raise ValueError('Current image population/patient splits differ from weight baseline')
    source = {'baseline_run': str(run_dir), 'reference_grader': grader,
              'baseline_manifest_sha256': sha256(manifest_path),
              'model_params_sha256': sha256(params_path), 'grade_order': [1, 2, 3, 4, 5]}
    if 'class_weights' in config:
        values = config['class_weights']
        if values is None or config.get('loss', 'CrossEntropyLoss') != 'CrossEntropyLoss':
            raise ValueError('Baseline has no reusable weighted CrossEntropyLoss vector')
        source.update({'method': 'saved run_config.json class_weights',
                       'baseline_config_sha256': sha256(config_path)})
    else:
        labels = pd.to_numeric(manifest.loc[manifest.split.eq('train'), f'{grader}_grade'])
        counts = [int(labels.eq(grade).sum()) for grade in range(1, 6)]
        if not all(counts):
            raise ValueError('Cannot reconstruct baseline: a training grade has no examples')
        # Nine patches and their flips multiply every class count equally.
        values = [len(labels) / (5 * count) for count in counts]
        source.update({'method': 'reconstructed inverse training frequency',
                       'formula': 'N_train_images / (5 * training_image_count_for_grade)',
                       'training_image_counts': counts, 'training_image_count': len(labels),
                       'uniform_training_patches_per_image_including_flip': 18})
    weights = np.asarray(values, dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Baseline weights must be five finite positive values in grade order 1-5')
    return weights.tolist(), source
