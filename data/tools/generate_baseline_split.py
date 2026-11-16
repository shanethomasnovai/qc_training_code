"""Save baseline/non-baseline training selections using '.0' in the full image ID.

Labels remain in reviewed_v2; these CSVs select all training images in each
category. Existing patient assignments and validation/test sets are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.labels import load_workbook
from auto_qc.CNN.utils import sha256


def select_training(records):
    training = records.loc[records.split.eq('train'), ['image_id', 'patient_id']].copy()
    training = training.sort_values('image_id')
    training['baseline'] = training.image_id.str.contains('.0', regex=False)
    baseline = training.loc[training.baseline, ['image_id', 'patient_id']].reset_index(drop=True)
    non_baseline = training.loc[~training.baseline, ['image_id', 'patient_id']].reset_index(drop=True)
    if baseline.empty or non_baseline.empty:
        raise ValueError('Both training categories must contain images')
    return baseline, non_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/datasets/reviewed_v2/multigrader_labels.xlsx')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing outputs are never overwritten')
    args = parser.parse_args()
    labels = args.labels.resolve()
    records, metadata = load_workbook(labels, check_files=True)
    baseline, non_baseline = select_training(records)
    output = args.output_dir or ROOT / 'data/datasets/baseline_split'
    output.mkdir(parents=True, exist_ok=False)
    baseline.to_csv(output / 'baseline_manifest.csv', index=False)
    non_baseline.to_csv(output / 'non_baseline_manifest.csv', index=False)
    summary = {
        'baseline_rule': "Literal '.0' occurs anywhere in image_id; image paths are not searched",
        'projects': 'All projects in the saved training split; no project-specific overrides',
        'selection': 'All training images in each category; no sampling',
        'baseline_images': len(baseline), 'non_baseline_images': len(non_baseline),
        'baseline_patients': int(baseline.patient_id.nunique()),
        'non_baseline_patients': int(non_baseline.patient_id.nunique()),
        'validation_image_ids': records.loc[records.split.eq('validation'), 'image_id'].tolist(),
        'test_policy': 'Never sampled or evaluated',
        'labels': {'path': str(labels), 'sha256': sha256(labels)},
        'dataset_manifest_sha256': sha256(labels.parent / 'manifest.json'),
        'patient_split': metadata['sources']['split'],
        'script_sha256': sha256(__file__),
        'output_sha256': {name: sha256(output / name) for name in ('baseline_manifest.csv', 'non_baseline_manifest.csv')},
    }
    (output / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Baseline: {len(baseline)} images, {baseline.patient_id.nunique()} patients')
    print(f'Non-baseline: {len(non_baseline)} images, {non_baseline.patient_id.nunique()} patients')
    print(f'Validation unchanged: {len(summary["validation_image_ids"])} images')
    print(f'Saved: {output.resolve()}')


if __name__ == '__main__':
    main()
