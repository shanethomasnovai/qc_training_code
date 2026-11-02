"""Generate all target workbooks from hash-pinned ratings and one saved split."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .dataset import (canonical_columns, canonical_graders, generate_targets, image_key,
                      logical_hash, ratings_array, save_workbook, load_workbook, validate_records)
from .utils import ROOT, read_config, sha256, write_json

VARIANTS = {
    'ganaav_labels.xlsx': ('hard_ganaav', ['ganaav_grade']),
    'shane_labels.xlsx': ('hard_shane', ['shane_grade']),
    'ganaav_shane_ceiling_labels.xlsx': ('ceiling_mean', ['ganaav_grade', 'shane_grade']),
    'ganaav_shane_mean_labels.xlsx': ('exact_mean', ['ganaav_grade', 'shane_grade']),
    'ganaav_shane_soft_labels.xlsx': ('soft_categorical', ['ganaav_grade', 'shane_grade']),
    'multigrader_soft_labels.xlsx': ('soft_categorical', None),
    'multigrader_mean_labels.xlsx': ('exact_mean', None),
    'ordinal_targets.xlsx': ('ordinal', None),
}
RULES = {
    'hard_ganaav': 'ganaav integer grade; class_index=grade-1',
    'hard_shane': 'shane integer grade; class_index=grade-1',
    'ceiling_mean': 'ceil(arithmetic mean), all selected ratings required',
    'exact_mean': 'exact arithmetic mean; paired mean requires both, historical multigrader mean ignores missing',
    'soft_categorical': 'fraction of available selected votes for each grade 1..5; equal image weight',
    'ordinal': 'P(grade>1),P(grade>2),P(grade>3),P(grade>4); available selected votes only',
}


def pinned_table(spec, excel=False):
    path = ROOT / spec['path']
    if sha256(path) != spec['sha256']:
        raise ValueError(f'Input hash mismatch: {path}')
    table = pd.read_excel(path, sheet_name=spec.get('sheet', 'Grades')) if excel else pd.read_csv(path)
    table = canonical_columns(table)
    if 'image_id' not in table:
        table['image_id'] = table.image_filename.map(image_key)
    if table.image_id.isna().any() or table.image_id.eq('').any() or table.image_id.duplicated().any():
        raise ValueError(f'Duplicate/missing full IDs in {path}')
    return table


def join_sources(build, profile):
    split = pinned_table(build['split'])
    # Only assignment/identity columns, never the carrier manifest's rating values.
    fields = ['image_id', 'image_path', 'patient_id', 'project', 'project_site', 'split']
    result = split[fields].sort_values('image_id').reset_index(drop=True)
    ganaav = pinned_table(build['ganaav'])
    shane = pinned_table(build['shane'])
    allowed = set(build['allowed_unmatched_ids'])
    exclusions = []
    for name, source in [('ganaav', ganaav), ('shane', shane)]:
        unknown = set(source.image_id) - set(result.image_id)
        if unknown - allowed:
            raise ValueError(f'Unapproved unmatched {name} IDs: {sorted(unknown-allowed)}')
        exclusions += [{'source': name, 'image_id': value, 'reason': 'outside fixed paired population'} for value in sorted(unknown)]
        aligned = source.set_index('image_id').reindex(result.image_id)
        for field in ('patient_id', 'project', 'project_site'):
            if field in aligned:
                supplied = aligned[field].notna()
                if (supplied.to_numpy() & aligned[field].ne(result[field].to_numpy()).to_numpy()).any():
                    raise ValueError(f'{name} metadata conflict: {field}')
        result[name+'_grade'] = aligned.image_grading.to_numpy()
        result[name+'_original_grade'] = aligned.image_grading.to_numpy()
        result[name+'_source_filename'] = aligned.image_filename.to_numpy()
    changes = []
    snapshot = pinned_table(profile['ratings'], excel=True)
    unknown = set(snapshot.image_id)-set(result.image_id)
    if unknown - allowed:
        raise ValueError('Unapproved unmatched rating snapshot IDs')
    aligned = snapshot.set_index('image_id').reindex(result.image_id)
    if not np.allclose(pd.to_numeric(aligned.ganaav_grade), result.ganaav_grade, equal_nan=True):
        raise ValueError('Historical ganaav ratings disagree with original source')
    for name in canonical_graders(profile['graders']):
        values = pd.to_numeric(aligned[name], errors='raise').to_numpy(dtype=float)
        if name == 'shane_grade':
            changed = ~np.isclose(result.shane_grade, values, equal_nan=True)
            changes = [{'image_id': result.iloc[i].image_id, 'original': float(result.iloc[i].shane_grade),
                        'selected': float(values[i]), 'split': result.iloc[i].split} for i in np.flatnonzero(changed)]
        result[name] = values
    if len(changes) != profile['expected_shane_changes']:
        raise ValueError('Selected rating-version change count differs')
    ratings_array(result, profile['graders'])
    result['image_filename'] = result.image_path
    result = canonical_columns(result)
    validate_records(result, check_files=True)
    return result, exclusions, changes


def build_datasets(config_path, verify_existing=False, output_dir=None):
    build = read_config(config_path)
    if output_dir is not None:
        build['output_dir'] = str(output_dir)
    code = ROOT / build['output_dir'] / build['profiles'][0]['name'] / '_generation_code'
    if not verify_existing:
        for profile in build['profiles']:
            directory = ROOT / build['output_dir'] / profile['name']
            if directory.exists():
                raise FileExistsError(directory)
        code.mkdir(parents=True, exist_ok=False)
        for name in ('dataset.py', 'generate_datasets.py', 'utils.py'):
            shutil.copyfile(Path(__file__).with_name(name), code / name)
    summary = []
    for profile in build['profiles']:
        directory = ROOT / build['output_dir'] / profile['name']
        frame, exclusions, changes = join_sources(build, profile)
        directory.mkdir(parents=True, exist_ok=True)
        for filename, (kind, selected) in VARIANTS.items():
            graders = canonical_graders(selected or profile['graders'])
            dataset = generate_targets(frame, kind, graders, 'available' if selected is None else 'require_all')
            metadata = {'schema': 'autoqc_dataset_v1', 'target_kind': kind, 'graders': graders,
                        'mean_missing_policy': 'available' if selected is None else 'require_all',
                        'label_rule': RULES[kind], 'grade_order': [1, 2, 3, 4, 5],
                        'ganaav_identity': 'confirmed by project owner; previousStudent is the historical alias',
                        'shane_source_authority': build['shane_authority'], 'rating_profile': profile['name'],
                        'sources': {key: build[key] for key in ('ganaav', 'shane', 'split')},
                        'selected_rating_snapshot': profile['ratings'], 'review_changes': changes,
                        'held_out_project': 'VISUS-NHP', 'source_exclusions': exclusions,
                        'generation_code_snapshot': str(code),
                        'generator_sha256': sha256(__file__), 'dataset_module_sha256': sha256(Path(__file__).with_name('dataset.py'))}
            path = directory / filename
            if verify_existing and path.exists():
                saved, _ = load_workbook(path)
                if logical_hash(saved.fillna({'exclusion_reason': ''})) != logical_hash(dataset):
                    raise ValueError(f'Regeneration differs: {path}')
            else:
                save_workbook(path, dataset, metadata)
                load_workbook(path, check_files=True)
            summary.append({'path': str(path.relative_to(ROOT)), 'images': len(dataset), 'eligible': int(dataset.eligible.sum()),
                            'logical_sha256': logical_hash(dataset), 'file_sha256': sha256(path)})
        if not verify_existing:
            pd.DataFrame(exclusions).to_csv(directory / 'exclusions.csv', index=False)
            pd.DataFrame(changes, columns=['image_id', 'original', 'selected', 'split']).to_csv(directory / 'rating_changes.csv', index=False)
    if not verify_existing:
        write_json(ROOT / build['output_dir'] / 'build_results.json', summary)
    print(f'{len(summary)} workbooks {"verified" if verify_existing else "generated"}; no splits regenerated')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/datasets.yaml'))
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='New output directory; never overwrites an existing build')
    args = parser.parse_args()
    build_datasets(args.config, args.verify_existing, args.output_dir)


if __name__ == '__main__':
    main()
