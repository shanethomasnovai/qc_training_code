"""Build five label tables from reviewed ratings and a saved patient split.

Flow: resolve the recipe -> check/join sources -> write labels -> save manifest.
Use --verify-existing to compare against an existing build without changing it.
Training target encoding and workbook loading live in data/labels.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from auto_qc.CNN.utils import ROOT, code_version, read_config, sha256, write_json
from data.labels import (canonical_columns, canonical_graders, generate_targets,
                         image_key, load_workbook, ratings_array, validate_records)
from data.tools import LOCAL_FOLDERS, local_paths, require_file, root_input, SOURCE_FILENAMES

VARIANTS = {
    'ganaav_labels.xlsx': ('hard_ganaav', ['ganaav_grade']),
    'shane_labels.xlsx': ('hard_shane', ['shane_grade']),
    'shane_ganaav_average_rounded_labels.xlsx': ('ceiling_mean', ['ganaav_grade', 'shane_grade']),
    'shane_ganaav_average_labels.xlsx': ('exact_mean', ['ganaav_grade', 'shane_grade']),
    'multigrader_labels.xlsx': ('ordinal', None),
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
    """Read a source only after its file hash matches the recipe."""
    path = require_file(ROOT / spec['path'], 'pinned label/split source')
    if sha256(path) != spec['sha256']:
        raise ValueError(f'Input hash mismatch: {path}')
    if excel:
        table = pd.read_excel(path, sheet_name=spec.get('sheet', 'Grades'))
    else:
        table = pd.read_csv(path)
    table = canonical_columns(table)
    if 'image_id' not in table:
        table['image_id'] = table.image_filename.map(image_key)
    if table.image_id.isna().any() or table.image_id.eq('').any() or table.image_id.duplicated().any():
        raise ValueError(f'Duplicate/missing full IDs in {path}')
    return table


def join_sources(build, profile):
    """Align grades to saved image IDs; never use grades carried in the split CSV."""
    split = pinned_table(build['split'])
    # Only assignment/identity columns, never the carrier manifest's rating values.
    fields = ['image_id', 'image_path', 'patient_id', 'project', 'project_site', 'split']
    result = split[fields].sort_values('image_id').reset_index(drop=True)
    if 'local_root' in build:
        result['image_path'] = result.image_path.map(lambda value: str(Path(build['local_root']) / LOCAL_FOLDERS['images'] / Path(str(value).replace('\\', '/')).name))
    ganaav = pinned_table(build['ganaav'])
    ratings = pinned_table(profile['ratings'], excel=True)
    allowed = set(build['allowed_unmatched_ids'])
    exclusions = []
    for name, source in [('ganaav', ganaav), ('ratings', ratings)]:
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
        if name == 'ganaav':
            result['ganaav_grade'] = aligned.image_grading.to_numpy()
    if set(result.image_id) - set(ratings.image_id):
        raise ValueError('Current ratings source does not cover the fixed image population')
    aligned = ratings.set_index('image_id').reindex(result.image_id)
    if not np.allclose(pd.to_numeric(aligned.ganaav_grade), result.ganaav_grade, equal_nan=True):
        raise ValueError('Ganaav ratings disagree with the supplied Ganaav source')
    for name in canonical_graders(profile['graders']):
        values = pd.to_numeric(aligned[name], errors='raise').to_numpy(dtype=float)
        result[name] = values
    ratings_array(result, profile['graders'])
    result['image_filename'] = result.image_path
    result = canonical_columns(result)
    validate_records(result, check_files=True)
    return result, exclusions


def write_labels(path, labels):
    """One worksheet, image filenames and labels only; no copied code or targets."""
    labels.to_excel(path, index=False, sheet_name='labels')


def read_build_config(config_path, output_dir=None):
    """Accept the build recipe or the normal training config's local_root."""
    build = read_config(config_path)
    if 'local_root' in build:
        root, paths = local_paths(build['local_root'], config_path)
        if 'data' in build:
            user = build
            if 'image_directory' in user['data']:
                raise ValueError('local_root conflicts with legacy data.image_directory; remove image_directory')
            build = read_config(Path(__file__).with_name('datasets.yaml'))
            relative = Path(user['data']['dataset'])
            if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2:
                raise ValueError('Dataset generation requires data.dataset: <profile>/<workbook.xlsx>')
            build['profiles'][0]['name'] = relative.parent.name
            if 'graders' in user['data']:
                build['profiles'][0]['graders'] = canonical_graders(user['data']['graders'])
        build['local_root'] = str(root)
        build['ganaav']['path'] = str(root_input(root, 'ganaav'))
        build['split']['path'] = str(root_input(root, 'split'))
        for profile in build['profiles']:
            profile['ratings']['path'] = str(root_input(root, 'ratings'))
        build['output_dir'] = paths['datasets']
    if output_dir is not None:
        override = Path(output_dir)
        if 'local_root' in build and not override.is_absolute():
            override = Path(config_path).resolve().parent / override
        build['output_dir'] = str(override.resolve() if 'local_root' in build else override)
    return build


def dataset_metadata(build, profile, exclusions):
    """Record where labels and patient assignments came from."""
    return {
        'grade_order': [1, 2, 3, 4, 5],
        'ganaav_identity': 'confirmed by project owner; previousStudent is the historical alias',
        'shane_source_authority': build['shane_authority'],
        'rating_profile': profile['name'],
        'sources': {'ganaav': build['ganaav'], 'ratings': profile['ratings'], 'split': build['split']},
        'held_out_project': 'VISUS-NHP', 'source_exclusions': exclusions,
        'generator_sha256': sha256(__file__),
        'dataset_module_sha256': sha256(ROOT / 'auto_qc/CNN/dataset.py'),
    }


def make_label_table(frame, kind, selected, graders):
    """Export filenames plus one label, or the individual multigrader ratings."""
    labels = pd.DataFrame({'image_filename': frame.image_path.map(lambda path: Path(path).name)})
    if selected is None:
        labels[graders] = frame[graders]
    else:
        targets = generate_targets(frame, kind, graders, 'require_all')
        column = 'target_mean_grade' if kind == 'exact_mean' else 'target_grade'
        labels['label'] = targets[column]
    return labels


def verify_manifest(directory, expected, local_root=None):
    """Check label provenance while allowing source files to move unchanged."""
    saved = read_config(directory / 'manifest.json')
    for key in ('variants', 'sources', 'rating_profile', 'source_exclusions'):
        if key == 'sources' and local_root is not None:
            for role, spec in expected['sources'].items():
                if saved['sources'][role]['sha256'] != spec['sha256']:
                    raise ValueError(f'Dataset provenance differs: sources.{role}.sha256')
        elif saved[key] != expected[key]:
            raise ValueError(f'Dataset provenance differs: {key}')
    for filename in VARIANTS:
        load_workbook(directory / filename, check_files=True, local_root=local_root)


def build_datasets(config_path, verify_existing=False, output_dir=None):
    """Build the five current label tables, or verify them without writing."""
    build = read_build_config(config_path, output_dir)
    if len(build['profiles']) != 1:
        raise ValueError('Active dataset generation requires one current ratings source')
    profile = build['profiles'][0]
    directory = ROOT / build['output_dir'] / profile['name']
    if not verify_existing and directory.exists():
        raise FileExistsError(directory)

    # The saved split supplies identity and assignments; only source ratings supply grades.
    frame, exclusions = join_sources(build, profile)
    if not verify_existing:
        directory.mkdir(parents=True, exist_ok=False)

    variants, summary = {}, []
    for filename, (kind, selected) in VARIANTS.items():
        graders = canonical_graders(selected or profile['graders'])
        labels = make_label_table(frame, kind, selected, graders)
        path = directory / filename
        if verify_existing:
            pd.testing.assert_frame_equal(pd.read_excel(path, sheet_name='labels'), labels, check_dtype=False)
        else:
            write_labels(path, labels)
        variants[filename] = {
            'target_kind': kind, 'graders': graders,
            'mean_missing_policy': 'available' if selected is None else 'require_all',
            'label_rule': RULES[kind], 'sha256': sha256(path),
        }
        summary.append({'path': str(path if 'local_root' in build else path.relative_to(ROOT)),
                        'images': len(labels), 'file_sha256': sha256(path)})

    manifest = dict(dataset_metadata(build, profile, exclusions), schema='autoqc_labels_v1',
                    variants=variants, ratings_file='multigrader_labels.xlsx',
                    graders=canonical_graders(profile['graders']), code_version=code_version())
    if verify_existing:
        verify_manifest(directory, manifest, build.get('local_root'))
    else:
        write_json(directory / 'manifest.json', manifest)
    print(f'{len(summary)} workbooks {"verified" if verify_existing else "generated"}; no splits regenerated')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'data/tools/datasets.yaml')
    parser.add_argument('--verify-existing', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='New output directory; never overwrites an existing build')
    args = parser.parse_args()
    build_datasets(args.config, args.verify_existing, args.output_dir)


if __name__ == '__main__':
    main()
