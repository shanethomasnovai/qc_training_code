"""Verify the supplied baseline and generate a review diff; never run experiments."""
import argparse
import ast
from collections import Counter
import contextlib
import csv
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_ARCHIVE = '6ccfb237b8358e28caaa9a0cf152b9aa52dc3352371fbcda5d73fc6c185a6c34'
PREFIX = 'ds-auto-qc-Multi-class-qc/'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definition(tree, name, kind):
    return next(n for n in tree.body if isinstance(n, kind) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def assignments(rows, key, mode):
    result = {}
    for row in rows:
        ident = Path(row[key].replace('\\', '/')).stem if key == 'image_filename' else row[key]
        if ident in result:
            raise ValueError('Duplicate baseline/comparison image ID')
        result[ident] = (row['patient_id'], 'validation' if row[mode] == 'val' else row[mode])
    return result


def split_summary(rows):
    return {split: {'images': sum(value[1] == split for value in rows.values()),
                    'patients': len({value[0] for value in rows.values() if value[1] == split})}
            for split in ('train', 'validation', 'test')}


def verify(archive_path, refresh_diff=False):
    if digest(archive_path) != EXPECTED_ARCHIVE:
        raise ValueError('Reference archive differs from the audited supplied baseline')
    runtime_before = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'auto_qc').rglob('*.py')}
    readme_before = digest(ROOT / 'README.md')
    original = HERE / 'original'
    with zipfile.ZipFile(archive_path) as archive:
        paths = [name for name in archive.namelist() if name.startswith(PREFIX+'auto_qc/CNN/') and name.endswith('.py')]
        for name in paths:
            relative = Path(name[len(PREFIX):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive entry')
            target = original / relative
            data = archive.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Original reference copy has changed: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
        baseline_rows = list(csv.DictReader(io.StringIO(archive.read(PREFIX+'data/datasets/multi_class/multi_class_qc_dataset.csv').decode('utf-8-sig'))))
        comment = archive.comment.decode('utf-8', errors='replace')
    report = {'archive_path': str(archive_path), 'archive_sha256': EXPECTED_ARCHIVE,
              'original_commit': 'unknown; archive comment is not independently verified Git provenance',
              'unverified_archive_comment': comment, 'scope': 'CNN source modules; no reference training/testing CLI executed',
              'original_source_sha256': {str(p.relative_to(original)): digest(p) for p in sorted(original.rglob('*.py'))},
              'current_source_sha256': runtime_before}
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from torchvision import transforms
    from auto_qc.CNN import dataset as current_dataset
    from auto_qc.CNN.model import IQAnet

    old_model_path = original / 'auto_qc/CNN/model.py'
    old_dataset_path = original / 'auto_qc/CNN/dataset.py'
    old_model = load_module('verified_original_iqanet', old_model_path)
    old_dataset = load_module('verified_original_dataset', old_dataset_path)
    old_class = definition(ast.parse(old_model_path.read_text()), 'IQAnet', ast.ClassDef)
    new_class = definition(ast.parse((ROOT / 'auto_qc/CNN/model.py').read_text()), 'IQAnet', ast.ClassDef)
    old_init = definition(old_class, '__init__', ast.FunctionDef)
    new_init = definition(new_class, '__init__', ast.FunctionDef)
    old_forward = definition(old_class, 'forward', ast.FunctionDef)
    new_features = definition(new_class, 'forward_features', ast.FunctionDef)
    cut = next(i for i,n in enumerate(old_forward.body) if isinstance(n, ast.Assign) and
               isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'fc3_d')
    report['checks'] = {
        'original_iqanet_constructor_ast_equal': dump(old_init) == dump(new_init),
        'feature_operations_ast_equal': [dump(n) for n in old_forward.body[:cut]] == [dump(n) for n in new_features.body[:-1]],
    }
    for name, value in report['checks'].items():
        if not value:
            raise AssertionError(name)
    model_cases = []
    with torch.random.fork_rng(devices=[]):
        x = torch.linspace(-1, 1, 2*256*256).reshape(2, 1, 256, 256)
        for bn in (False, True):
            for sigmoid in (False, True):
                torch.manual_seed(42)
                before = old_model.IQAnet(num_classes=5, batch_normalization=bn, output_sigmoid=sigmoid)
                torch.manual_seed(42)
                after = IQAnet(num_classes=5, batch_normalization=bn, output_sigmoid=sigmoid)
                assert set(before.state_dict()) == set(after.state_dict())
                assert all(torch.equal(value, after.state_dict()[key]) for key,value in before.state_dict().items())
                for training in (False, True):
                    before.train(training)
                    after.train(training)
                    with torch.inference_mode():
                        torch.manual_seed(101)
                        expected = before(x)
                        torch.manual_seed(101)
                        actual = after(x)
                    assert torch.equal(expected, actual)
                    model_cases.append({'batch_normalization': bn, 'output_sigmoid': sigmoid,
                                        'training_mode_forward_only': training, 'exact_output_equality': True})
    report['model_cases'] = model_cases
    old_training = ast.parse((original / 'auto_qc/CNN/training.py').read_text())
    transform = next(n.value for n in ast.walk(old_training) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'transform' for t in n.targets))
    active_transform = definition(ast.parse((ROOT / 'auto_qc/CNN/dataset.py').read_text()), 'image_transform', ast.FunctionDef).body[0].value
    assert dump(transform) == dump(active_transform)
    report['checks']['preprocessing_transform_ast_equal'] = True
    # Exact syntax was checked above before evaluating this constructor expression.
    original_transform = eval(compile(ast.Expression(transform), '<verified original transform>', 'eval'), {'transforms': transforms})
    with tempfile.TemporaryDirectory(prefix='autoqc_baseline_parity_') as temp:
        image_path = Path(temp) / 'DARC-P2_SITE_001_OD_V1_2020_1.1_0.png'
        rgb = (np.arange(317*509*3, dtype=np.uint32).reshape(317, 509, 3) % 256).astype(np.uint8)
        Image.fromarray(rgb).save(image_path)
        frame = pd.DataFrame({'image_filename': [str(image_path)], 'image_grading': [3], 'mode': ['train']})
        before_data = old_dataset.ImageQualityDataset(frame, original_transform, label_mapping={k:k-1 for k in range(1,6)})
        with contextlib.redirect_stdout(io.StringIO()):
            patched, _ = old_dataset.subsample_subimages(before_data, method='all', df=frame)
        expected = torch.stack([item[0] for item in patched])
        actual = current_dataset.extract_patches(image_path)
        assert torch.equal(expected, actual)
        records = pd.DataFrame({'image_id': [image_path.stem], 'image_path': [str(image_path)], 'split': ['train'], 'ganaav_grade': [3]})
        records = current_dataset.generate_targets(records, 'hard_ganaav', ['ganaav_grade'])
        data = current_dataset.ImageQualityDataset(records, 'hard_ganaav', augment=True)
        flip = transforms.RandomHorizontalFlip(p=1.)
        assert len(data) == 18
        for i in range(9):
            assert torch.equal(data[i][0], expected[i])
            assert torch.equal(data[i+9][0], flip(expected[i]))
            assert int(data[i][1]) == int(patched[i][1])
    report['checks'].update(nine_patch_pixels_and_order_equal=True, eighteen_training_patches_including_flips_equal=True)
    with (ROOT / 'prepared_visus_nhp/manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        later_rows = list(csv.DictReader(handle))
    old = assignments(baseline_rows, 'image_filename', 'mode')
    new = assignments(later_rows, 'image_id', 'split')
    assert set(old) == set(new)
    report['split_comparison'] = {'same_full_image_ids': True, 'image_count': len(old),
        'original': split_summary(old), 'current_saved_assignments': split_summary(new),
        'changed_assignments': sum(old[key][1] != new[key][1] for key in old),
        'patient_ids_unchanged': all(old[key][0] == new[key][0] for key in old)}
    assert runtime_before == {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'auto_qc').rglob('*.py')}
    assert readme_before == digest(ROOT / 'README.md')
    report.update(runtime_sources_unchanged_by_verification=True, root_readme_unchanged=True,
                  real_images_decoded=False, optimizer_steps=0, experiment_training_started=False,
                  statement='Numerical parity applies to original IQAnet and synthetic preprocessing, not to the new heads, targets or evaluation protocol')
    old_files = {str(p.relative_to(original)): p for p in original.rglob('*.py')}
    new_files = {str(p.relative_to(ROOT)): p for p in (ROOT / 'auto_qc/CNN').glob('*.py')}
    diffs, changed = [], []
    for name in sorted(set(old_files) | set(new_files)):
        left = old_files[name].read_text(encoding='utf-8-sig').splitlines() if name in old_files else []
        right = new_files[name].read_text(encoding='utf-8-sig').splitlines() if name in new_files else []
        if left == right:
            continue
        path = name.replace('\\', '/')
        diffs.append(f'diff --git a/{path} b/{path}\n')
        if name not in old_files:
            diffs.append('new file mode 100644\n')
        elif name not in new_files:
            diffs.append('deleted file mode 100644\n')
        diffs.extend(difflib.unified_diff([s+'\n' for s in left], [s+'\n' for s in right],
                     fromfile='a/'+path if name in old_files else '/dev/null',
                     tofile='b/'+path if name in new_files else '/dev/null'))
        changed.append({'path': path, 'kind': 'added' if name not in old_files else 'not_carried_forward' if name not in new_files else 'modified'})
    report['changed_cnn_files'] = changed
    patch = ''.join(diffs)
    destination = HERE / 'changes.patch'
    if refresh_diff or not destination.exists():
        destination.write_text(patch, encoding='utf-8')
    elif destination.read_text(encoding='utf-8') != patch:
        raise ValueError('Code has changed; use --refresh-diff to update the review snapshot')
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'checks': report['checks'], 'model_forward_cases': len(model_cases),
                      'split_comparison': report['split_comparison'], 'runtime_changed': False,
                      'patch': str(destination)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(r'C:\Users\Shane\Downloads\ds-auto-qc-Multi-class-qc (1).zip'))
    parser.add_argument('--refresh-diff', action='store_true')
    args = parser.parse_args()
    verify(args.archive, args.refresh_diff)
