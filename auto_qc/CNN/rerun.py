"""Prepare verified historical configurations, check them, and run a fail-fast queue."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .dataset import canonical_graders, load_workbook
from .generate_datasets import build_datasets
from .training import preflight, train
from .utils import ROOT, read_config, sha256, write_json

GRADERS = ['ganaav_grade', 'shane_grade', 'reviewer1_grade', 'reviewer2_grade', 'reviewer3_grade']


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)).replace('\\', '/') if path.is_relative_to(ROOT) else str(path)


def prepare(inventory):
    if (ROOT / 'experiments/rerun_queue.json').exists():
        raise FileExistsError('Rerun queue already exists; use check/run rather than overwriting it')
    rows = list(csv.DictReader(Path(inventory).open(encoding='utf-8')))
    completed = [r for r in rows if r['completion_status'] == 'completed_artifacts_present']
    profiles, snapshots = [], {}
    for path in sorted((ROOT / 'runs').glob('*/ratings_input.xlsx')):
        snapshots.setdefault(sha256(path), path)
    configurations = {}
    for row in completed:
        path = Path(row.get('config_path', ''))
        if path.is_file():
            configurations[row['experiment_id']] = read_config(path)
    hashes = sorted({c.get('ratings_sha256', c.get('input_ratings_sha256')) for c in configurations.values()} - {None})
    for digest in hashes:
        if digest not in snapshots:
            continue
        snapshot = pd.read_excel(snapshots[digest], sheet_name='Grades')
        raw = pd.read_csv(ROOT / 'data/shane-image-file-listing.csv')
        raw['image_id'] = raw.image_filename.map(lambda p: Path(str(p).replace('\\', '/')).stem)
        aligned = snapshot.set_index('image_id').loc[raw.image_id]
        changes = int((aligned.shane_grade.to_numpy() != raw.image_grading.to_numpy()).sum())
        profiles.append({'name': 'ratings_'+digest[:12], 'ratings': {'path': relative(snapshots[digest]), 'sha256': digest, 'sheet': 'Grades'},
                         'graders': GRADERS, 'expected_shane_changes': changes})
    split = ROOT / 'prepared_visus_nhp/manifest.csv'
    ganaav = ROOT / 'data/baseline_annotation_20261023/multi_class_qc_dataset.csv'
    shane = ROOT / 'data/shane-image-file-listing.csv'
    exclusions = read_config(ROOT / 'prepared_visus_nhp/split_report.json')['shane_only_ids']
    build = {'schema': 'autoqc_dataset_build_v1', 'ganaav': {'path': relative(ganaav), 'sha256': sha256(ganaav)},
             'shane': {'path': relative(shane), 'sha256': sha256(shane)},
             'shane_authority': 'This is the raw version used by the recorded historical snapshots; earliest/original authority among differing copies remains unresolved',
             'split': {'path': relative(split), 'sha256': sha256(split)}, 'allowed_unmatched_ids': exclusions,
             'output_dir': 'model_dir/historical_datasets', 'profiles': profiles,
             'original_archive_sha256': '6ccfb237b8358e28caaa9a0cf152b9aa52dc3352371fbcda5d73fc6c185a6c34',
             'original_commit': 'unknown', 'ganaav_identity_confirmation': 'project owner, 2026-10-30'}
    (ROOT / 'configs').mkdir(exist_ok=True)
    write_json(ROOT / 'configs/historical_datasets.yaml', build)
    build_datasets(ROOT / 'configs/historical_datasets.yaml')
    queue, seen = [], {}
    for number, row in enumerate(completed, 1):
        historical = row['experiment_id']
        c = configurations.get(historical, {})
        entry = {'historical_experiment': historical, 'status': 'blocked', 'reasons': []}
        if not c:
            entry['reasons'] = ['No saved effective optimiser/loss/weighting configuration; folder names are not evidence']
            queue.append(entry)
            continue
        required = ['seed', 'batch_size', 'epochs', 'learning_rate', 'optimizer', 'optimizer_settings', 'reduce_lr',
                    'patience', 'batch_normalization', 'grader_columns', 'preprocessing', 'class_weights']
        entry['reasons'] = ['Unknown saved setting: '+key for key in required if key not in c]
        if c.get('reduce_lr') and 'scheduler_settings' not in c:
            entry['reasons'].append('Unknown saved scheduler settings')
        ratings_hash = c.get('ratings_sha256', c.get('input_ratings_sha256'))
        profile = next((p for p in profiles if p['ratings']['sha256'] == ratings_hash), None)
        if profile is None:
            entry['reasons'].append('Hash-matched ratings snapshot unavailable')
        if entry['reasons']:
            queue.append(entry)
            continue
        method = c.get('loss_selection', 'cross_entropy' if c.get('loss') == 'CrossEntropyLoss' else None)
        mapping = {'cross_entropy': ('five_logits', 'soft_ce', 'multigrader_soft_labels.xlsx'),
                   'continuous_cdw_ce': ('five_logits', 'continuous_cdw_ce', 'multigrader_mean_labels.xlsx'),
                   'soft_ordinal_coral': ('coral', 'ordinal_bce', 'ordinal_targets.xlsx'),
                   'or_soft': ('or_soft', 'ordinal_bce', 'ordinal_targets.xlsx'),
                   'or_soft_binary': ('or_soft_binary', 'joint_bce', 'ordinal_targets.xlsx'),
                   'or_soft_frozen_binary': ('frozen_binary', 'binary_bce', 'ordinal_targets.xlsx')}
        if method not in mapping:
            entry['reasons'] = ['Training interpretation unresolved: '+str(method)]
            queue.append(entry)
            continue
        head, loss_name, filename = mapping[method]
        dataset = ROOT / build['output_dir'] / profile['name'] / filename
        frame, meta = load_workbook(dataset)
        graders = canonical_graders(c['grader_columns'])
        if graders != meta['graders']:
            entry['reasons'] = ['Reviewer selection differs; a separate generated profile is required']
            queue.append(entry)
            continue
        historical_run = ROOT / 'runs' / historical
        cohort = pd.read_csv(historical_run / 'manifest.csv')
        if 'targets.csv' in {p.name for p in historical_run.iterdir()}:
            saved_targets = pd.read_csv(historical_run / 'targets.csv')
            saved_targets = saved_targets.set_index('image_id')
            candidate = frame.set_index('image_id').loc[saved_targets.index]
            columns = [f'target_above_grade_{k}' for k in range(1, 5)] if head != 'five_logits' else [f'target_grade_{k}' for k in range(1, 6)]
            if loss_name == 'continuous_cdw_ce':
                comparisons = [(candidate.target_mean_grade, saved_targets.continuous_target)]
            else:
                comparisons = [(candidate[name], saved_targets[name]) for name in columns]
            if not all(np.allclose(left, right, rtol=0, atol=1e-8, equal_nan=True) for left, right in comparisons):
                entry['reasons'] = ['Generated targets disagree with historical saved targets']
                queue.append(entry)
                continue
        identity = frame.set_index('image_id').loc[cohort.image_id]
        for field in ('patient_id', 'project', 'split'):
            if not identity[field].reset_index(drop=True).equals(cohort[field].reset_index(drop=True)):
                raise ValueError(f'Historical assignment conflict: {historical}, {field}')
        model = {'head': head, 'batch_normalization': c['batch_normalization'], 'output_sigmoid': False, 'dropout_p': .5,
                 'ker_size': 5, 'n_kers': 64, 'n1_nodes': 600, 'n2_nodes': 400, 'num_grades': 5}
        if head == 'frozen_binary':
            source = Path(c['source_checkpoint_path'])
            if not source.is_file() or sha256(source) != c['source_checkpoint_sha256']:
                entry['reasons'] = ['Frozen source missing or checkpoint hash changed']
                queue.append(entry)
                continue
            model['source'] = {'checkpoint': str(source), 'sha256': sha256(source), 'architecture': dict(model, head='or_soft')}
        options = {key: value for key, value in c['optimizer_settings'].items() if key in
                   ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused', 'momentum', 'dampening', 'nesterov')}
        scheduler = {key: value for key, value in c.get('scheduler_settings', {}).items() if key != 'name'}
        weights = c.get('effective_class_weights_float32', c['class_weights'])
        weight_spec = {'mode': 'none'} if weights is None else {'mode': 'explicit', 'values': weights, 'grade_order': [1, 2, 3, 4, 5], 'source': c.get('class_weight_source')}
        subsets = {}
        for name in ('shared_validation', 'baseline_validation', 'reference_shared_validation'):
            subset_file = historical_run / (name+'_targets.csv')
            if subset_file.is_file():
                subsets[name] = pd.read_csv(subset_file).image_id.tolist()
        if loss_name == 'continuous_cdw_ce' and (c.get('alpha') is None or c['alpha'] <= 0):
            entry['reasons'] = ['Recorded CDW alpha unavailable']
            queue.append(entry)
            continue
        identifier = f'rerun_{number:03}_{method}_seed{c["seed"]}'
        config = {'schema': 'autoqc_cnn_config_v1', 'experiment_id': identifier, 'status': 'ready',
                  'historical': {'run': relative(historical_run), 'config_sha256': sha256(historical_run / 'run_config.json'),
                                 'manifest_sha256': sha256(historical_run / 'manifest.csv'),
                                 'classification': 'corrected implementation rerun; same saved patient assignments and historical target records',
                                 'corrections': ['cloned best checkpoint', 'explicit class order', 'seeded RNGs', 'split before patches', 'architecture-aware reload', 'unique outputs'],
                                 'source_version': c.get('code_version', 'unknown')},
                  'dataset': {'path': relative(dataset), 'sha256': sha256(dataset), 'target_kind': meta['target_kind'], 'graders': graders,
                              'target_definition': meta['label_rule'], 'profile': profile['name']},
                  'model': model, 'loss': {'name': loss_name, 'alpha': c.get('alpha'), 'binary_loss_weight': c.get('binary_loss_weight', 1.),
                                          'class_weights': weight_spec, 'threshold_weights': [1, 1, 1, 1] if head != 'five_logits' else None,
                                          'reduction': c.get('loss_reduction', 'example mean; threshold sum for ordinal'),
                                          'epoch_aggregation': 'batch_size_weighted_batch_means' if loss_name == 'hard_ce' else 'example_mean'},
                  'training': {'seed': c['seed'], 'learning_rate': c['learning_rate'], 'batch_size': c['batch_size'], 'batch_size_unit': 'patches', 'epochs': c['epochs'],
                               'patience': c['patience'], 'optimizer': c['optimizer'], 'optimizer_settings': options,
                               'scheduler': {'enabled': c['reduce_lr'], 'settings': scheduler}, 'device': 'auto'},
                  'preprocessing': dict(c['preprocessing'], horizontal_flip=True),
                  'cohort': {'training_image_ids': cohort.loc[cohort.split.eq('train'), 'image_id'].tolist(),
                             'monitor_image_ids': cohort.loc[cohort.split.eq('validation'), 'image_id'].tolist(),
                             'baseline_definition': c.get('baseline_metadata'), 'matched_sampling': c.get('matched_baseline_metadata')},
                  'selection': {'criterion': 'strict minimum binary patch validation BCE' if head == 'frozen_binary' else 'strict minimum primary patch validation loss',
                                'monitor': 'binary' if head == 'frozen_binary' else 'ordinal' if head != 'five_logits' else 'configured primary loss',
                                'early_stopping': 'patience epochs without strict improvement'},
                  'evaluation': {'enable_test': False, 'held_out_project': 'VISUS-NHP', 'image_ids': frame.loc[frame.split.eq('validation'), 'image_id'].tolist(),
                                 'subsets': subsets, 'unit': 'whole_image', 'aggregation': 'mean of patch grade probabilities; mean of patch sigmoid for binary',
                                 'grade_decision': 'argmax, lower grade for ties', 'binary_decision': 'grade argmax primary; sum>=.5 and dedicated>=.5 separate',
                                 'reference': 'available-grader mode; lower grade for ties; binary majority with ties acceptable',
                                 'projection': 'inference_only_unweighted_least_squares_descending_pava_v1' if head in ('or_soft', 'or_soft_binary', 'frozen_binary') else 'none'}}
        scientific = {k: v for k, v in config.items() if k not in ('experiment_id', 'historical')}
        signature = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
        entry.update(config=f'configs/reruns/{identifier}.yaml', experiment_id=identifier, signature=signature)
        if signature in seen:
            entry.update(status='duplicate_excluded', duplicate_of=seen[signature])
        else:
            entry['status'] = 'ready'
            seen[signature] = historical
        config['status'] = 'ready' if entry['status'] == 'ready' else 'duplicate_excluded'
        config['duplicate_of'] = entry.get('duplicate_of')
        target = ROOT / entry['config']
        target.parent.mkdir(exist_ok=True)
        write_json(target, config)
        queue.append(entry)
    materialize_blocked_configs(queue, configurations)
    for entry in queue:
        entry['config_sha256'] = sha256(ROOT / entry['config'])
    write_json(ROOT / 'experiments/rerun_queue.json', {'schema': 1, 'inventory_sha256': sha256(inventory),
               'completed_training_records': len(completed), 'entries': queue})
    template = read_config(ROOT / next(e['config'] for e in queue if e['status'] == 'ready'))
    current = read_config(ROOT / 'configs/datasets.yaml')
    prepare_diagnostic_configs(template, current['profiles'][0], current['output_dir'])
    print(json.dumps({state: sum(e['status'] == state for e in queue) for state in ('ready', 'blocked', 'duplicate_excluded')}))
    return queue


def materialize_blocked_configs(entries, saved):
    for number, entry in enumerate(entries, 1):
        if entry['status'] != 'blocked':
            continue
        name = entry['historical_experiment'].split('_')[0].replace('previousStudent', 'ganaav')
        identifier = f'blocked_{number:03}_{name}'
        entry['config'] = f'configs/reruns/{identifier}.yaml'
        write_json(ROOT / entry['config'], {'schema': 'autoqc_cnn_config_v1', 'experiment_id': identifier,
                   'status': 'blocked', 'blocked_reasons': entry['reasons'],
                   'historical_run': 'runs/'+entry['historical_experiment'],
                   'known_saved_settings': saved.get(entry['historical_experiment'], 'See preserved model_params.csv; remaining settings unknown'),
                   'unknown_setting_policy': 'do not fill from folder names or current defaults'})


def prepare_diagnostic_configs(template, profile, output_dir='datasets'):
    """Explicit smoke settings, never inferred settings for undocumented old runs."""
    from .utils import class_weights
    checks = [
        ('hard_ganaav', 'ganaav_labels.xlsx', 'hard_ce', False),
        ('hard_ganaav_weighted', 'ganaav_labels.xlsx', 'hard_ce', True),
        ('hard_shane', 'shane_labels.xlsx', 'hard_ce', False),
        ('ceiling_mean', 'ganaav_shane_ceiling_labels.xlsx', 'hard_ce', False),
        ('exact_pair_mean', 'ganaav_shane_mean_labels.xlsx', 'continuous_cdw_ce', False),
        ('exact_multigrader_mean', 'multigrader_mean_labels.xlsx', 'continuous_cdw_ce', False),
        ('soft_categorical', 'multigrader_soft_labels.xlsx', 'soft_ce', False),
        ('soft_categorical_weighted', 'multigrader_soft_labels.xlsx', 'soft_ce', True),
    ]
    for name, filename, loss, weighted in checks:
        config = copy.deepcopy(template)
        dataset = ROOT / output_dir / profile['name'] / filename
        frame, meta = load_workbook(dataset)
        config.update(experiment_id='check_'+name, diagnostic_only=True,
                      historical={'classification': 'diagnostic only; settings borrowed explicitly from a ready template, not a recovered historical experiment'})
        config['model'] = {'head': 'five_logits', 'batch_normalization': False, 'output_sigmoid': False, 'dropout_p': .5,
                           'ker_size': 5, 'n_kers': 64, 'n1_nodes': 600, 'n2_nodes': 400, 'num_grades': 5}
        config['dataset'] = {'path': relative(dataset), 'sha256': sha256(dataset), 'target_kind': meta['target_kind'],
                             'graders': meta['graders'], 'target_definition': meta['label_rule'], 'profile': profile['name']}
        config['loss'] = {'name': loss, 'alpha': 1. if loss == 'continuous_cdw_ce' else None,
                          'class_weights': {'mode': 'none'}, 'threshold_weights': None,
                          'reduction': 'mean examples; hard CE weighted batches divide by target-weight sum',
                          'epoch_aggregation': 'global_target_weight_sum' if loss == 'hard_ce' else 'example_mean'}
        if weighted:
            reference = frame.loc[frame.split.eq('train')].copy()
            if 'target_class_index' not in reference:
                reference['target_class_index'] = reference.shane_grade-1
            values = class_weights(reference, {'mode': 'inverse_frequency'}).tolist()
            config['loss']['class_weights'] = {'mode': 'explicit', 'values': values, 'source': 'training-image inverse frequency; Shane reference for soft CE'}
        config['training']['seed'] = 42
        config['cohort'] = {'training_image_ids': frame.loc[frame.split.eq('train'), 'image_id'].tolist(),
                            'monitor_image_ids': frame.loc[frame.split.eq('validation'), 'image_id'].tolist()}
        config['evaluation']['image_ids'] = frame.loc[frame.split.eq('validation'), 'image_id'].tolist()
        config['evaluation']['subsets'] = {}
        config['evaluation']['projection'] = 'none'
        config['selection'] = {'criterion': 'strict minimum configured primary patch validation loss', 'monitor': 'configured primary loss'}
        destination = ROOT / f'configs/checks/{name}.yaml'
        destination.parent.mkdir(exist_ok=True)
        write_json(destination, config)


def run_queue(mode, queue_path, resume=None):
    queue = read_config(queue_path)
    entries = [e for e in queue['entries'] if e['status'] == 'ready']
    for entry in entries:
        if entry.get('config_sha256') and sha256(ROOT / entry['config']) != entry['config_sha256']:
            raise ValueError(f'Queued configuration changed: {entry["config"]}; run edited configurations individually')
    if mode == 'check':
        for entry in entries:
            train(ROOT / entry['config'], check_only=True)
        print(f'{len(entries)} configurations passed')
        return
    if mode == 'run' and resume is None:
        for path in (ROOT / 'model_dir').glob('queue_run_*/queue_status.json'):
            previous = read_config(path)
            if previous['status'] == 'running':
                raise RuntimeError(f'A rerun queue is already marked running: {path.parent}; check its owner_pid before resuming an interrupted queue')
    if mode == 'smoke':
        # Distinct loss/head paths plus both historical CE weighting variants.
        selected, seen = [], set()
        for entry in entries:
            config = read_config(ROOT / entry['config'])
            key = (config['model']['head'], config['loss']['name'], config['loss']['class_weights']['mode'])
            if key not in seen:
                selected.append(entry)
                seen.add(key)
        entries = selected
        # Hard-label and ceiling paths are new controlled comparisons, not recovered historical settings.
        entries += [{'config': str(p.relative_to(ROOT))} for p in sorted((ROOT / 'configs/checks').glob('*.yaml'))]
    directory = Path(resume).resolve() if resume else ROOT / 'model_dir' / ('queue_'+mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    if resume:
        state = read_config(directory / 'queue_status.json')
        accepted = [sha256(queue_path), *queue.get('compatible_previous_queue_sha256', [])]
        if state['queue_sha256'] not in accepted or state['mode'] != mode:
            raise ValueError('Cannot resume a different queue/mode')
        if state['queue_sha256'] != sha256(queue_path):
            for item in queue.get('dataset_path_relocations', []):
                if item['config'] not in {entry['config'] for entry in entries}:
                    continue  # An explicitly deferred experiment is not resumed.
                before = read_config(item['original_config'])
                after = read_config(ROOT / item['config'])
                if sha256(item['archived_path']) != item['sha256']:
                    raise ValueError('Archived dataset hash changed')
                before['dataset']['path'] = after['dataset']['path']
                if before != after:
                    raise ValueError('Queue compatibility permits dataset-path relocation only')
            state['queue_sha256'] = sha256(queue_path)
        deferred = {entry['config']: entry for entry in queue['entries'] if entry['status'] == 'deferred_by_user'}
        skipped = [path for path in state['remaining'] if path in deferred]
        if skipped:
            state.setdefault('skipped', []).extend(
                {'config': path, 'reason': deferred[path]['reasons']} for path in skipped)
            state['remaining'] = [path for path in state['remaining'] if path not in deferred]
        for completed in state['completed']:
            status = read_config(Path(completed['run']) / 'run_status.json')
            if status['status'] != 'completed' or not status['required_outputs_verified']:
                raise ValueError('A previously completed queue entry no longer verifies')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        state = {'status': 'running', 'mode': mode, 'owner_pid': os.getpid(), 'queue_sha256': sha256(queue_path), 'completed': [], 'remaining': [e['config'] for e in entries]}
    write_json(directory / 'queue_status.json', state)
    try:
        for entry in entries:
            if entry['config'] not in state['remaining']:
                continue
            state.update(status='running', active=entry['config'])
            write_json(directory / 'queue_status.json', state)
            result = train(ROOT / entry['config'], output_root=directory.relative_to(ROOT), smoke=mode == 'smoke')
            status = read_config(result / 'run_status.json')
            if status['status'] != 'completed' or not status['required_outputs_verified']:
                raise RuntimeError('Required outputs not verified; stopping queue')
            state['completed'].append({'config': entry['config'], 'run': str(result)})
            state['remaining'].remove(entry['config'])
            write_json(directory / 'queue_status.json', state)
        state.update(status='completed', active=None)
    except BaseException as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(directory / 'queue_status.json', state)
    print(f'Queue {state["status"]}: {directory}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'smoke', 'run', 'status'])
    parser.add_argument('--inventory', type=Path, default=Path('experiments/historical_inventory.csv'))
    parser.add_argument('--queue', type=Path, default=Path('experiments/rerun_queue.json'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.inventory)
    elif args.action == 'status':
        paths = sorted((ROOT / 'model_dir').glob('queue_*/queue_status.json'))
        for path in paths:
            state = read_config(path)
            print(path, state['status'], 'completed', len(state['completed']), 'remaining', len(state['remaining']), 'active', state.get('active'))
    else:
        run_queue(args.action, args.queue, args.resume)


if __name__ == '__main__':
    main()
