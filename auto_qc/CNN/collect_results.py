"""Collect completed reruns into one readable results folder, without moving runs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from .configuration import load_run_config, model_config_name, write_model_config
from .utils import run_config_path

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / 'model_dir' / 'results_support'
CURVE_STYLE = 'labelled_axes_v1'
FILES = {
    'checkpoint.pth': 'checkpoint.pth',
    'training_history.png': 'learning_curve.png',
    'dataset.xlsx': 'dataset.xlsx',
    'validation_predictions.csv': 'validation_predictions.csv',
    'validation_metrics.json': 'validation_metrics.json',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def condition(config):
    matched = config['cohort'].get('matched_sampling')
    if matched:
        return 'matched_' + matched['condition']
    source = config.get('historical', {}).get('run', '')
    if '_baseline_only_' in source:
        return 'baseline_only'
    if '_without_baseline_' in source:
        return 'non_baseline'
    return 'full_data' if len(config['cohort']['training_image_ids']) == 508 else 'cohort'


def refresh_curve(source, destination, support, receipt, config, status):
    """Render a review plot from the CSV; preserve the original run's PNG."""
    derivation = {'style': CURVE_STYLE, 'source_file': 'training_history.csv',
                  'source_sha256': digest(source / 'training_history.csv'),
                  'original_plot_sha256': digest(source / 'training_history.png'),
                  'selected_epoch': status['selected_epoch']}
    if receipt.get('learning_curve') == derivation:
        return
    import pandas as pd
    from .training import history_figure
    history_figure(pd.read_csv(source / 'training_history.csv'), destination / 'learning_curve.png',
                   config, status['selected_epoch'])
    receipt['learning_curve'] = derivation
    receipt['file_sha256']['learning_curve.png'] = digest(destination / 'learning_curve.png')
    receipt['note'] = 'Verified copies; learning_curve.png rendered from the original epoch CSV; original run artifacts retained'
    write_json(support / 'source_run.json', receipt)


def refresh_predictions(source, destination, support, receipt, config):
    """Simplify the review CSV without editing or relabelling source results."""
    filename = 'validation_predictions.csv'
    derivation = {'style': 'selected_columns_v1', 'source_file': filename,
                  'source_sha256': digest(source / filename),
                  'graders': config['dataset']['graders']}
    if receipt.get('predictions') == derivation:
        return
    import pandas as pd
    from .testing import prediction_summary
    # Preserve every retained CSV cell, including full precision and missing ratings.
    records = pd.read_csv(source / filename, dtype=str, keep_default_na=False)
    temporary = destination / (filename + '.tmp')
    prediction_summary(records, config['dataset']['graders']).to_csv(temporary, index=False)
    temporary.replace(destination / filename)
    receipt['predictions'] = derivation
    receipt['file_sha256'][filename] = digest(destination / filename)
    receipt['note'] = ('Original run artifacts retained; learning curve rendered from recorded losses; '
                       'predictions contain selected columns with unchanged values')
    write_json(support / 'source_run.json', receipt)


def refresh_dataset(source, destination, support, receipt, config):
    """Export labels for the actual training/evaluation cohorts, not internal fields.

    The original run snapshot remains the reproducible input for checkpoint loading.
    This single-sheet review copy is not a replacement training snapshot.
    """
    derivation = {'style': 'used_labels_v1', 'source_file': 'dataset.xlsx',
                  'source_sha256': digest(source / 'dataset.xlsx'),
                  'config_sha256': digest(run_config_path(source)),
                  'population': 'configured training and evaluation image IDs'}
    if receipt.get('dataset') == derivation:
        return
    import pandas as pd
    records = pd.read_excel(source / 'dataset.xlsx', sheet_name='dataset').set_index('image_id', drop=False)
    training = config['cohort']['training_image_ids']
    validation = config['evaluation']['image_ids']
    if not records.index.is_unique or len(set(training + validation)) != len(training + validation):
        raise ValueError('Duplicate or overlapping dataset image IDs')
    selected = records.loc[training + validation]
    if (not records.loc[training, 'split'].eq('train').all()
            or not records.loc[validation, 'split'].eq('validation').all()
            or not records.loc[training, 'training_eligible'].all()
            or not records.loc[validation, 'evaluation_eligible'].all()):
        raise ValueError('Dataset selections disagree with saved splits/eligibility')
    columns = ['image_id', 'split', *config['dataset']['graders']]
    labels = selected[columns].reset_index(drop=True)
    temporary = destination / 'dataset.tmp.xlsx'
    labels.to_excel(temporary, index=False, sheet_name='dataset', freeze_panes=(1, 0))
    temporary.replace(destination / 'dataset.xlsx')
    receipt['dataset'] = derivation
    receipt['file_sha256']['dataset.xlsx'] = digest(destination / 'dataset.xlsx')
    receipt['note'] = ('Original run artifacts retained; review curve, predictions and dataset '
                       'are derived views with unchanged recorded values')
    write_json(support / 'source_run.json', receipt)


def refresh_config(source, destination, support, receipt, config):
    """Replace the visible full config with a model-named settings summary."""
    derivation = {'style': 'model_settings_v2',
                  'source_sha256': digest(run_config_path(source)),
                  'dataset_sha256': digest(source / 'dataset.xlsx')}
    filename = model_config_name(config)
    if receipt.get('config_summary') == derivation and (destination / filename).is_file():
        return
    path = write_model_config(destination, config)
    receipt['config_summary'] = derivation
    receipt['file_sha256'][filename] = digest(path)
    receipt['source_files'][filename] = str(run_config_path(source).relative_to(source))
    write_json(support / 'source_run.json', receipt)


def export_run(source, output):
    source = Path(source).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status, config = read(source / 'run_status.json'), load_run_config(source)
    if status['status'] != 'completed' or status.get('smoke') or not status.get('required_outputs_verified'):
        raise ValueError(f'Only verified completed full runs can be collected: {source}')
    identifier = config['experiment_id']
    tag = (f'{identifier.split("_")[1]}_{source.name.split("_")[-1]}'
           if identifier.startswith('rerun_') else source.name.removeprefix(identifier + '_'))
    label = f'{config["model"]["head"]}__{condition(config)}__seed{config["training"]["seed"]}__{tag}'
    destination = output / label
    support = SUPPORT / label
    support.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        receipt_path = destination / 'source_run.json'
        receipt = read(receipt_path if receipt_path.exists() else support / 'source_run.json')
        if receipt['source_run'] != str(source):
            raise FileExistsError(f'Result folder belongs to another run: {destination}')
        for filename, expected in receipt['file_sha256'].items():
            collected = destination / filename
            if not collected.exists():
                collected = support / filename
            if filename == 'learning_curve.png' and receipt.get('learning_curve'):
                curve = receipt['learning_curve']
                source_matches = (digest(source / curve['source_file']) == curve['source_sha256']
                                  and digest(source / receipt['source_files'][filename]) == curve['original_plot_sha256'])
            elif filename == 'validation_predictions.csv' and receipt.get('predictions'):
                source_matches = digest(source / filename) == receipt['predictions']['source_sha256']
            elif filename == 'dataset.xlsx' and receipt.get('dataset'):
                source_matches = (digest(source / filename) == receipt['dataset']['source_sha256']
                                  and digest(run_config_path(source)) == receipt['dataset']['config_sha256'])
            elif filename == model_config_name(config) and receipt.get('config_summary'):
                source_matches = (digest(run_config_path(source)) == receipt['config_summary']['source_sha256']
                                  and digest(source / 'dataset.xlsx') == receipt['config_summary']['dataset_sha256'])
            else:
                source_matches = digest(source / receipt['source_files'][filename]) == expected
            if digest(collected) != expected or not source_matches:
                raise ValueError(f'Collected result/source changed; refusing overwrite: {destination / filename}')
        # Backfill newly included artifacts in folders collected by older versions.
        for original, target in FILES.items():
            collected = destination / target
            if target in receipt['file_sha256'] and collected.exists():
                continue
            expected = digest(source / original)
            if collected.exists():
                if digest(collected) != expected:
                    raise ValueError(f'Collected result differs from source: {collected}')
            else:
                temporary = collected.with_suffix(collected.suffix + '.tmp')
                shutil.copy2(source / original, temporary)
                if digest(temporary) != expected:
                    raise ValueError(f'Copy verification failed: {target}')
                temporary.replace(collected)
            receipt['file_sha256'][target] = expected
            receipt['source_files'][target] = original
        write_json(support / 'source_run.json', receipt)
        # Keep the selected review files visible. Preserve older collected
        # extras and their receipt with the supporting records under model_dir.
        for extra in destination.iterdir():
            if extra.name not in {*FILES.values(), model_config_name(config)}:
                archived = support / extra.name
                if archived.exists():
                    raise FileExistsError(f'Refusing to overwrite supporting file: {archived}')
                extra.rename(archived)
    else:
        staging = output / ('.' + label + '.partial')
        staging.mkdir(exist_ok=False)
        hashes, sources = {}, {}
        for original, target in FILES.items():
            if not (source / original).is_file():
                raise FileNotFoundError(source / original)
            shutil.copy2(source / original, staging / target)
            hashes[target] = digest(source / original)
            if digest(staging / target) != hashes[target]:
                raise ValueError(f'Copy verification failed: {target}')
            sources[target] = original
        receipt = {'source_run': str(source), 'experiment_id': identifier,
                   'file_sha256': hashes, 'source_files': sources,
                   'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                   'note': 'Verified copies of the completed experiment; original artifacts retained'}
        write_json(support / 'source_run.json', receipt)
        staging.rename(destination)
        print(f'Collected: {label}', flush=True)
    refresh_curve(source, destination, support, receipt, config, status)
    refresh_predictions(source, destination, support, receipt, config)
    refresh_dataset(source, destination, support, receipt, config)
    refresh_config(source, destination, support, receipt, config)
    metrics = read(destination / 'validation_metrics.json')
    profile = config['dataset'].get('profile')
    label_version = {'ratings_8cc903b74f9f': 'reviewed_v1', 'ratings_281f9fcc448e': 'reviewed_v2',
                     'ratings_e4ea1ce05a02': 'pre_review'}.get(profile, profile)
    row = {'folder': label, 'experiment_id': identifier, 'method': config['model']['head'],
           'condition': condition(config), 'seed': config['training']['seed'],
           'label_version': label_version, 'rating_profile': profile,
           'training_images': len(config['cohort']['training_image_ids']),
           'monitoring_images': len(config['cohort']['monitor_image_ids']),
           'validation_images': metrics['image_count'], 'best_epoch': status['selected_epoch'],
           'learning_rate': config['training']['optimizer_settings']['lr'],
           'binary_loss_weight': config['loss'].get('binary_loss_weight'),
           'consensus_qwk': metrics.get('consensus_reference_quadratic_weighted_kappa'),
           'consensus_mae': metrics.get('consensus_reference_mae'), 'emd': metrics.get('earth_movers_distance')}
    for key, prefix in [('binary_grade_argmax', 'grade_argmax'), ('binary', 'summed_probability'), ('binary_dedicated_head', 'dedicated_head')]:
        row.update({prefix+'_'+name: metrics.get(key, {}).get(name) for name in ('f1', 'sensitivity', 'specificity')})
    return row


def write_summary(rows, output):
    """Update collected rows while retaining runs collected outside the queue."""
    summary = output / 'summary.csv'
    existing = []
    if summary.exists():
        with summary.open(newline='', encoding='utf-8') as handle:
            existing = list(csv.DictReader(handle))
    merged = {row['folder']: row for row in existing}
    merged.update({row['folder']: row for row in rows})
    fieldnames = list(dict.fromkeys(key for row in merged.values() for key in row))
    SUPPORT.mkdir(parents=True, exist_ok=True)
    temporary = SUPPORT / 'summary.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged.values())
    temporary.replace(summary)


def collect_run(source, output=ROOT / 'results'):
    output = Path(output)
    row = export_run(source, output)
    write_summary([row], output)
    return output / row['folder']


def collect(queue_status, output):
    queue = read(queue_status)
    if queue.get('mode') != 'run':
        raise ValueError('Collect the full-run queue, not diagnostic smoke runs')
    output.mkdir(parents=True, exist_ok=True)
    SUPPORT.mkdir(parents=True, exist_ok=True)
    rows = [export_run(item['run'], output) for item in queue['completed']]
    summary_pending = False
    if rows:
        try:
            write_summary(rows, output)
        except PermissionError:
            # Excel can keep the CSV locked. Models still collect normally;
            # refresh the summary on a later pass after the file is released.
            summary_pending = True
    write_json(SUPPORT / 'collection_status.json', {'source_queue': str(queue_status), 'queue_status': queue['status'],
               'completed_models_collected': len(rows), 'remaining_including_active': len(queue['remaining']),
               'summary_update_pending': summary_pending,
               'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    queue['summary_update_pending'] = summary_pending
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--queue-status', type=Path)
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--watch', action='store_true', help='Collect new completions until the queue finishes or fails')
    args = parser.parse_args()
    if args.run_dir:
        collect_run(args.run_dir.resolve(), args.output_dir.resolve())
        return
    while True:
        try:
            queue = collect(args.queue_status.resolve(), args.output_dir.resolve())
        except PermissionError as error:
            if not args.watch:
                raise
            print(f'Waiting for a locked results file: {error}', flush=True)
            time.sleep(30)
            continue
        if not args.watch or (queue['status'] != 'running' and not queue['summary_update_pending']):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
