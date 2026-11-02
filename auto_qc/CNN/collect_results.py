"""Collect completed reruns into one readable results folder, without moving runs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'checkpoint.pth': 'checkpoint.pth',
    'training_history.png': 'learning_curve.png',
    'training_history.csv': 'training_history.csv',
    'config.yaml': 'config.yaml',
    'dataset.xlsx': 'dataset.xlsx',
    'validation_predictions.csv': 'validation_predictions.csv',
    'validation_metrics.json': 'validation_metrics.json',
    'validation_confusion_matrix.png': 'validation_confusion_matrix.png',
    'validation_confusion_matrix.csv': 'validation_confusion_matrix.csv',
    'training.log': 'training.log',
    'run_status.json': 'run_status.json',
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


def export_run(source, output):
    source = Path(source).resolve()
    status, config = read(source / 'run_status.json'), read(source / 'config.yaml')
    if status['status'] != 'completed' or status.get('smoke') or not status.get('required_outputs_verified'):
        raise ValueError(f'Only verified completed full runs can be collected: {source}')
    identifier = config['experiment_id']
    number = identifier.split('_')[1]
    label = f'{config["model"]["head"]}__{condition(config)}__seed{config["training"]["seed"]}__{number}_{source.name.split("_")[-1]}'
    destination = output / label
    if destination.exists():
        receipt = read(destination / 'source_run.json')
        if receipt['source_run'] != str(source):
            raise FileExistsError(f'Result folder belongs to another run: {destination}')
        for filename, expected in receipt['file_sha256'].items():
            if digest(destination / filename) != expected or digest(source / receipt['source_files'][filename]) != expected:
                raise ValueError(f'Collected result/source changed; refusing overwrite: {destination / filename}')
    else:
        staging = output / ('.' + label + '.partial')
        staging.mkdir(exist_ok=False)
        hashes, sources = {}, {}
        selected = dict(FILES)
        selected.update({p.name: p.name for p in source.glob('validation_binary*_confusion_matrix.csv')})
        for original, target in selected.items():
            if not (source / original).is_file():
                raise FileNotFoundError(source / original)
            shutil.copy2(source / original, staging / target)
            hashes[target] = digest(source / original)
            if digest(staging / target) != hashes[target]:
                raise ValueError(f'Copy verification failed: {target}')
            sources[target] = original
        write_json(staging / 'source_run.json', {'source_run': str(source), 'experiment_id': identifier,
                   'file_sha256': hashes, 'source_files': sources,
                   'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                   'note': 'Verified copies of the completed experiment; original artifacts retained'})
        staging.rename(destination)
        print(f'Collected: {label}', flush=True)
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


def collect(queue_status, output):
    queue = read(queue_status)
    if queue.get('mode') != 'run':
        raise ValueError('Collect the full-run queue, not diagnostic smoke runs')
    output.mkdir(parents=True, exist_ok=True)
    rows = [export_run(item['run'], output) for item in queue['completed']]
    summary_pending = False
    if rows:
        temporary = output / 'summary.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            temporary.replace(output / 'summary.csv')
        except PermissionError:
            # Excel can keep the CSV locked. Models still collect normally;
            # refresh the summary on a later pass after the file is released.
            summary_pending = True
    write_json(output / 'collection_status.json', {'source_queue': str(queue_status), 'queue_status': queue['status'],
               'completed_models_collected': len(rows), 'remaining_including_active': len(queue['remaining']),
               'summary_update_pending': summary_pending,
               'updated_at_utc': datetime.now(timezone.utc).isoformat()})
    queue['summary_update_pending'] = summary_pending
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-status', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--watch', action='store_true', help='Collect new completions until the queue finishes or fails')
    args = parser.parse_args()
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
