"""One Excel row per experiment, extracted from saved outputs without loading models.

Training appends/updates its row automatically. The CLI also imports older runs:
python -m auto_qc.CNN.experiment_log --local-root C:/Users/Shane/Desktop/qc_data
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import pandas as pd
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
FILENAME = 'experiment_log.xlsx'
COLUMNS = [
    'Run date (UTC)', 'Experiment', 'Architecture', 'Head', 'Status', 'Diagnostic', 'Input mode',
    'Validation QWK', 'Validation accuracy', 'Validation MAE',
    'Binary head F1', 'Binary head sensitivity', 'Binary head specificity',
    'Binary head FP', 'Binary head FN', 'Grade argmax binary F1',
    'Seed', 'Learning rate', 'Binary loss weight', 'Loss', 'Dataset', 'Target',
    'Training images', 'Validation images', 'Monitoring images', 'Epochs completed',
    'Best loss epoch', 'Best validation loss', 'Epoch limit', 'Batch size',
    'Optimizer', 'Weight decay', 'Dropout', 'Pretrained', 'Scheduler patience',
    'Class weights', 'Graders', 'Dataset SHA256', 'Source type', 'Date source',
    'Output folder', 'Notes',
]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def first(*values):
    return next((v for v in values if v is not None), None)


def run_row(directory, config=None):
    """Read current simple YAML, older resolved YAML, or original run_config.json."""
    directory = Path(directory).resolve()
    legacy = (directory / 'run_config.json').exists()
    if config is None:
        candidates = sorted(directory.glob('*.yaml'))
        if candidates:
            config = yaml.safe_load(candidates[0].read_text(encoding='utf-8-sig'))
        elif legacy:
            config = read_json(directory / 'run_config.json')
        else:
            raise ValueError(f'No saved settings: {directory}')
    resolved = config.get('schema') == 'autoqc_cnn_config_v1'
    model = config.get('model', {})
    training = config.get('training', config if legacy else {})
    optimizer = config.get('optimizer', {})
    optimizer = optimizer if isinstance(optimizer, dict) else {'name': optimizer}
    optimizer_settings = training.get('optimizer_settings', {})
    dataset = config.get('dataset', {})
    data = config.get('data', {})
    loss = config.get('loss', {})
    loss = loss if isinstance(loss, dict) else {'name': loss}
    scheduler = training.get('scheduler', {}) if resolved else config.get('scheduler', {})
    scheduler_settings = scheduler.get('settings', scheduler)
    metrics = read_json(directory / 'validation_metrics.json')
    distribution = read_json(directory / 'distribution_metrics.json')
    metrics = {**distribution, **metrics}
    status = read_json(directory / 'run_status.json')
    history_path = directory / 'training_history.csv'
    if not history_path.exists():
        history_path = directory / 'learning_curves.csv'
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    val_column = next((c for c in ('val_primary_loss', 'val_loss') if c in history), None)
    best = history.loc[history[val_column].idxmin()] if val_column and history[val_column].notna().any() else {}
    counts, snapshot_metadata = {}, {}
    snapshot = directory / 'dataset.xlsx'
    if snapshot.exists():
        frame = pd.read_excel(snapshot, sheet_name='dataset', usecols=lambda c: c in
                              ('split', 'training_eligible', 'monitor_eligible'))
        if 'training_eligible' in frame:
            counts['train'] = int(frame.training_eligible.sum())
        if 'monitor_eligible' in frame:
            counts['monitor'] = int(frame.monitor_eligible.sum())
        values = pd.read_excel(snapshot, sheet_name='metadata')
        snapshot_metadata = {r.key: json.loads(r.value) for r in values.itertuples()}
    elif (directory / 'manifest.csv').exists():
        frame = pd.read_csv(directory / 'manifest.csv')
        split = next((c for c in ('split', 'mode') if c in frame), None)
        if split:
            counts['train'] = int(frame[split].eq('train').sum())
    cohort = config.get('cohort', {})
    for key, role in [('train', 'training'), ('monitor', 'monitor')]:
        if key not in counts and role + '_image_ids' in cohort:
            counts[key] = len(cohort[role + '_image_ids'])
    match = re.search(r'_(\d{8})_(\d{6})(?:_|$)', directory.name)
    date = (datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S') if match
            else datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc).replace(tzinfo=None))
    head = model.get('head')
    if legacy:
        head = first(config.get('head'), config.get('ordinal_head'), config.get('model_type'))
        if head is None:
            head = next((name for name in ('or_soft_frozen_binary', 'or_soft_binary', 'coral', 'or_soft')
                         if name in directory.name), 'five_logits')
    weights = first(loss.get('class_weights'), config.get('class_weight_mode'), config.get('class_weight_source'))
    if isinstance(weights, dict):
        weights = first(weights.get('mode'), weights.get('method'))
    binary = metrics.get('binary_dedicated_head', {})
    argmax = metrics.get('binary_grade_argmax', {})
    diagnostic = bool(status.get('smoke') or config.get('diagnostic_smoke') or
                      config.get('diagnostic_only') or 'smoke' in directory.name.lower())
    row = {
        'Run date (UTC)': date,
        'Experiment': first(config.get('experiment_name'), config.get('experiment_id'),
                            re.sub(r'_\d{8}_\d{6}_.*$', '', directory.name)),
        'Architecture': model.get('backbone', 'iqanet'), 'Head': head,
        'Input mode': first(data.get('input_mode'), config.get('preprocessing', {}).get('input_mode'), 'patches'),
        'Status': status.get('status', 'completed' if metrics else 'incomplete'),
        'Diagnostic': diagnostic,
        'Validation QWK': metrics.get('consensus_reference_quadratic_weighted_kappa'),
        'Validation accuracy': metrics.get('consensus_reference_exact_accuracy'),
        'Validation MAE': metrics.get('consensus_reference_mae'),
        'Grade argmax binary F1': argmax.get('f1'),
        'Seed': training.get('seed'),
        'Learning rate': first(optimizer.get('lr'), optimizer_settings.get('lr'), training.get('learning_rate')),
        'Binary loss weight': first(loss.get('binary_loss_weight'), config.get('binary_loss_weight')),
        'Loss': first(loss.get('name'), config.get('loss_selection')),
        'Dataset': first(data.get('dataset'), dataset.get('path'), config.get('dataset_csv_path')),
        'Target': first(data.get('target'), dataset.get('target_kind'), config.get('target_construction')),
        'Training images': counts.get('train'), 'Validation images': metrics.get('image_count'),
        'Monitoring images': first(counts.get('monitor'), metrics.get('monitoring_population_size')),
        'Epochs completed': int(history.epoch.max()) if 'epoch' in history and len(history) else status.get('epoch'),
        'Best loss epoch': int(best['epoch']) if 'epoch' in best else status.get('selected_epoch'),
        'Best validation loss': float(best[val_column]) if val_column and len(best) else status.get('best_selection_loss'),
        'Epoch limit': training.get('epochs'), 'Batch size': training.get('batch_size'),
        'Optimizer': first(optimizer.get('name'), training.get('optimizer')),
        'Weight decay': first(optimizer.get('weight_decay'), optimizer_settings.get('weight_decay')),
        'Dropout': model.get('dropout_p'), 'Pretrained': model.get('pretrained', False),
        'Scheduler patience': scheduler_settings.get('patience'), 'Class weights': weights,
        'Graders': ', '.join(first(dataset.get('graders'), config.get('grader_columns'),
                                   data.get('graders'), snapshot_metadata.get('graders'), [])),
        'Dataset SHA256': first(dataset.get('sha256'), config.get('manifest_sha256'), config.get('input_manifest_sha256')),
        'Source type': 'historical implementation' if legacy else 'current implementation',
        'Date source': 'run name (UTC)' if match else 'folder modified time (UTC; original date unavailable)',
        'Output folder': str(directory),
        'Notes': 'Consensus validation metrics unavailable in saved outputs' if metrics and
                 metrics.get('consensus_reference_quadratic_weighted_kappa') is None else None,
    }
    for label, key in [('F1', 'f1'), ('sensitivity', 'sensitivity'), ('specificity', 'specificity'), ('FP', 'FP'), ('FN', 'FN')]:
        row['Binary head ' + label] = binary.get(key)
    return row


@contextmanager
def workbook_lock(path):
    lock = path.with_suffix('.lock')
    for attempt in range(30):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if attempt == 29:
                raise OSError(f'Experiment log is busy: {lock}')
            time.sleep(.1)
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def update_log(local_root, rows):
    """Upsert by output folder; preserve existing rows and user Notes; save atomically."""
    path = Path(local_root) / FILENAME
    with workbook_lock(path):
        records = {}
        if path.exists():
            previous = load_workbook(path)
            sheet = previous['Experiments']
            values = sheet.iter_rows(values_only=True)
            headers = next(values)
            # Upgrade logs written before whole-image mode was added.
            if list(headers) not in (COLUMNS, [c for c in COLUMNS if c != 'Input mode']):
                raise ValueError('Experiment log columns changed; restore the original headers before updating')
            records = {r['Output folder']: r for values in values
                       if (r := dict(zip(headers, values))).get('Output folder')}
            for record in records.values():
                record.setdefault('Input mode', 'patches')
            previous.close()
        for row in rows:
            key = row['Output folder']
            row = dict(row, Notes=first(records.get(key, {}).get('Notes'), row.get('Notes')))
            records[key] = row
        book = Workbook()
        sheet = book.active
        sheet.title = 'Experiments'
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = 'C2'
        sheet.append(COLUMNS)
        for row in sorted(records.values(), key=lambda r: (r['Run date (UTC)'], r['Experiment']), reverse=True):
            values = [row.get(key) for key in COLUMNS]
            values = [None if isinstance(v, float) and not math.isfinite(v) else v for v in values]
            sheet.append(values)
            sheet.row_dimensions[sheet.max_row].height = max(28, 14 * math.ceil(len(row['Experiment']) / 45) + 6)
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='24364B')
            cell.font = Font(name='Arial', size=10, bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        sheet.row_dimensions[1].height = 32
        percent = {'Validation accuracy', 'Binary head sensitivity', 'Binary head specificity'}
        decimal = {'Validation QWK', 'Validation MAE', 'Binary head F1', 'Grade argmax binary F1', 'Best validation loss'}
        for index, name in enumerate(COLUMNS, 1):
            width = 20 if name in percent or name in decimal else 17
            if name == 'Run date (UTC)':
                width = 23
            if name in ('Experiment', 'Dataset', 'Graders', 'Output folder', 'Notes', 'Date source', 'Target'):
                width = 48 if name == 'Experiment' else 35
            sheet.column_dimensions[get_column_letter(index)].width = width
            for column in sheet.iter_cols(min_col=index, max_col=index, min_row=2):
                for cell in column:
                    cell.font = Font(name='Arial', size=10)
                    cell.alignment = Alignment(vertical='center', wrap_text=name == 'Experiment')
                    if name in percent:
                        cell.number_format = '0.0%'
                    elif name in decimal:
                        cell.number_format = '0.000'
                    elif name == 'Learning rate':
                        cell.number_format = '0.000000'
                    elif name == 'Run date (UTC)':
                        cell.number_format = 'yyyy-mm-dd hh:mm:ss'
                    if cell.data_type == 'f':  # Treat names/notes as literal text, never formulas.
                        cell.data_type = 's'
        if sheet.max_row > 1:
            table = Table(displayName='ExperimentLog', ref=sheet.dimensions)
            table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
            sheet.add_table(table)
        fd, temporary = tempfile.mkstemp(prefix='experiment_log_', suffix='.xlsx', dir=path.parent)
        os.close(fd)
        try:
            book.save(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
            book.close()
    return path, len(records)


def backfill(local_root):
    """Original run locations only; copied results are not counted twice."""
    directories = set()
    for root in (Path(local_root) / 'models', ROOT / 'model_dir', ROOT / 'runs'):
        if root.exists():
            for name in ('run_config.json', '*.yaml'):
                for settings in root.rglob(name):
                    directory = settings.parent
                    if 'results_support' in directory.parts or 'source_code' in directory.parts:
                        continue
                    if any((directory / file).exists() for file in
                           ('checkpoint.pth', 'model.pth', 'training_history.csv', 'learning_curves.csv')):
                        directories.add(directory.resolve())
    rows, errors = [], []
    for directory in sorted(directories):
        try:
            rows.append(run_row(directory))
        except (ValueError, KeyError, OSError) as error:
            errors.append(f'{directory}: {error}')
    if errors:
        raise ValueError('Could not import saved runs:\n' + '\n'.join(errors))
    return update_log(local_root, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, required=True)
    args = parser.parse_args()
    path, count = backfill(args.local_root)
    print(f'Experiment log: {path} ({count} experiments)')


if __name__ == '__main__':
    main()
