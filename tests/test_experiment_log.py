"""Saved-run extraction, repeatable upserts and safe failure when Excel is open."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpyxl import load_workbook
import yaml

from auto_qc.CNN.experiment_log import COLUMNS, backfill, run_row, update_log


class ExperimentLogTests(unittest.TestCase):
    def make_run(self, root):
        run = root / 'models' / 'example_20261110_120201_123456_abcd1234'
        run.mkdir(parents=True)
        (run / 'resnet18_or_soft_binary.yaml').write_text(yaml.safe_dump({
            'model': {'backbone': 'resnet18', 'head': 'or_soft_binary', 'pretrained': True},
            'data': {'dataset': 'reviewed_v2/multigrader_labels.xlsx', 'target': 'ordinal'},
            'training': {'seed': 42, 'epochs': 100, 'batch_size': 16},
            'optimizer': {'name': 'AdamW', 'lr': .00001},
            'loss': {'name': 'joint_bce', 'binary_loss_weight': 1.},
        }))
        (run / 'validation_metrics.json').write_text(json.dumps({
            'image_count': 127, 'consensus_reference_quadratic_weighted_kappa': .79,
            'binary_dedicated_head': {'f1': .846, 'FN': 2},
            'binary_grade_argmax': {'f1': .80},
        }))
        (run / 'training_history.csv').write_text(
            'epoch,val_primary_loss\n1,1.2\n2,1.0\n3,1.1\n')
        return run

    def test_extract_upsert_preserves_notes_dates_and_distinct_binary_methods(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root)
            row = run_row(run)
            self.assertEqual(row['Run date (UTC)'].isoformat(), '2026-11-10T12:02:01')
            self.assertEqual(row['Best loss epoch'], 2)
            self.assertEqual(row['Epochs completed'], 3)
            self.assertEqual(row['Learning rate'], .00001)
            self.assertEqual(row['Binary head F1'], .846)
            self.assertEqual(row['Grade argmax binary F1'], .80)
            self.assertIsNone(row['Validation accuracy'])
            path, count = update_log(root, [row])
            self.assertEqual(count, 1)
            book = load_workbook(path)
            book.active.cell(2, COLUMNS.index('Notes') + 1, 'Supervisor review')
            book.save(path)
            book.close()
            row['Validation QWK'] = .80
            self.assertEqual(update_log(root, [row])[1], 1)
            book = load_workbook(path)
            values = dict(zip(COLUMNS, next(book.active.iter_rows(min_row=2, values_only=True))))
            self.assertEqual(values['Notes'], 'Supervisor review')
            self.assertEqual(values['Validation QWK'], .80)
            self.assertEqual(book.active.max_row, 2)
            self.assertEqual(book.active.freeze_panes, 'C2')
            self.assertEqual(len(book.active.tables), 1)
            book.close()

    def test_failed_replace_leaves_previous_workbook_and_cleans_temporary_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            row = run_row(self.make_run(root))
            path, _ = update_log(root, [row])
            before = path.read_bytes()
            with patch('auto_qc.CNN.experiment_log.os.replace', side_effect=PermissionError('Excel open')):
                with self.assertRaises(PermissionError):
                    update_log(root, [row])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(root.glob('*.lock')))
            self.assertEqual(list(root.glob('*.xlsx')), [path])

    def test_old_log_gains_input_mode_without_losing_notes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            row = run_row(self.make_run(root))
            path, _ = update_log(root, [row])
            book = load_workbook(path)
            book.active.cell(2, COLUMNS.index('Notes') + 1, 'Keep this note')
            book.active.delete_cols(COLUMNS.index('Input mode') + 1)
            book.save(path)
            book.close()
            self.assertEqual(update_log(root, [row])[1], 1)
            book = load_workbook(path)
            values = dict(zip(COLUMNS, next(book.active.iter_rows(min_row=2, values_only=True))))
            self.assertEqual(values['Input mode'], 'patches')
            self.assertEqual(values['Notes'], 'Keep this note')
            book.close()

    def test_backfill_excludes_support_copies_and_keeps_incomplete_runs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root)
            (run / 'validation_metrics.json').unlink()
            copied = root / 'models' / 'results_support' / 'copy'
            copied.mkdir(parents=True)
            (copied / 'config.yaml').write_text('{}')
            (copied / 'training_history.csv').write_text('epoch,val_loss\n1,2\n')
            with patch('auto_qc.CNN.experiment_log.ROOT', root / 'empty_repo'):
                path, count = backfill(root)
            self.assertEqual(count, 1)
            book = load_workbook(path)
            values = dict(zip(COLUMNS, next(book.active.iter_rows(min_row=2, values_only=True))))
            self.assertEqual(values['Status'], 'incomplete')
            self.assertIsNone(values['Validation QWK'])
            book.close()


if __name__ == '__main__':
    unittest.main()
