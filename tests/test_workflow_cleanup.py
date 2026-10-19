import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image

from autoqc import training
from autoqc.model import IQAnet
from autoqc.validation import evaluate_saved_run


ROOT = Path(__file__).resolve().parents[1]


class WorkflowCleanupChecks(unittest.TestCase):
    def test_training_help_and_retired_single_grader_rejection(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'autoqc.training', '--help'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn('--loss {cross_entropy,continuous_cdw_ce}', result.stdout)
        self.assertIn('--grader {soft}', result.stdout)
        self.assertIn('--class_weights_from', result.stdout)
        for retired in ('--test_split', '--val_split', '--images_dir', '--output_sigmoid'):
            self.assertNotIn(retired, result.stdout)
        for grader in ('previousStudent', 'shane'):
            argv = ['training.py', '--grader', grader, '--dataset_csv_path', 'unused',
                    '--ratings_csv_path', 'unused', '--output_dir', 'unused']
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    training.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn('invalid choice', errors.getvalue())

    def test_historical_single_grader_state_dict_and_metadata_reload(self):
        for grader in ('previousStudent', 'shane'):
            with self.subTest(grader=grader), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                image = run / 'validation.png'
                Image.new('L', (768, 768), 100).save(image)
                manifest = pd.DataFrame([{
                    'image_id': split, 'image_path': str(image) if split == 'validation' else 'DO_NOT_OPEN',
                    'patient_id': split, 'project': 'VISUS-NHP' if split == 'test' else 'other',
                    'split': split, f'{grader}_grade': 4,
                } for split in ('train', 'validation', 'test')])
                # Distinct unavailable paths let the legacy validation guards pass without decoding them.
                manifest.loc[manifest.split.eq('test'), 'image_path'] = 'DO_NOT_OPEN_test'
                manifest.to_csv(run / 'manifest.csv', index=False)
                params = {'batch_normalization': False, 'output_sigmoid': False,
                          'batch_size': 4, 'dataset_manifest': 'manifest.csv'}
                if grader == 'previousStudent':
                    params['grader'] = grader
                pd.DataFrame([params]).to_csv(run / 'model_params.csv', index=False)
                torch.save(IQAnet(num_classes=5).state_dict(), run / 'model.pth')
                with patch('PIL.Image.open', wraps=Image.open) as opened, contextlib.redirect_stdout(io.StringIO()):
                    metrics, predictions = evaluate_saved_run(run, grader=grader, device='cpu')
                self.assertEqual([call.args[0] for call in opened.call_args_list], [str(image)])
                self.assertEqual(metrics['grader'], grader)
                self.assertEqual(metrics['image_count'], 1)
                self.assertEqual(predictions.reference_grade.tolist(), [4])
                self.assertTrue((run / 'validation_predictions.csv').is_file())


if __name__ == '__main__':
    unittest.main()
