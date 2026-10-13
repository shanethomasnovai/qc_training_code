"""Setup regression checks; no real training or result files are produced."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image

import prepare_data
import training


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'prepared_visus_nhp' / 'manifest.csv'


class SetupChecks(unittest.TestCase):
    def test_actual_images_and_patient_groups(self):
        df = pd.read_csv(MANIFEST)
        self.assertEqual(df.groupby('patient_id')['split'].nunique().max(), 1)
        self.assertTrue((df.loc[df.project == 'VISUS-NHP', 'split'] == 'test').all())
        # Independently check patient keys against full image identifiers.
        expected = df.image_id.str.split('_').str[:3].str.join('_')
        self.assertTrue((expected == df.patient_id).all())
        for path in df.image_path:
            with Image.open(path) as image:
                image.verify()
        for grader in ('previousStudent', 'shane'):
            for split in ('train', 'validation'):
                self.assertEqual(set(df.loc[df.split == split, f'{grader}_grade']), set(range(1, 6)))

    def test_valid_saved_split_is_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            argv = ['prepare_inputs.py', '--previous-student', str(ROOT / 'data/multi_class_qc_dataset.csv'),
                    '--shane', str(ROOT / 'data/shane-image-file-listing.csv'),
                    '--image-dir', str(Path(pd.read_csv(MANIFEST).image_path.iloc[0]).parent),
                    '--split-file', str(MANIFEST), '--seed', '999', '--output-dir', folder]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                prepare_data.main()
            pd.testing.assert_frame_equal(pd.read_csv(MANIFEST), pd.read_csv(Path(folder) / 'manifest.csv'))

    def test_grade_weights_and_selected_checkpoint(self):
        # Reverse encounter order and unequal counts expose unordered class weights.
        patches = [(torch.zeros(1, 8, 8), torch.tensor(index), 'train')
                   for index in reversed(range(5)) for _ in range(index + 1)]
        patches += [(torch.zeros(1, 8, 8), torch.tensor(index), 'val') for index in range(5)]
        for grader in ('previousStudent', 'shane'):
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                calls = []
                def fake_train(model, train_loader, val_loader, criterion, optimizer, device):
                    self.assertTrue(torch.allclose(criterion.weight.cpu(), torch.tensor([3 / (i + 1) for i in range(5)])))
                    calls.append(True)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.fill_(len(calls))
                    return 1.0, float(len(calls))
                argv = ['training.py', '--grader', grader, '--dataset_csv_path', str(MANIFEST),
                        '--output_dir', folder, '--epochs', '2']
                with patch('sys.argv', argv), patch.object(training, 'subsample_subimages', return_value=(patches, None)), \
                     patch.object(training, 'IQAnet', side_effect=lambda **kwargs: torch.nn.Linear(1, 5)), \
                     patch.object(training, 'train', side_effect=fake_train), \
                     patch.object(training, 'evaluate_saved_run'), \
                     patch.object(training, 'save_model') as save, contextlib.redirect_stdout(io.StringIO()):
                    training.main()
                state, _, loss, epoch = save.call_args.args[:4]
                self.assertEqual((loss, epoch), (1.0, 1))
                self.assertTrue(all(torch.all(value == 1) for value in state.values()))


if __name__ == '__main__':
    unittest.main()
