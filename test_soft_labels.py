import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

import training
from dataset import ImageQualityDataset, subsample_subimages
from model import IQAnet
from soft_labels import TARGET_COLUMNS, construct_targets, load_soft_manifest, read_table
from validation import evaluate_validation, evaluate_saved_run, image_transform


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'prepared_visus_nhp/manifest.csv'


class SoftLabelChecks(unittest.TestCase):
    def test_targets_missing_and_invalid(self):
        df = pd.DataFrame({'image_id': list('abcde'), 'a_grade': [2, 2, 4, '', ' '],
                           'b_grade': [3, '', '', None, ''], 'c_grade': [3, 3, '', '', '']})
        result = construct_targets(df, ['a_grade', 'b_grade', 'c_grade'])
        np.testing.assert_allclose(result[TARGET_COLUMNS].to_numpy(),
                                   [[0, 1/3, 2/3, 0, 0], [0, .5, .5, 0, 0],
                                    [0, 0, 0, 1, 0], [0]*5, [0]*5])
        self.assertEqual(result.rating_count.tolist(), [3, 2, 1, 0, 0])
        for invalid in [0, 6, 2.5, 'no', 'NaN', 'inf', 'NA', True]:
            bad = df.copy()
            bad['a_grade'] = bad.a_grade.astype(object)
            bad.loc[0, 'a_grade'] = invalid
            with self.assertRaisesRegex(ValueError, 'image.*a.*a_grade'):
                construct_targets(bad, ['a_grade'])
        with self.assertRaises(ValueError):
            construct_targets(df, ['image_id'])

    def test_actual_population_join_and_split_preservation(self):
        base = read_table(MANIFEST)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ratings.csv'
            ratings = base[['image_id', 'previousStudent_grade', 'shane_grade']].copy()
            ratings['future_grade'] = ''
            ratings['metadata'] = 'not a rating'
            ratings.iloc[::-1].to_csv(path, index=False)
            result, columns, report = load_soft_manifest(MANIFEST, path)
            self.assertEqual(columns, ['previousStudent_grade', 'shane_grade', 'future_grade'])
            pd.testing.assert_frame_equal(result[base.columns], base)
            self.assertEqual(sum(report.values()), 0)
            ratings['image_filename'] = ratings.image_id + '.tif'
            ratings.drop(columns='image_id').to_csv(path, index=False)
            renamed, _, _ = load_soft_manifest(MANIFEST, path)
            self.assertEqual(result.image_id.tolist(), renamed.image_id.tolist())
            ratings.assign(image_filename=ratings.image_id).drop(columns='image_id').to_csv(path, index=False)
            extensionless, _, _ = load_soft_manifest(MANIFEST, path)
            self.assertEqual(result.image_id.tolist(), extensionless.image_id.tolist())
            for bad in [pd.concat([ratings, ratings.iloc[:1]]),
                        ratings.assign(image_id=lambda x: x.image_id.where(x.index != 0, 'unknown')),
                        ratings.assign(split='train')]:
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_soft_manifest(MANIFEST, path)
            ratings.drop(columns='image_filename').iloc[1:].to_csv(path, index=False)
            result, _, report = load_soft_manifest(MANIFEST, path)
            self.assertEqual(len(result), len(base))
            self.assertEqual(sum(report.values()), 1)
            self.assertFalse(result.has_ratings.iloc[0])
            path.write_text('image_id,a_grade,a_grade\nx,2,3\n')
            with self.assertRaisesRegex(ValueError, 'duplicate column'):
                load_soft_manifest(MANIFEST, path)
            known = json.loads((MANIFEST.parent / 'split_report.json').read_text())['shane_only_ids'][0]
            extended = pd.concat([ratings.drop(columns='image_filename'), pd.DataFrame([
                {'image_id': known, 'shane_grade': 3}])], ignore_index=True)
            extended.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'outside existing manifest'):
                load_soft_manifest(MANIFEST, path)
            result, _, _ = load_soft_manifest(MANIFEST, path, exclude_known_unmatched=True)
            self.assertEqual(len(result), len(base))
            extended.loc[len(extended)-1, 'image_id'] = 'unapproved_new_image'
            extended.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'not in saved split exclusions'):
                load_soft_manifest(MANIFEST, path, exclude_known_unmatched=True)

    def test_one_hot_equivalence_and_real_patch_gradients(self):
        torch.manual_seed(42)
        logits = torch.randn(5, 5, requires_grad=True)
        labels = torch.arange(5)
        hard = torch.nn.functional.cross_entropy(logits, labels)
        soft = torch.nn.functional.cross_entropy(logits, torch.nn.functional.one_hot(labels, 5).float())
        torch.testing.assert_close(hard, soft)
        torch.testing.assert_close(torch.autograd.grad(hard, logits, retain_graph=True)[0],
                                   torch.autograd.grad(soft, logits)[0])
        df = read_table(MANIFEST)
        row = df[df.split == 'train'].iloc[:1].copy().reset_index(drop=True)
        row = construct_targets(row, ['previousStudent_grade', 'shane_grade'])
        row['image_filename'] = row.image_path
        data = ImageQualityDataset(row, image_transform(), target_columns=TARGET_COLUMNS)
        with contextlib.redirect_stdout(io.StringIO()):
            patches, _ = subsample_subimages(data, method='all', df=row)
        self.assertEqual(len(patches), 9)
        model = IQAnet(num_classes=5)
        inputs = torch.stack([p[0] for p in patches[:2]])
        targets = torch.stack([p[1] for p in patches[:2]])
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()

    def test_training_save_reload_and_separate_references(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = folder / 'variant.1_0.png'
            Image.new('L', (768, 768), 100).save(image)
            base = pd.DataFrame([
                dict(image_id='train', image_path=str(folder / 'train.png'), patient_id='t',
                     project='other', project_site='s', split='train', previousStudent_grade=2, shane_grade=3),
                dict(image_id='variant.1_0', image_path=str(image), patient_id='v',
                     project='other', project_site='s', split='validation', previousStudent_grade=2, shane_grade=3),
                dict(image_id='test', image_path=str(folder / 'test.png'), patient_id='h',
                     project='VISUS-NHP', project_site='s', split='test', previousStudent_grade=2, shane_grade=3)])
            for name in ('train.png', 'test.png'):
                Image.new('L', (32, 32)).save(folder / name)
            manifest, ratings = folder / 'manifest.csv', folder / 'ratings.csv'
            base.to_csv(manifest, index=False)
            labels = base[['image_id', 'previousStudent_grade', 'shane_grade']].copy()
            labels['pending_grade'] = ''
            labels['shane_grade'] = labels.shane_grade.astype(object)
            labels.loc[1, 'shane_grade'] = ''
            labels.to_csv(ratings, index=False)
            argv = ['training.py', '--grader', 'soft', '--dataset_csv_path', str(manifest),
                    '--ratings_csv_path', str(ratings), '--output_dir', str(folder), '--epochs', '1']
            def fake_train(model, train_loader, val_loader, criterion, optimizer, device):
                self.assertIsNone(criterion.weight)
                for loader in (train_loader, val_loader):
                    _, targets = next(iter(loader))
                    self.assertEqual(targets.shape[1], 5)
                    torch.testing.assert_close(targets.sum(1), torch.ones(len(targets)))
                return 1., 1.
            # Exercise real targets, patch inheritance, save metadata, state loading and evaluation.
            with patch('sys.argv', argv), patch.object(training, 'train', side_effect=fake_train), \
                 contextlib.redirect_stdout(io.StringIO()), patch('PIL.Image.open', wraps=Image.open) as opened:
                training.main()
            self.assertNotIn(str(folder / 'test.png'), [c.args[0] for c in opened.call_args_list])
            run = folder / 'Auto_qc_soft_SGD_seed_42'
            config = json.loads((run / 'run_config.json').read_text())
            self.assertIsNone(config['class_weights'])
            self.assertEqual(config['grader_columns'], ['previousStudent_grade', 'shane_grade', 'pending_grade'])
            metrics = json.loads((run / 'validation_metrics.json').read_text())
            self.assertEqual(metrics['references']['previousStudent_grade']['image_count'], 1)
            self.assertEqual(metrics['references']['shane_grade']['image_count'], 0)
            self.assertTrue(np.isfinite(metrics['soft_cross_entropy']))
            predictions = pd.read_csv(run / 'validation_predictions.csv')
            self.assertEqual(predictions.rating_count.tolist(), [1])
            self.assertEqual(predictions.target_grade_2.tolist(), [1.])
            # Saved targets remain independent of later input edits.
            ratings.write_text('changed')
            with self.assertRaises(FileExistsError):
                evaluate_saved_run(run)


if __name__ == '__main__':
    unittest.main()
