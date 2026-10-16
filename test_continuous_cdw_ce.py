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
from torch.nn import functional as F

import training
from dataset import ImageQualityDataset, subsample_subimages
from losses import ContinuousCDWCELoss, continuous_cdw_ce, make_criterion
from score_distributions import prepare_report
from soft_labels import TARGET_COLUMNS, construct_continuous_targets, load_soft_manifest, read_table
from validation import calculate_combined_metrics, evaluate_saved_run, image_transform


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'prepared_visus_nhp/manifest.csv'
RATINGS = ROOT / 'runs/soft_20261016_122507_325_f156d443/ratings_input.xlsx'


def supplied_loss(logits, targets, alpha=1.0):
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(1, 6, device=logits.device, dtype=logits.dtype)
    distances = torch.abs(classes.unsqueeze(0) - targets.unsqueeze(1))
    weights = distances.pow(alpha)
    log_not_probs = torch.log((1 - probs).clamp(min=1e-7))
    loss = -(weights * log_not_probs).sum(dim=1)
    return loss.mean()


class ContinuousCDWChecks(unittest.TestCase):
    def test_mean_targets_missing_decimal_scale_and_equal_image_weight(self):
        ratings = pd.DataFrame({'image_id': list('abcdef'), 'a_grade': [2, 2, 4, 1, 5, ''],
                                'b_grade': [3, '', None, '', '', ''], 'c_grade': [3, 3, '', '', '', '']})
        targets = construct_continuous_targets(ratings, ['a_grade', 'b_grade', 'c_grade'])
        np.testing.assert_allclose(targets.continuous_target, [8/3, 2.5, 4, 1, 5, np.nan], equal_nan=True)
        self.assertEqual(targets.rating_count.tolist(), [3, 2, 1, 1, 1, 0])
        self.assertFalse(targets.has_ratings.iloc[-1])
        np.testing.assert_allclose(targets[TARGET_COLUMNS].iloc[0], [0, 1/3, 2/3, 0, 0])
        pd.testing.assert_frame_equal(targets[ratings.columns], ratings)
        repeated = pd.DataFrame({'image_id': ['two', 'four'], 'a_grade': [2, 2], 'b_grade': [3, 3],
                                 'c_grade': ['', 2], 'd_grade': ['', 3]})
        means = construct_continuous_targets(repeated, ['a_grade', 'b_grade', 'c_grade', 'd_grade'])
        self.assertEqual(means.continuous_target.tolist(), [2.5, 2.5])
        logits = torch.tensor([[.2, -.3, .4, .1, -.1]])
        self.assertEqual(continuous_cdw_ce(logits, torch.tensor([2.5])).item(),
                         continuous_cdw_ce(logits.repeat(2, 1), torch.tensor(means.continuous_target.to_numpy(), dtype=torch.float32)).item())
        bad = ratings.copy()
        bad.loc[0, 'a_grade'] = 2.5
        with self.assertRaisesRegex(ValueError, 'Invalid rating'):
            construct_continuous_targets(bad, ['a_grade'])

    def test_exact_formula_and_gradients(self):
        logits = torch.tensor([[.2, -.3, .1, .5, -.1], [.8, -.7, .4, -.2, .3], [.1, .2, -.3, .4, .5]], dtype=torch.double, requires_grad=True)
        targets = torch.tensor([1., 8/3, 5.], dtype=torch.double)
        for alpha in (.5, 1., 2.):
            actual, expected = continuous_cdw_ce(logits, targets, alpha), supplied_loss(logits, targets, alpha)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            reference = torch.autograd.grad(expected, logits, retain_graph=True)[0]
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual))
            self.assertTrue(torch.isfinite(gradient).all())

    def test_extreme_logits_clamp_saturation(self):
        logits = torch.tensor([[1000., -1000., -1000., -1000., -1000.]], requires_grad=True)
        for target, distance in [(1., 0.), (1.5, .5), (5., 4.)]:
            loss = continuous_cdw_ce(logits, torch.tensor([target]))
            self.assertAlmostEqual(loss.item(), -distance * np.log(1e-7), places=5)
            self.assertTrue(torch.equal(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits)))

    def test_invalid_inputs_and_incompatible_options(self):
        logits, targets = torch.zeros(2, 5), torch.tensor([2.5, 4.])
        for alpha in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                continuous_cdw_ce(logits, targets, alpha)
        for bad_logits, bad_targets in [(torch.zeros(2, 4), targets), (logits, targets[:, None]),
                                        (logits, torch.tensor([0., 4.])), (logits, torch.tensor([2., 6.])),
                                        (logits, torch.tensor([2, 4])), (logits, torch.tensor([float('nan'), 4.])),
                                        (logits + float('inf'), targets), (logits, targets.to('meta'))]:
            with self.assertRaises(ValueError):
                continuous_cdw_ce(bad_logits, bad_targets)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            make_criterion('continuous_cdw_ce', torch.ones(5))
        for options in [('--class_weights_from', 'some_baseline'), ('--weight_reference_grader', 'shane'),
                        ('--alpha', 'nan'), ('--alpha', '0')]:
            argv = ['training.py', '--grader', 'soft', '--loss', 'continuous_cdw_ce', '--ratings_csv_path', 'unused',
                    '--dataset_csv_path', 'unused', '--output_dir', 'unused', *options]
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.main()
        self.assertIsInstance(make_criterion(), torch.nn.CrossEntropyLoss)
        self.assertIsInstance(make_criterion('continuous_cdw_ce'), ContinuousCDWCELoss)

    def test_real_image_train_validation_smoke(self):
        if not RATINGS.is_file():
            self.skipTest('Local private ratings snapshot unavailable')
        with contextlib.redirect_stdout(io.StringIO()):
            data, columns, _ = load_soft_manifest(MANIFEST, RATINGS, exclude_known_unmatched=True)
            data = construct_continuous_targets(data, columns)
            datasets = []
            for split in ('train', 'validation'):
                row = data[data.split.eq(split) & data.has_ratings].iloc[:1].copy().reset_index(drop=True)
                row['image_filename'] = row.image_path
                dataset = ImageQualityDataset(row, image_transform(), scalar_target_column='continuous_target')
                patches, _ = subsample_subimages(dataset, method='all', df=row)
                self.assertEqual(len(patches), 9)
                self.assertEqual(patches[0][1].ndim, 0)
                self.assertAlmostEqual(patches[0][1].item(), row.continuous_target.iloc[0], places=6)
                datasets.append(patches[:2])
            train_loader, val_loader = training.make_loaders(*datasets, batch_size=2, seed=42)
            model = training.initialise_model(42)
            calls = []
            class RecordingLoss(ContinuousCDWCELoss):
                def forward(self, logits, targets):
                    calls.append((torch.is_grad_enabled(), targets.shape, logits.shape))
                    self.assert_device = targets.device == logits.device
                    return super().forward(logits, targets)
            criterion = RecordingLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
            before = training.model_fingerprint(model)
            losses = training.train(model, train_loader, val_loader, criterion, optimizer, 'cpu')
            self.assertTrue(np.isfinite(losses).all())
            self.assertEqual([entry[0] for entry in calls], [True, False])
            self.assertTrue(criterion.assert_device)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertNotEqual(before, training.model_fingerprint(model))

    def test_save_reload_evaluation_and_shared_reference_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = pd.DataFrame([{'image_id': split, 'image_path': str(folder / f'{split}.png'),
                                  'patient_id': split, 'project': 'VISUS-NHP' if split == 'test' else 'other',
                                  'project_site': 'site', 'split': split, 'previousStudent_grade': 2,
                                  'shane_grade': 4 if split == 'validation' else 3} for split in ('train', 'validation', 'test')])
            for split in base.split:
                Image.new('L', (768, 768), 100).save(folder / f'{split}.png')
            manifest, ratings = folder / 'manifest.csv', folder / 'ratings.csv'
            base.to_csv(manifest, index=False)
            panel = base[['image_id', 'previousStudent_grade', 'shane_grade']].copy()
            panel['extra_grade'] = [3, '', '']
            panel.to_csv(ratings, index=False)
            configs = []
            for loss in ('cross_entropy', 'continuous_cdw_ce'):
                argv = ['training.py', '--grader', 'soft', '--loss', loss, '--dataset_csv_path', str(manifest),
                        '--ratings_csv_path', str(ratings), '--output_dir', str(folder), '--model_name', loss, '--epochs', '1']
                def fake_train(model, train_loader, val_loader, criterion, optimizer, device):
                    train_target, val_target = next(iter(train_loader))[1], next(iter(val_loader))[1]
                    if loss == 'continuous_cdw_ce':
                        self.assertIsInstance(criterion, ContinuousCDWCELoss)
                        torch.testing.assert_close(train_target, torch.full_like(train_target, 8/3))
                        torch.testing.assert_close(val_target, torch.full_like(val_target, 3.))
                    else:
                        self.assertEqual(train_target.shape[1], 5)
                    return 1., 1.
                with patch('sys.argv', argv), patch.object(training, 'train', side_effect=fake_train), \
                     patch('PIL.Image.open', wraps=Image.open) as opened, contextlib.redirect_stdout(io.StringIO()):
                    training.main()
                self.assertNotIn(str(folder / 'test.png'), [call.args[0] for call in opened.call_args_list])
                config = json.loads((folder / loss / 'run_config.json').read_text())
                configs.append(config)
            saved = pd.read_csv(folder / 'continuous_cdw_ce/targets.csv')
            self.assertAlmostEqual(saved.continuous_target.iloc[0], 8/3)
            pd.testing.assert_frame_equal(saved[base.columns], base, check_dtype=False)
            self.assertEqual(configs[1]['loss'], 'continuous_cdw_ce')
            self.assertEqual(configs[1]['alpha'], 1.)
            self.assertEqual(configs[1]['class_weight_mode'], 'none')
            self.assertIsNone(configs[1]['class_weights'])
            self.assertEqual(configs[0]['initial_model_sha256'], configs[1]['initial_model_sha256'])
            first, _ = prepare_report(folder / 'cross_entropy', folder / 'cross_entropy')
            second, details = prepare_report(folder / 'continuous_cdw_ce', folder / 'cross_entropy')
            for key in ('earth_movers_distance', 'soft_cross_entropy', 'matches_any_grader_accuracy',
                        'consensus_reference_exact_accuracy', 'consensus_reference_mae',
                        'consensus_reference_quadratic_weighted_kappa'):
                self.assertEqual(first[key], second[key])
            self.assertEqual(details.consensus_reference_grade.tolist(), [2])
            self.assertNotEqual(details.consensus_reference_grade.iloc[0], saved.continuous_target.iloc[1])
            with self.assertRaises(FileExistsError):
                evaluate_saved_run(folder / 'continuous_cdw_ce')


if __name__ == '__main__':
    unittest.main()
