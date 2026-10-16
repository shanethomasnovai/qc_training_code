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

import training
from class_weights import baseline_class_weights
from soft_labels import TARGET_COLUMNS, construct_targets, read_table


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'prepared_visus_nhp/manifest.csv'
BASELINES = {
    'previousStudent': ROOT / 'runs/previousStudent_20261013_130615_231_f9f4cb56',
    'shane': ROOT / 'runs/shane_20261013_121344_539_933fb0f3',
}


class WeightedTrainingChecks(unittest.TestCase):
    def test_baseline_reconstruction_and_saved_vector_precedence(self):
        for grader, counts in [('previousStudent', [73, 163, 156, 112, 4]),
                               ('shane', [56, 176, 181, 87, 8])]:
            values, source = baseline_class_weights(BASELINES[grader], grader, MANIFEST)
            expected = [508 / (5 * count) for count in counts]
            self.assertEqual(values, expected)
            self.assertEqual(source['training_image_counts'], counts)
            # Independently execute the original patch-count formula.
            patches = [(torch.zeros(1), torch.tensor(i)) for i, n in enumerate(counts) for _ in range(n * 18)]
            loaders = training.make_loaders(patches, patches[:1], 16, 42)
            with contextlib.redirect_stdout(io.StringIO()):
                original, _ = training.class_balance(*loaders)
            torch.testing.assert_close(torch.tensor(values), torch.tensor([original[i] for i in range(5)]), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'weight_reference_grader'):
            baseline_class_weights(BASELINES['shane'])
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            baseline_class_weights(BASELINES['previousStudent'], 'shane')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            df = read_table(MANIFEST)
            df.loc[df.split.ne('train'), ['previousStudent_grade', 'shane_grade']] = 5
            df.to_csv(folder / 'manifest.csv', index=False)
            pd.DataFrame([{'grader': 'previousStudent', 'dataset_manifest': 'manifest.csv'}]).to_csv(folder / 'model_params.csv', index=False)
            values, _ = baseline_class_weights(folder)
            self.assertEqual(values, [508 / (5 * n) for n in [73, 163, 156, 112, 4]])
            config = {'grader': 'previousStudent', 'loss': 'CrossEntropyLoss', 'class_weights': [1., 2., 3., 4., 5.]}
            (folder / 'run_config.json').write_text(json.dumps(config))
            values, source = baseline_class_weights(folder)
            self.assertEqual(values, config['class_weights'])
            self.assertEqual(source['method'], 'saved run_config.json class_weights')
            config['class_weights'] = [1, 2, 3, 4, float('inf')]
            (folder / 'run_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'finite positive'):
                baseline_class_weights(folder)

    def test_partial_coverage_distribution_invariance_and_weighted_manual_loss(self):
        rows = pd.DataFrame({'image_id': ['two', 'three', 'four', 'five'],
                             'a_grade': [2, 2, 2, 2], 'b_grade': [3, 3, 3, 3],
                             'c_grade': ['', 3, 2, 2], 'd_grade': ['', '', 3, 3],
                             'e_grade': ['', '', '', 3]})
        result = construct_targets(rows, [c for c in rows if c.endswith('_grade')])
        self.assertEqual(result.rating_count.tolist(), [2, 3, 4, 5])
        self.assertTrue(result.has_ratings.all())
        np.testing.assert_allclose(result[TARGET_COLUMNS].iloc[:2], [[0, .5, .5, 0, 0], [0, 1/3, 2/3, 0, 0]])
        weights = torch.tensor([1., 2., 4., 3., 5.], dtype=torch.double)
        probabilities = torch.tensor([[.1, .2, .3, .25, .15]], dtype=torch.double)
        logits = probabilities.log().repeat(4, 1).requires_grad_()
        targets = torch.tensor(result[TARGET_COLUMNS].to_numpy(), dtype=torch.double)
        loss = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction='none')
        torch.testing.assert_close(loss[0], loss[2], rtol=0, atol=0)
        expected = -(.5 * 2 * np.log(.2) + .5 * 4 * np.log(.3))
        self.assertAlmostEqual(loss[0].item(), expected)
        mean = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction='mean')
        self.assertAlmostEqual(mean.item(), loss.sum().item() / 4)
        mean.backward()
        self.assertTrue(torch.isfinite(mean))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_mean_denominators(self):
        logits = torch.tensor([[.2, -.3, .1, .5, -.1], [-.1, .3, -.2, .6, .1]], dtype=torch.double)
        labels = torch.tensor([1, 2])
        probabilities = torch.nn.functional.one_hot(labels, 5).double()
        weights = torch.tensor([1., 2., 4., 3., 5.], dtype=torch.double)
        hard_sum = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction='sum')
        soft_sum = torch.nn.functional.cross_entropy(logits, probabilities, weight=weights, reduction='sum')
        torch.testing.assert_close(hard_sum, soft_sum)
        hard = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        soft = torch.nn.functional.cross_entropy(logits, probabilities, weight=weights)
        torch.testing.assert_close(hard, hard_sum / weights[labels].sum())
        torch.testing.assert_close(soft, soft_sum / len(labels))
        self.assertFalse(torch.isclose(hard, soft))

    def test_counting_is_rng_neutral_and_loader_order_is_independent(self):
        data = [(torch.tensor(i), torch.tensor(i % 5)) for i in range(25)]
        train, val = training.make_loaders(data, data, 4, 42)
        torch.manual_seed(71)
        state = torch.get_rng_state().clone()
        loader_state = train.generator.get_state().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            training.class_balance(train, val)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_state, train.generator.get_state()))
        other_train, _ = training.make_loaders(data, data, 4, 42)
        for _ in range(2):
            first = torch.cat([inputs for inputs, _ in train])
            torch.rand(123)
            list(val)
            second = torch.cat([inputs for inputs, _ in other_train])
            self.assertTrue(torch.equal(first, second))

    def test_main_modes_share_initial_state_and_epoch_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            ratings = Path(folder) / 'ratings.csv'
            read_table(MANIFEST)[['image_id', 'previousStudent_grade', 'shane_grade']].to_csv(ratings, index=False)
            captured = []
            for mode in ('previousStudent', 'shane', 'soft'):
                record = {'states': [], 'orders': [], 'dropout': []}
                soft = mode == 'soft'
                patches = [(torch.full((1, 8, 8), float(i)),
                            torch.nn.functional.one_hot(torch.tensor(i % 5), 5).float() if soft else torch.tensor(i % 5), split)
                           for split in ('train', 'val') for i in range(15)]
                argv = ['training.py', '--grader', mode, '--dataset_csv_path', str(MANIFEST),
                        '--output_dir', folder, '--epochs', '2', '--batch_size', '4']
                if soft:
                    argv += ['--ratings_csv_path', str(ratings), '--class_weights_from', str(BASELINES['previousStudent'])]
                def fake_train(model, train_loader, val_loader, criterion, optimizer, device):
                    record['states'].append({k: v.clone() for k, v in model.state_dict().items()})
                    record['orders'].append(torch.cat([inputs[:, 0, 0, 0] for inputs, _ in train_loader]))
                    list(val_loader)
                    record['dropout'].append(torch.nn.functional.dropout(torch.ones(200), p=.5, training=True))
                    return 1., 1.
                with patch('sys.argv', argv), patch.object(training, 'subsample_subimages', return_value=(patches, None)), \
                     patch.object(training, 'train', side_effect=fake_train), patch.object(training, 'save_model'), \
                     patch.object(training, 'evaluate_saved_run'), contextlib.redirect_stdout(io.StringIO()):
                    training.main()
                captured.append(record)
            for record in captured[1:]:
                for epoch in range(2):
                    for key in captured[0]['states'][epoch]:
                        self.assertTrue(torch.equal(captured[0]['states'][epoch][key], record['states'][epoch][key]))
                    self.assertTrue(torch.equal(captured[0]['orders'][epoch], record['orders'][epoch]))
                    self.assertTrue(torch.equal(captured[0]['dropout'][epoch], record['dropout'][epoch]))


if __name__ == '__main__':
    unittest.main()
