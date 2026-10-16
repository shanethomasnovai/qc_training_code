import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import cohen_kappa_score

from distribution_metrics import PROBABILITY_COLUMNS, TARGET_COLUMNS, calculate_distribution_metrics
from score_distributions import prepare_report, score_runs
from soft_labels import construct_targets
from validation import calculate_combined_metrics


class DistributionMetricChecks(unittest.TestCase):
    def test_combined_reference_examples_minority_matches_and_summary(self):
        ratings = [[2, 3, 3, None, None], [2, 3, None, None, None],
                   [2, 2, 4, 4, None], [1, 3, 3, 5, 5], [4, None, None, None, None]]
        columns = [f'{letter}_grade' for letter in 'abcde']
        reference = pd.DataFrame(ratings, columns=columns)
        reference['image_id'] = ['a', 'b', 'c', 'd', 'e']
        targets = construct_targets(reference, columns)
        predicted = np.array([2, 3, 3, 5, 4])
        probabilities = np.eye(5)[predicted - 1]
        scores, details = calculate_combined_metrics(targets, probabilities, columns)
        self.assertEqual(details.consensus_reference_grade.tolist(), [3, 2, 2, 3, 4])
        self.assertEqual(details.reference_tied.tolist(), [False, True, True, True, False])
        self.assertEqual(details.rating_count.tolist(), [3, 2, 4, 5, 1])
        self.assertEqual(details.matches_any_grader.tolist(), [True, True, False, True, True])
        self.assertEqual(details.consensus_exact_match.tolist(), [False, False, False, False, True])
        self.assertEqual(details.absolute_grade_error.tolist(), [1, 1, 1, 2, 0])
        self.assertEqual(scores['matches_any_grader_accuracy'], .8)
        self.assertEqual(scores['consensus_reference_exact_accuracy'], .2)
        self.assertEqual(scores['consensus_reference_mae'], 1.)
        self.assertEqual(scores['reference_tie_count'], 3)
        self.assertEqual(scores['reference_tie_percentage'], 60.)
        self.assertAlmostEqual(scores['consensus_reference_quadratic_weighted_kappa'],
                               cohen_kappa_score([3, 2, 2, 3, 4], predicted, labels=range(1, 6), weights='quadratic'))
        old_scores, old_details = calculate_distribution_metrics(targets[TARGET_COLUMNS], probabilities)
        for key in ('earth_movers_distance', 'soft_cross_entropy'):
            self.assertEqual(scores[key], old_scores[key])
            np.testing.assert_array_equal(details[key], old_details[key])
        undefined, _ = calculate_combined_metrics(targets.iloc[-1:], probabilities[-1:], columns)
        self.assertIsNone(undefined['consensus_reference_quadratic_weighted_kappa'])
        self.assertEqual(undefined['consensus_reference_quadratic_weighted_kappa_undefined_reason'], 'Zero expected disagreement')

    def test_analytic_distances_cross_entropy_and_zeros(self):
        targets = np.array([[1, 0, 0, 0, 0], [0, .5, .5, 0, 0], [0, 1, 0, 0, 0]])
        probabilities = np.array([[0, 0, 0, 0, 1], [0, .5, .5, 0, 0], [0, 0, 1, 0, 0]])
        scores, details = calculate_distribution_metrics(targets, probabilities)
        np.testing.assert_allclose(details.earth_movers_distance, [4, 0, 1])
        self.assertAlmostEqual(scores['earth_movers_distance'], 5 / 3)
        self.assertAlmostEqual(details.soft_cross_entropy.iloc[1], np.log(2))
        self.assertTrue(np.isfinite(details.soft_cross_entropy).all())
        perfect, _ = calculate_distribution_metrics(targets[:1], targets[:1])
        self.assertEqual(perfect['earth_movers_distance'], 0)
        self.assertEqual(perfect['soft_cross_entropy'], 0)

    def test_matches_independent_wasserstein_and_equal_image_weighting(self):
        rng = np.random.default_rng(42)
        targets, probabilities = rng.dirichlet(np.ones(5), 15), rng.dirichlet(np.ones(5), 15)
        scores, details = calculate_distribution_metrics(targets, probabilities)
        expected = [wasserstein_distance(np.arange(1, 6), np.arange(1, 6), q, p) for q, p in zip(targets, probabilities)]
        np.testing.assert_allclose(details.earth_movers_distance, expected)
        self.assertAlmostEqual(scores['soft_cross_entropy'], float(-(targets * np.log(probabilities)).sum(1).mean()))

    def test_rejects_invalid_distributions(self):
        valid = np.ones((1, 5)) / 5
        for bad in [np.zeros((0, 5)), np.ones((1, 4)), np.zeros((1, 5)),
                    [[np.nan, .2, .2, .2, .2]], [[-.1, .1, .2, .3, .5]]]:
            with self.assertRaises(ValueError):
                calculate_distribution_metrics(bad, valid)
        with self.assertRaises(ValueError):
            calculate_distribution_metrics(valid, valid * 1.01)
        drift, _ = calculate_distribution_metrics(valid, valid * (1 + 1e-7))
        self.assertAlmostEqual(drift['earth_movers_distance'], 0)

    def make_runs(self, folder):
        reference, baseline = folder / 'soft', folder / 'hard'
        reference.mkdir()
        baseline.mkdir()
        manifest = pd.DataFrame({
            'image_id': ['training', 'val_b', 'test', 'val_a'],
            'patient_id': ['t', 'vb', 'test', 'va'],
            'project': ['other', 'other', 'VISUS-NHP', 'other'],
            'split': ['train', 'validation', 'test', 'validation'],
            'a_grade': [1, 2, 5, 3], 'b_grade': ['', 3, '', 3],
        })
        targets = construct_targets(manifest, ['a_grade', 'b_grade'])
        targets.to_csv(reference / 'targets.csv', index=False)
        (reference / 'run_config.json').write_text(json.dumps({'grader_columns': ['a_grade', 'b_grade']}))
        predictions = pd.DataFrame({'image_id': ['val_a', 'val_b']})
        predictions[PROBABILITY_COLUMNS] = [[0, 0, 1, 0, 0], [0, .5, .5, 0, 0]]
        for run in [reference, baseline]:
            manifest.to_csv(run / 'manifest.csv', index=False)
            predictions.to_csv(run / 'validation_predictions.csv', index=False)
        return reference, baseline

    def test_saved_predictions_keyed_join_split_guards_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            reference, baseline = self.make_runs(folder)
            metrics, details = prepare_report(baseline, reference)
            self.assertEqual(details.image_id.tolist(), ['val_b', 'val_a'])
            self.assertEqual(metrics['earth_movers_distance'], 0)
            self.assertAlmostEqual(metrics['soft_cross_entropy'], np.log(2) / 2)
            self.assertEqual(details.predicted_grade.tolist(), [2, 3])
            self.assertEqual(details.consensus_reference_grade.tolist(), [2, 3])
            self.assertEqual(metrics['consensus_reference_exact_accuracy'], 1.)
            self.assertEqual(metrics['reference_tie_count'], 1)
            self.assertEqual(metrics['reference_tie_percentage'], 50.)
            original = (baseline / 'validation_predictions.csv').read_bytes()
            summary = score_runs([reference, baseline], reference, folder / 'reports')
            self.assertEqual(summary.image_count.tolist(), [2, 2])
            self.assertEqual(original, (baseline / 'validation_predictions.csv').read_bytes())
            with self.assertRaises(FileExistsError):
                score_runs([baseline], reference, folder / 'reports')
            changed = pd.read_csv(baseline / 'manifest.csv')
            changed.loc[0, 'split'] = 'validation'
            changed.to_csv(baseline / 'manifest.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'split assignments differ'):
                prepare_report(baseline, reference)

    def test_missing_duplicate_nonvalidation_predictions_and_corrupt_targets(self):
        for error in ['missing', 'duplicate', 'test', 'targets', 'prediction_split', 'predicted_grade']:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                reference, baseline = self.make_runs(Path(tmp))
                path = baseline / 'validation_predictions.csv'
                predictions = pd.read_csv(path)
                if error == 'missing':
                    predictions = predictions.iloc[:1]
                elif error == 'duplicate':
                    predictions = pd.concat([predictions, predictions.iloc[:1]])
                elif error == 'test':
                    predictions.loc[0, 'image_id'] = 'test'
                elif error == 'prediction_split':
                    predictions['split'] = 'train'
                elif error == 'predicted_grade':
                    predictions['predicted_grade'] = 5
                else:
                    targets = pd.read_csv(reference / 'targets.csv')
                    targets.loc[1, TARGET_COLUMNS] = [1, 0, 0, 0, 0]
                    targets.to_csv(reference / 'targets.csv', index=False)
                predictions.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    prepare_report(baseline, reference)


if __name__ == '__main__':
    unittest.main()
