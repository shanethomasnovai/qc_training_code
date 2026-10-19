import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from autoqc.binary_metrics import binary_confusion_matrix
from autoqc.distribution_metrics import PROBABILITY_COLUMNS, TARGET_COLUMNS, calculate_distribution_metrics
from autoqc.score_distributions import prepare_report, score_runs
from autoqc.soft_labels import construct_targets
from autoqc.validation import calculate_combined_metrics, calculate_validation_metrics, evaluate_validation
from tests import test_distribution_metrics
from tests.test_validation import PixelModel


class BinaryMetricChecks(unittest.TestCase):
    columns = ['a_grade', 'b_grade', 'c_grade', 'd_grade', 'e_grade']

    def evaluate(self, ratings, probabilities, threshold=0.5):
        reference = pd.DataFrame(ratings, columns=self.columns)
        return calculate_combined_metrics(reference, probabilities, self.columns, threshold)

    def test_mapping_missing_votes_ties_and_majority_not_five_grade_mode(self):
        ratings = [[grade, None, None, None, None] for grade in range(1, 6)]
        ratings += [[2, 3, None, None, None], [1, 1, 3, 4, 5]]
        scores, details = self.evaluate(ratings, np.tile([0, 1, 0, 0, 0], (7, 1)))
        self.assertEqual(details.binary_reference.tolist(),
                         ['acceptable', 'acceptable', 'unacceptable', 'unacceptable',
                          'unacceptable', 'acceptable', 'unacceptable'])
        self.assertEqual(details.rating_count.tolist(), [1, 1, 1, 1, 1, 2, 5])
        self.assertEqual(details.acceptable_vote_count.tolist(), [1, 1, 0, 0, 0, 1, 2])
        self.assertEqual(details.unacceptable_vote_count.tolist(), [0, 0, 1, 1, 1, 1, 3])
        self.assertEqual(details.binary_reference_tied.tolist(), [False] * 5 + [True, False])
        self.assertEqual(details.consensus_reference_grade.iloc[-1], 1)
        self.assertEqual(scores['binary']['reference_tie_count'], 1)
        self.assertAlmostEqual(scores['binary']['reference_tie_percentage'], 100 / 7)
        self.assertEqual(scores['binary']['unique_majority']['image_count'], 6)
        np.testing.assert_allclose(details.unacceptable_vote_proportion, [0, 0, 1, 1, 1, .5, .6])

    def test_threshold_exactly_half_summed_probability_and_configurability(self):
        probabilities = [[.1, .4, .2, .15, .15], [.1, .35, .2, .2, .15]]
        ratings = [[2, 3, None, None, None]] * 2
        scores, details = self.evaluate(ratings, probabilities)
        self.assertEqual(details.predicted_grade.tolist(), [2, 2])
        self.assertEqual(details.binary_prediction.tolist(), ['unacceptable'] * 2)
        np.testing.assert_allclose(details.p_acceptable, [.5, .45])
        np.testing.assert_allclose(details.p_unacceptable, [.5, .55])
        self.assertEqual(scores['binary']['decision_threshold'], .5)
        changed, changed_details = self.evaluate(ratings, probabilities, .6)
        self.assertEqual(changed_details.binary_prediction.tolist(), ['acceptable'] * 2)
        self.assertEqual(changed['binary']['decision_threshold'], .6)
        for key in scores:
            if key != 'binary':
                self.assertEqual(scores[key], changed[key])
        for threshold in [np.nan, np.inf, -.1, 1.1]:
            with self.assertRaises(ValueError):
                self.evaluate(ratings, probabilities, threshold)

    def test_hand_calculated_confusion_metrics_and_equal_image_weight(self):
        ratings = [[3, 3, 3, 3, 3], [5, None, None, None, None],
                   [3, None, None, None, None], [1, None, None, None, None],
                   [2, None, None, None, None], [2, 3, None, None, None]]
        probabilities = np.eye(5)[np.array([3, 4, 1, 2, 3, 1]) - 1]
        scores, details = self.evaluate(ratings, probabilities)
        binary = scores['binary']
        self.assertEqual(details.binary_confusion_outcome.tolist(), ['TP', 'TP', 'FN', 'TN', 'FP', 'TN'])
        self.assertEqual([binary[k] for k in ('TP', 'TN', 'FP', 'FN')], [2, 2, 1, 1])
        for key in ('exact_accuracy', 'sensitivity', 'specificity', 'balanced_accuracy', 'precision', 'f1'):
            self.assertAlmostEqual(binary[key], 2 / 3)
        self.assertEqual(binary['reference_acceptable_count'], 3)
        self.assertEqual(binary['reference_unacceptable_count'], 3)
        unique = binary['unique_majority']
        self.assertEqual(unique['image_count'], 5)
        self.assertEqual(unique['specificity'], .5)
        self.assertAlmostEqual(unique['sensitivity'], 2 / 3)
        self.assertAlmostEqual(unique['balanced_accuracy'], (2 / 3 + .5) / 2)
        self.assertEqual(binary_confusion_matrix(binary).to_numpy().tolist(), [[2, 1], [1, 2]])
        targets = construct_targets(pd.DataFrame(ratings, columns=self.columns), self.columns)
        old, old_details = calculate_distribution_metrics(targets[TARGET_COLUMNS], probabilities)
        for key in ('earth_movers_distance', 'soft_cross_entropy'):
            self.assertEqual(scores[key], old[key])
            np.testing.assert_array_equal(details[key], old_details[key])
        consensus, _, _ = calculate_validation_metrics(details.consensus_reference_grade, details.predicted_grade)
        self.assertEqual(scores['consensus_reference_exact_accuracy'], consensus['exact_accuracy'])
        self.assertEqual(scores['consensus_reference_mae'], consensus['mean_absolute_grade_error'])
        self.assertEqual(scores['consensus_reference_quadratic_weighted_kappa'], consensus['quadratic_weighted_kappa'])

    def test_undefined_metrics_and_empty_unique_majority_are_null_with_reasons(self):
        for rating, prediction, absent in [(1, 1, 'sensitivity'), (3, 3, 'specificity')]:
            scores, _ = self.evaluate([[rating, None, None, None, None]], np.eye(5)[[prediction - 1]])
            binary = scores['binary']
            for key in (absent, 'balanced_accuracy'):
                self.assertIsNone(binary[key])
                self.assertTrue(binary['undefined_reasons'][key])
            json.dumps(binary, allow_nan=False)
        scores, _ = self.evaluate([[2, 3, None, None, None]], [[1, 0, 0, 0, 0]])
        self.assertIsNone(scores['binary']['precision'])
        self.assertIsNone(scores['binary']['f1'])
        unique = scores['binary']['unique_majority']
        self.assertEqual(unique['image_count'], 0)
        for key in ('sensitivity', 'specificity', 'balanced_accuracy', 'exact_accuracy'):
            self.assertIsNone(unique[key])
            self.assertTrue(unique['undefined_reasons'][key])

    def test_saved_runs_identical_reference_alignment_threshold_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            reference, baseline = test_distribution_metrics.DistributionMetricChecks().make_runs(folder)
            path = baseline / 'validation_predictions.csv'
            pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
            before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            metrics, details = prepare_report(baseline, reference)
            self.assertEqual(details.image_id.tolist(), ['val_b', 'val_a'])
            self.assertEqual(details.binary_confusion_outcome.tolist(), ['FP', 'TP'])
            summary = score_runs([reference, baseline], reference, folder / 'report', .6)
            self.assertEqual(summary.binary_decision_threshold.tolist(), [.6, .6])
            self.assertEqual(summary.binary_exact_accuracy.tolist(), [1., 1.])
            self.assertEqual(summary.reference_targets_sha256.nunique(), 1)
            for run in [reference, baseline]:
                output = folder / 'report' / run.name
                report = json.loads((output / 'distribution_metrics.json').read_text())
                self.assertEqual(report['binary']['unique_majority']['image_count'], 1)
                matrix = pd.read_csv(output / 'binary_confusion_matrix.csv', index_col=0)
                self.assertEqual(matrix.index.tolist(), ['acceptable', 'unacceptable'])
                self.assertEqual(matrix.columns.tolist(), ['predicted_acceptable', 'predicted_unacceptable'])
                self.assertEqual(matrix.to_numpy().tolist(), [[1, 0], [0, 1]])
            for p, contents in before.items():
                self.assertEqual(p.read_bytes(), contents)

    def test_automatic_soft_validation_emits_binary_outputs_without_opening_other_splits(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            image = folder / 'validation.png'
            Image.fromarray(np.zeros((768, 768), dtype=np.uint8)).save(image)
            frame = pd.DataFrame({
                'image_id': ['v', 't', 'holdout'],
                'image_path': [str(image), str(folder / 'DO_NOT_OPEN_train.png'), str(folder / 'DO_NOT_OPEN_test.png')],
                'patient_id': ['v', 't', 'h'], 'project': ['other', 'other', 'VISUS-NHP'],
                'split': ['validation', 'train', 'test'], 'a_grade': [2, 1, 5], 'b_grade': [3, 2, 4],
            })
            metrics, predictions = evaluate_validation(PixelModel(), frame, 'soft', folder / 'out',
                                                       grader_columns=['a_grade', 'b_grade'], binary_threshold=.6)
            self.assertEqual(metrics['binary']['image_count'], 1)
            self.assertEqual(metrics['binary']['decision_threshold'], .6)
            self.assertEqual(predictions.binary_reference.tolist(), ['acceptable'])
            self.assertTrue((folder / 'out' / 'binary_confusion_matrix.csv').is_file())
            saved = json.loads((folder / 'out' / 'validation_metrics.json').read_text())
            self.assertEqual(saved['binary'], metrics['binary'])
            with self.assertRaises(FileExistsError):
                evaluate_validation(PixelModel(), frame, 'soft', folder / 'out', grader_columns=['a_grade', 'b_grade'])


if __name__ == '__main__':
    unittest.main()
