import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autoqc.analyse_binary_cases import analyse
from tests import test_distribution_metrics


class BinaryCaseReviewChecks(unittest.TestCase):
    def test_keyed_review_counts_blinded_template_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            reference, baseline = test_distribution_metrics.DistributionMetricChecks().make_runs(folder)
            targets_path = reference / 'targets.csv'
            targets = pd.read_csv(targets_path)
            targets['image_path'] = [str(folder / f'{image_id}.png') for image_id in targets.image_id]
            targets.to_csv(targets_path, index=False)
            prediction_path = baseline / 'validation_predictions.csv'
            predictions = pd.read_csv(prediction_path).iloc[::-1]
            # Unacceptable val_a is missed; tied val_b remains predicted unacceptable.
            predictions.loc[predictions.image_id.eq('val_a'), 'probability_grade_3'] = 0
            predictions.loc[predictions.image_id.eq('val_a'), 'probability_grade_1'] = 1
            predictions.to_csv(prediction_path, index=False)
            original = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            output = folder / 'review'
            summary = analyse([reference, baseline], reference, baseline, output)
            self.assertEqual(summary.false_negative_count.tolist(), [0, 1])
            self.assertEqual(summary.tied_p_exactly_half_count.tolist(), [1, 1])
            self.assertEqual(summary.tied_p_0p4_to_0p6_count.tolist(), [1, 1])
            queue = pd.read_csv(output / 'image_review_queue.csv').set_index('image_id')
            self.assertEqual(queue.loc['val_a', 'runs_with_false_negative'], 1)
            self.assertEqual(queue.loc['val_b', 'runs_with_false_negative'], 0)
            self.assertEqual(queue.loc['val_b', 'runs_predicting_unacceptable'], 2)
            template = pd.read_csv(output / 'independent_rating_template.csv')
            self.assertEqual(template.columns.tolist(), ['image_id', 'image_path', 'new_independent_grade_1_to_5', 'reviewer_notes'])
            self.assertTrue(template.new_independent_grade_1_to_5.isna().all())
            metadata = json.loads((output / 'analysis_metadata.json').read_text())
            self.assertEqual(metadata['review_queue_count'], 2)
            self.assertEqual(metadata['unique_false_negative_images_across_runs'], 1)
            self.assertIn('Original image unavailable', (output / 'visual_review.html').read_text())
            for path, contents in original.items():
                self.assertEqual(path.read_bytes(), contents)
            with self.assertRaises(FileExistsError):
                analyse([reference, baseline], reference, baseline, output)
            with self.assertRaisesRegex(ValueError, 'Focus run'):
                analyse([reference], reference, baseline, folder / 'bad')


if __name__ == '__main__':
    unittest.main()
