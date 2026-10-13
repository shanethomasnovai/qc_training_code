import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, f1_score

from dataset import ImageQualityDataset, subsample_subimages
from validation import calculate_validation_metrics, evaluate_validation, image_transform


class PixelModel(torch.nn.Module):
    def forward(self, inputs):
        value = inputs.mean(dim=(1, 2, 3))
        return torch.stack([value * 0, value * 8 - 2, value * -5 + 1,
                            value * 3 - 1, value * -2], dim=1)


class ValidationChecks(unittest.TestCase):
    def test_nine_patch_probability_average_alignment_and_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            first, second = folder / 'variant_b.png', folder / 'variant_a.png'
            array = np.zeros((768, 768), dtype=np.uint8)
            for index, value in enumerate([0, 32, 64, 96, 128, 160, 192, 224, 255]):
                r, col = divmod(index, 3)
                array[r * 256:(r + 1) * 256, col * 256:(col + 1) * 256] = value
            Image.fromarray(array).save(first)
            Image.fromarray(np.full((768, 768), 220, dtype=np.uint8)).save(second)
            df = pd.DataFrame([
                {'image_id': 'variant_b', 'image_path': str(first), 'patient_id': 'p2', 'project': 'other', 'split': 'validation', 'shane_grade': 3},
                {'image_id': 'train', 'image_path': str(folder / 'DO_NOT_OPEN_train.png'), 'patient_id': 'p3', 'project': 'other', 'split': 'train', 'shane_grade': 2},
                {'image_id': 'test', 'image_path': str(folder / 'DO_NOT_OPEN_test.png'), 'patient_id': 'p4', 'project': 'VISUS-NHP', 'split': 'test', 'shane_grade': 4},
                {'image_id': 'variant_a', 'image_path': str(second), 'patient_id': 'p1', 'project': 'other', 'split': 'validation', 'shane_grade': 1},
            ], index=[14, 2, 99, 6])
            val = df[df.split == 'validation'].copy().reset_index(drop=True)
            val['image_filename'], val['image_grading'] = val.image_path, val.shane_grade
            dataset = ImageQualityDataset(val, image_transform(), label_mapping={i: i - 1 for i in range(1, 6)})
            with contextlib.redirect_stdout(io.StringIO()):
                patches, _ = subsample_subimages(dataset, method='all', df=val)
            model = PixelModel()
            with torch.inference_mode():
                expected = torch.softmax(model(torch.stack([item[0] for item in patches])), dim=1).reshape(2, 9, 5).mean(1).numpy()
            with patch('PIL.Image.open', wraps=Image.open) as opened, contextlib.redirect_stdout(io.StringIO()):
                metrics, predictions = evaluate_validation(model, df, 'shane', folder / 'output', batch_size=4)
            self.assertEqual([call.args[0] for call in opened.call_args_list], [str(first), str(second)])
            self.assertEqual(predictions.image_id.tolist(), ['variant_b', 'variant_a'])
            self.assertEqual(predictions.reference_grade.tolist(), [3, 1])
            actual = predictions[[f'probability_grade_{i}' for i in range(1, 6)]].to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)
            self.assertEqual(predictions.predicted_grade.tolist(), (expected.argmax(1) + 1).tolist())
            self.assertEqual(metrics['image_count'], 2)
            saved = pd.read_csv(folder / 'output' / 'validation_predictions.csv')
            pd.testing.assert_frame_equal(saved, predictions, check_exact=False)
            self.assertEqual(pd.read_csv(folder / 'output' / 'validation_confusion_matrix.csv', index_col=0).shape, (5, 5))
            with self.assertRaises(FileExistsError):
                evaluate_validation(model, df, 'shane', folder / 'output')
            bad = df.copy()
            bad.loc[99, 'split'] = 'validation'
            with self.assertRaisesRegex(ValueError, 'VISUS-NHP'):
                evaluate_validation(model, bad, 'shane', folder / 'bad')

    def test_metrics_and_undefined_policy(self):
        reference, predicted = [1, 1, 2, 3, 4, 5], [1, 2, 3, 3, 5, 1]
        metrics, grades, _ = calculate_validation_metrics(reference, predicted)
        self.assertAlmostEqual(metrics['exact_accuracy'], 2 / 6)
        self.assertAlmostEqual(metrics['within_one_grade_accuracy'], 5 / 6)
        self.assertAlmostEqual(metrics['mean_absolute_grade_error'], 7 / 6)
        self.assertAlmostEqual(metrics['quadratic_weighted_kappa'], cohen_kappa_score(reference, predicted, labels=range(1, 6), weights='quadratic'))
        for average in ('macro', 'weighted'):
            self.assertAlmostEqual(metrics[f'{average}_f1'], f1_score(reference, predicted, labels=range(1, 6), average=average, zero_division=0))
        self.assertFalse(grades.loc[grades.grade == 4, 'precision_defined'].iloc[0])
        metrics, grades, _ = calculate_validation_metrics([3, 3], [3, 3])
        self.assertIsNone(metrics['quadratic_weighted_kappa'])
        self.assertEqual(metrics['macro_f1'], 0.2)
        self.assertEqual(metrics['weighted_f1'], 1.0)
        self.assertFalse(grades.loc[grades.grade == 1, 'f1_defined'].iloc[0])


if __name__ == '__main__':
    unittest.main()
