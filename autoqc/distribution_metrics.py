"""Image-level agreement with observed grader distributions, without class weights."""
import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = [f'probability_grade_{grade}' for grade in range(1, 6)]
TARGET_COLUMNS = [f'target_grade_{grade}' for grade in range(1, 6)]


def calculate_distribution_metrics(targets, probabilities):
    targets, probabilities = np.asarray(targets, dtype=float), np.asarray(probabilities, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 5 or targets.shape[0] == 0 or targets.shape != probabilities.shape:
        raise ValueError('Expected matching, nonempty [images, 5] distributions')
    for name, values in [('targets', targets), ('probabilities', probabilities)]:
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f'{name} must contain finite probabilities between zero and one')
        if not np.allclose(values.sum(1), 1, rtol=0, atol=1e-5):
            raise ValueError(f'{name} rows must sum to one')
    # Remove floating-point summation drift, not substantive probability errors.
    targets = targets / targets.sum(1, keepdims=True)
    probabilities = probabilities / probabilities.sum(1, keepdims=True)
    floor = float(np.finfo(np.float32).tiny)
    cross_entropy = -(targets * np.log(np.maximum(probabilities, floor))).sum(1)
    emd = np.abs(np.cumsum(targets, axis=1)[:, :-1] - np.cumsum(probabilities, axis=1)[:, :-1]).sum(1)
    per_image = pd.DataFrame({'earth_movers_distance': emd, 'soft_cross_entropy': cross_entropy})
    metrics = {
        'image_count': len(targets),
        'earth_movers_distance': float(emd.mean()),
        'soft_cross_entropy': float(cross_entropy.mean()),
        'reference': 'fractions of available grader ratings per image',
        'class_weights_applied': False,
        'aggregation': 'equal-weight mean over images; no rating-count weighting',
        'earth_movers_distance_definition': 'Wasserstein-1: sum of absolute cumulative-probability differences at boundaries 1-4',
        'earth_movers_distance_units': 'grade steps, range 0 to 4; lower is better',
        'soft_cross_entropy_definition': '-sum(target * ln(predicted_probability)); lower is better; minimum depends on reference uncertainty',
        'probability_floor': floor,
        'rounding_policy': 'rows within 1e-5 of unit sum are renormalised for floating-point drift',
        'grade_order': [1, 2, 3, 4, 5],
    }
    return metrics, per_image
