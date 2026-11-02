from .dataset import generate_targets

"""Evaluation only: grades 1-2 acceptable, grades 3-5 unacceptable (positive).

Binary references use available individual votes. Ties are assumed acceptable;
this evaluation convention is not adjudicated ground truth. Every image counts once.
"""
import numpy as np
import pandas as pd


def validate_threshold(threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Binary threshold must be finite and between 0 and 1')
    return float(threshold)


def _scores(reference, predicted):
    tp = int((reference & predicted).sum())
    tn = int((~reference & ~predicted).sum())
    fp = int((~reference & predicted).sum())
    fn = int((reference & ~predicted).sum())
    result = {'image_count': len(reference), 'reference_acceptable_count': tn + fp,
              'reference_unacceptable_count': tp + fn, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    reasons = {}
    def ratio(name, numerator, denominator, reason):
        result[name] = float(numerator / denominator) if denominator else None
        reasons[name] = None if denominator else reason
    ratio('exact_accuracy', tp + tn, len(reference), 'No images')
    ratio('sensitivity', tp, tp + fn, 'No reference unacceptable images')
    ratio('specificity', tn, tn + fp, 'No reference acceptable images')
    ratio('precision', tp, tp + fp, 'No predicted unacceptable images')
    ratio('f1', 2 * tp, 2 * tp + fp + fn, 'No reference or predicted unacceptable images')
    defined = result['sensitivity'] is not None and result['specificity'] is not None
    result['balanced_accuracy'] = (result['sensitivity'] + result['specificity']) / 2 if defined else None
    reasons['balanced_accuracy'] = None if defined else 'Both reference classes are required'
    result['undefined_reasons'] = reasons
    return result


def calculate_binary_metrics(votes, probabilities, threshold=0.5):
    """Consume validated integer five-grade votes and whole-image probabilities."""
    threshold = validate_threshold(threshold)
    votes, probabilities = np.asarray(votes), np.asarray(probabilities, dtype=float)
    if (votes.ndim != 2 or votes.shape[1] != 5 or not len(votes)
            or probabilities.shape != votes.shape or not np.isfinite(votes).all()
            or (votes < 0).any() or (votes != np.floor(votes)).any()
            or (votes.sum(1) == 0).any()):
        raise ValueError('Expected nonempty [images, 5] integer votes and matching probabilities')
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Expected finite probabilities summing to one')
    acceptable_votes = votes[:, :2].sum(1)
    unacceptable_votes = votes[:, 2:].sum(1)
    count = votes.sum(1)
    tied = acceptable_votes == unacceptable_votes
    reference = unacceptable_votes > acceptable_votes
    p_acceptable = probabilities[:, :2].sum(1)
    p_unacceptable = probabilities[:, 2:].sum(1)
    predicted = p_unacceptable >= threshold
    metrics = _scores(reference, predicted)
    metrics.update({
        'decision_threshold': threshold, 'positive_class': 'unacceptable',
        'acceptable_grades': [1, 2], 'unacceptable_grades': [3, 4, 5],
        'reference_tie_count': int(tied.sum()), 'reference_tie_percentage': float(tied.mean() * 100),
        'reference_convention': 'Majority of available individual binary grader votes; ties are assumed acceptable. This is an evaluation convention, not adjudicated ground truth.',
        'prediction_rule': 'sum of whole-image probabilities for grades 3-5 >= decision_threshold',
        'aggregation': 'equal weight per image; missing ratings ignored',
        'unique_majority': _scores(reference[~tied], predicted[~tied]),
    })
    details = pd.DataFrame({
        'p_acceptable': p_acceptable, 'p_unacceptable': p_unacceptable,
        'rating_count': count, 'acceptable_vote_count': acceptable_votes,
        'unacceptable_vote_count': unacceptable_votes,
        'acceptable_vote_proportion': acceptable_votes / count,
        'unacceptable_vote_proportion': unacceptable_votes / count,
        'binary_reference': np.where(reference, 'unacceptable', 'acceptable'),
        'binary_prediction': np.where(predicted, 'unacceptable', 'acceptable'),
        'binary_reference_tied': tied,
        'binary_confusion_outcome': np.where(reference, np.where(predicted, 'TP', 'FN'),
                                             np.where(predicted, 'FP', 'TN')),
    })
    return metrics, details


def calculate_binary_methods(votes, probabilities, threshold=0.5):
    """Score two decision rules against identical votes; preserve legacy sum fields."""
    summed, details = calculate_binary_metrics(votes, probabilities, threshold)
    summed['method'] = 'summed_probability'
    predicted_grade = np.asarray(probabilities).argmax(axis=1) + 1
    predicted = predicted_grade >= 3
    reference = details.binary_reference.eq('unacceptable').to_numpy()
    tied = details.binary_reference_tied.to_numpy()
    grade = _scores(reference, predicted)
    grade.update({key: summed[key] for key in (
        'positive_class', 'acceptable_grades', 'unacceptable_grades',
        'reference_tie_count', 'reference_tie_percentage', 'reference_convention', 'aggregation')})
    grade.update({'method': 'grade_argmax', 'decision_threshold': None,
                  'decision_threshold_reason': 'Not applicable: argmax over five grade probabilities, then map grade to binary class',
                  'prediction_rule': 'argmax of five whole-image grade probabilities + 1; grades 3-5 predict unacceptable',
                  'grade_probability_tie_rule': 'Exact highest-probability ties select the lowest numeric grade; no rounding',
                  'unique_majority': _scores(reference[~tied], predicted[~tied])})
    details['binary_summed_probability_prediction'] = details.binary_prediction
    details['binary_summed_probability_confusion_outcome'] = details.binary_confusion_outcome
    details['binary_grade_argmax_grade'] = predicted_grade
    details['binary_grade_argmax_prediction'] = np.where(predicted, 'unacceptable', 'acceptable')
    details['binary_grade_argmax_confusion_outcome'] = np.where(reference, np.where(predicted, 'TP', 'FN'),
                                                              np.where(predicted, 'FP', 'TN'))
    disagree = details.binary_summed_probability_prediction.ne(details.binary_grade_argmax_prediction)
    details['binary_methods_disagree'] = disagree
    agreement = {'disagreement_count': int(disagree.sum()), 'disagreement_percentage': float(disagree.mean() * 100)}
    return summed, grade, agreement, details






def calculate_binary_head_metrics(votes, probabilities, threshold=.5):
    """Separate binary probabilities; hard references keep the existing vote policy."""
    threshold = validate_threshold(threshold)
    votes, probabilities = np.asarray(votes), np.asarray(probabilities, dtype=float)
    if (votes.ndim != 2 or votes.shape[1] != 5 or len(votes) == 0 or probabilities.shape != (len(votes),)
            or not np.isfinite(votes).all() or (votes < 0).any() or (votes != np.floor(votes)).any()
            or (votes.sum(1) == 0).any() or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Expected integer five-grade votes and one binary probability per image')
    acceptable, unacceptable = votes[:, :2].sum(1), votes[:, 2:].sum(1)
    reference, tied = unacceptable > acceptable, unacceptable == acceptable
    predicted = probabilities >= threshold
    scores = _scores(reference, predicted)
    scores.update({'method': 'dedicated_binary_head', 'decision_threshold': threshold,
                   'positive_class': 'unacceptable', 'acceptable_grades': [1, 2], 'unacceptable_grades': [3, 4, 5],
                   'reference_tie_count': int(tied.sum()), 'reference_tie_percentage': float(tied.mean()*100),
                   'reference_convention': 'Majority of available individual binary votes; tied references assumed acceptable',
                   'prediction_rule': 'mean of patch sigmoid binary probabilities >= threshold; exact ties unacceptable',
                   'aggregation': 'equal weight per image; missing ratings ignored',
                   'unique_majority': _scores(reference[~tied], predicted[~tied])})
    details = pd.DataFrame({'binary_head_p_unacceptable': probabilities,
                           'binary_head_prediction': np.where(predicted, 'unacceptable', 'acceptable'),
                           'binary_head_confusion_outcome': np.where(reference, np.where(predicted, 'TP', 'FN'),
                                                                    np.where(predicted, 'FP', 'TN'))})
    return scores, details







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

def calculate_validation_metrics(reference, predicted):
    """Zero undefined P/R/F1 explicitly; undefined kappa becomes JSON null."""
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    if reference.size == 0 or reference.shape != predicted.shape:
        raise ValueError('Expected matching, nonempty grade arrays')
    if not np.isin(reference, range(1, 6)).all() or not np.isin(predicted, range(1, 6)).all():
        raise ValueError('Grades must be 1 through 5')
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (reference.astype(int) - 1, predicted.astype(int) - 1), 1)
    support, predicted_count, tp = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    precision = np.divide(tp, predicted_count, out=np.zeros(5), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    denominator = support + predicted_count
    f1 = np.divide(2 * tp, denominator, out=np.zeros(5), where=denominator != 0)
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
    expected = np.outer(support, predicted_count) / reference.size
    expected_disagreement = float((weights * expected).sum())
    kappa = 1 - float((weights * cm).sum()) / expected_disagreement if expected_disagreement > 0 else None
    error = np.abs(reference - predicted)
    metrics = {
        'image_count': int(reference.size),
        'exact_accuracy': float((error == 0).mean()),
        'within_one_grade_accuracy': float((error <= 1).mean()),
        'mean_absolute_grade_error': float(error.mean()),
        'quadratic_weighted_kappa': kappa,
        'macro_f1': float(f1.mean()),
        'weighted_f1': float(np.dot(f1, support) / support.sum()),
        'undefined_metric_policy': 'Undefined precision/recall/F1 are 0 with per-grade defined flags; macro F1 includes all five grades. Undefined kappa is null.',
        'quadratic_weighted_kappa_undefined_reason': 'Zero expected disagreement' if kappa is None else None,
    }
    per_grade = pd.DataFrame({
        'grade': range(1, 6), 'precision': precision, 'recall': recall,
        'f1': f1, 'support': support,
        'precision_defined': predicted_count != 0, 'recall_defined': support != 0,
        'f1_defined': denominator != 0,
    })
    return metrics, per_grade, pd.DataFrame(cm, index=range(1, 6), columns=range(1, 6))


def calculate_combined_metrics(reference, probabilities, grader_columns, binary_threshold=0.5, binary_head_probabilities=None):
    rated = generate_targets(reference.reset_index(drop=True), 'soft_categorical', grader_columns)
    if not rated.eligible.all():
        raise ValueError('Combined metrics require at least one selected rating per image')
    targets = reference[TARGET_COLUMNS].to_numpy() if set(TARGET_COLUMNS).issubset(reference.columns) else rated[TARGET_COLUMNS].to_numpy()
    metrics, details = calculate_distribution_metrics(targets, probabilities)
    if not np.allclose(targets, rated[TARGET_COLUMNS].to_numpy(), rtol=0, atol=1e-8):
        raise ValueError('Soft targets disagree with selected original ratings')
    ratings = rated[grader_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    votes = np.stack([(ratings == grade).sum(axis=1) for grade in range(1, 6)], axis=1)
    winners = votes == votes.max(axis=1, keepdims=True)
    reference_grade = votes.argmax(axis=1) + 1
    predicted = np.asarray(probabilities).argmax(axis=1) + 1
    tied = winners.sum(axis=1) > 1
    matches_any = votes[np.arange(len(votes)), predicted - 1] > 0
    errors = np.abs(reference_grade - predicted)
    consensus, _, _ = calculate_validation_metrics(reference_grade, predicted)
    metrics.update({
        'matches_any_grader_accuracy': float(matches_any.mean()),
        'consensus_reference_exact_accuracy': consensus['exact_accuracy'],
        'consensus_reference_mae': consensus['mean_absolute_grade_error'],
        'consensus_reference_quadratic_weighted_kappa': consensus['quadratic_weighted_kappa'],
        'consensus_reference_quadratic_weighted_kappa_undefined_reason': consensus['quadratic_weighted_kappa_undefined_reason'],
        'reference_tie_count': int(tied.sum()),
        'reference_tie_percentage': float(tied.mean() * 100),
        'single_grade_reference_convention': 'Reference grades are the most frequent available grader rating; ties are resolved in favour of the lower grade.',
    })
    details['predicted_grade'] = predicted
    details['consensus_reference_grade'] = reference_grade
    details['reference_tied'] = tied
    details['rating_count'] = rated.rating_count.to_numpy()
    details['matches_any_grader'] = matches_any
    details['consensus_exact_match'] = errors == 0
    details['absolute_grade_error'] = errors
    binary, grade_argmax, agreement, binary_details = calculate_binary_methods(votes, probabilities, binary_threshold)
    metrics['binary'] = binary
    metrics['binary_grade_argmax'] = grade_argmax
    metrics['binary_method_comparison'] = agreement
    for column in binary_details:
        details[column] = binary_details[column].to_numpy()
    if binary_head_probabilities is not None:
        dedicated, dedicated_details = calculate_binary_head_metrics(votes, binary_head_probabilities, binary_threshold)
        metrics['binary_dedicated_head'] = dedicated
        for column in dedicated_details:
            details[column] = dedicated_details[column].to_numpy()
    return metrics, details



