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


def binary_confusion_matrix(metrics):
    return pd.DataFrame([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]],
                        index=pd.Index(['acceptable', 'unacceptable'], name='reference_class'),
                        columns=['predicted_acceptable', 'predicted_unacceptable'])


def binary_summary_fields(metrics):
    fields = {f'binary_{key}': metrics[key] for key in (
        'image_count', 'decision_threshold', 'exact_accuracy', 'sensitivity', 'specificity',
        'balanced_accuracy', 'precision', 'f1', 'reference_acceptable_count',
        'reference_unacceptable_count', 'reference_tie_count', 'reference_tie_percentage',
        'TP', 'TN', 'FP', 'FN')}
    fields.update({f'binary_unique_majority_{key}': metrics['unique_majority'][key]
                   for key in ('image_count', 'sensitivity', 'specificity', 'balanced_accuracy')})
    return fields
