"""Review binary disagreement and false negatives from saved validation predictions.

No inference, threshold search or training. Diagnostic probability bands are fixed
at 0.4 and 0.6; they do not alter the 0.5 decision rule. Reference vote proportions
describe observed disagreement, not an adjudicated probability of acceptability.
"""
import argparse
import base64
import contextlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .score_distributions import score_runs


def analyse(run_dirs, reference_run, focus_run, output_dir):
    output_dir, reference_run, focus_run = Path(output_dir), Path(reference_run), Path(focus_run)
    run_dirs = [Path(run) for run in run_dirs]
    if focus_run.resolve() not in {run.resolve() for run in run_dirs}:
        raise ValueError('Focus run must be included in the comparison')
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory to preserve prior reports')
    # Reuse all existing validation-only, split, snapshot and image-ID guards.
    with contextlib.redirect_stdout(io.StringIO()):
        comparison = score_runs(run_dirs, reference_run, output_dir / 'comparison', 0.5)
    config = json.loads((reference_run / 'run_config.json').read_text())
    columns = config['grader_columns']
    targets = pd.read_csv(output_dir / 'comparison' / 'reference_targets.csv')
    validation = targets.loc[targets.split.eq('validation')].set_index('image_id')
    summaries, frames = [], []
    for run in run_dirs:
        frame = pd.read_csv(output_dir / 'comparison' / run.name / 'distribution_per_image.csv')
        frame['run_name'] = run.name
        tied = frame.binary_reference_tied
        probability = frame.p_unacceptable.to_numpy()
        q = frame.unacceptable_vote_proportion.to_numpy()
        clipped = np.clip(probability, 1e-12, 1 - 1e-12)
        frame['binary_vote_brier_error'] = (probability - q) ** 2
        frame['binary_vote_cross_entropy'] = -(q * np.log(clipped) + (1 - q) * np.log1p(-clipped))
        frame['distance_from_half'] = np.abs(probability - .5)
        frame['diagnostic_probability_band'] = np.where(probability < .4, 'below_0.4',
                                                        np.where(probability > .6, 'above_0.6', '0.4_to_0.6'))
        row = {'run_name': run.name, 'image_count': len(frame)}
        for label, subset in [('tied', frame.loc[tied]), ('unique_majority', frame.loc[~tied])]:
            row.update({f'{label}_count': len(subset),
                        f'{label}_mean_p_unacceptable': float(subset.p_unacceptable.mean()),
                        f'{label}_mean_binary_vote_brier_error': float(subset.binary_vote_brier_error.mean()),
                        f'{label}_mean_binary_vote_cross_entropy': float(subset.binary_vote_cross_entropy.mean())})
        tie_frame = frame.loc[tied]
        row.update({
            'tied_predicted_acceptable_count': int(tie_frame.binary_prediction.eq('acceptable').sum()),
            'tied_predicted_unacceptable_count': int(tie_frame.binary_prediction.eq('unacceptable').sum()),
            'tied_p_exactly_half_count': int(tie_frame.p_unacceptable.eq(.5).sum()),
            'tied_p_below_0p4_count': int(tie_frame.p_unacceptable.lt(.4).sum()),
            'tied_p_0p4_to_0p6_count': int(tie_frame.p_unacceptable.between(.4, .6).sum()),
            'tied_p_above_0p6_count': int(tie_frame.p_unacceptable.gt(.6).sum()),
            'tied_rating_count_distribution': json.dumps(tie_frame.rating_count.value_counts().sort_index().to_dict()),
            'false_negative_count': int(frame.binary_confusion_outcome.eq('FN').sum()),
        })
        assert row['tied_predicted_acceptable_count'] + row['tied_predicted_unacceptable_count'] == row['tied_count']
        assert row['tied_p_below_0p4_count'] + row['tied_p_0p4_to_0p6_count'] + row['tied_p_above_0p6_count'] == row['tied_count']
        assert frame.loc[frame.binary_confusion_outcome.eq('FN'), 'binary_reference_tied'].eq(False).all()
        summaries.append(row)
        frames.append(frame)
    summary = pd.DataFrame(summaries)
    summary = comparison.merge(summary, on=['run_name', 'image_count'], validate='one_to_one')
    summary.to_csv(output_dir / 'run_review_summary.csv', index=False)
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(output_dir / 'all_validation_predictions.csv', index=False)
    all_predictions.loc[all_predictions.binary_reference_tied].to_csv(output_dir / 'tied_predictions.csv', index=False)
    false_negatives = all_predictions.loc[all_predictions.binary_confusion_outcome.eq('FN')]
    false_negatives.to_csv(output_dir / 'false_negatives.csv', index=False)
    grouped = all_predictions.groupby('image_id', sort=False)
    cases = validation.loc[frames[0].image_id].copy()
    cases['runs_with_false_negative'] = grouped.binary_confusion_outcome.apply(lambda x: int(x.eq('FN').sum()))
    cases['runs_predicting_unacceptable'] = grouped.binary_prediction.apply(lambda x: int(x.eq('unacceptable').sum()))
    cases['mean_p_unacceptable_across_runs'] = grouped.p_unacceptable.mean()
    cases['min_p_unacceptable_across_runs'] = grouped.p_unacceptable.min()
    cases['max_p_unacceptable_across_runs'] = grouped.p_unacceptable.max()
    first = frames[0].set_index('image_id')
    for column in ['binary_reference_tied', 'binary_reference', 'acceptable_vote_count', 'unacceptable_vote_count']:
        cases[column] = first[column]
    review = cases.loc[cases.binary_reference_tied | cases.runs_with_false_negative.gt(0)].copy()
    review['review_reason'] = np.where(review.binary_reference_tied, 'binary vote tie', 'false negative in at least one run')
    review = review.sort_values(['runs_with_false_negative', 'mean_p_unacceptable_across_runs'], ascending=[False, False])
    review.to_csv(output_dir / 'image_review_queue.csv')
    # Independent reviewers should not see model probabilities or previous ratings.
    template = review.reset_index()[['image_id', 'image_path']].copy()
    template['new_independent_grade_1_to_5'] = ''
    template['reviewer_notes'] = ''
    template.to_csv(output_dir / 'independent_rating_template.csv', index=False)
    focus = all_predictions.loc[all_predictions.run_name.eq(focus_run.name)].copy()
    tied_focus = focus.loc[focus.binary_reference_tied]
    selected = []
    selections = [
        ('Focus-run false negative', focus.loc[focus.binary_confusion_outcome.eq('FN')].sort_values('p_unacceptable').head(4)),
        ('Tied votes: highest unacceptable probability', tied_focus.sort_values('p_unacceptable', ascending=False).head(4)),
        ('Tied votes: closest to 0.5', tied_focus.sort_values('distance_from_half').head(4)),
    ]
    for reason, subset in selections:
        for _, row in subset.iterrows():
            if row.image_id not in [record['image_id'] for record in selected]:
                selected.append({'image_id': row.image_id, 'selection_reason': reason,
                                 'p_unacceptable': row.p_unacceptable,
                                 'binary_reference_tied': bool(row.binary_reference_tied)})
    pd.DataFrame(selected).to_csv(output_dir / 'visual_review_selection.csv', index=False)
    cards = []
    for record in selected:
        reference = validation.loc[record['image_id']]
        image_path = Path(reference.image_path)
        assert reference.split == 'validation'
        if image_path.is_file():
            encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
            extension = image_path.suffix.lower().lstrip('.')
            mime = 'jpeg' if extension in ('jpg', 'jpeg') else extension
            image = f'<img src="data:image/{mime};base64,{encoded}" alt="Validation image">'
        else:
            image = '<p>Original image unavailable</p>'
        ratings = ', '.join(f'{column}: {reference[column]:g}' for column in columns if pd.notna(reference[column]))
        cards.append(f'<article><h3>{html.escape(record["selection_reason"])}</h3>{image}'
                     f'<p>{html.escape(record["image_id"])}</p><p>{html.escape(ratings)}</p>'
                     f'<p>p(unacceptable) = {record["p_unacceptable"]:.6f}</p></article>')
    chart = ['<svg viewBox="0 0 700 210" role="img" aria-label="Tied-case probability histogram">']
    histogram, _ = np.histogram(tied_focus.p_unacceptable, bins=np.linspace(0, 1, 11))
    maximum = max(int(histogram.max()), 1)
    for i, count in enumerate(histogram):
        height = int(count) / maximum * 140
        chart.append(f'<rect x="{i * 68 + 10}" y="{160-height}" width="58" height="{height}" fill="#3976a8"/>'
                     f'<text x="{i*68+39}" y="{152-height}" text-anchor="middle">{count}</text>'
                     f'<text x="{i*68+39}" y="185" text-anchor="middle">{i/10:.1f}–{(i+1)/10:.1f}</text>')
    chart.append('</svg>')
    report_html = ('<!doctype html><meta charset="utf-8"><title>Validation disagreement review</title>'
                   '<style>body{font:16px Arial;margin:30px;background:#fafafa}article{border:1px solid #ccc;'
                   'padding:16px;background:white;overflow-wrap:anywhere}img{width:100%;max-height:500px;object-fit:contain}'
                   '.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}svg{max-width:900px;width:100%}</style>'
                   f'<h1>Validation disagreement review</h1><p>Focus: {html.escape(focus_run.name)}</p>'
                   '<p>Threshold fixed at 0.5. Binary vote ties are assumed acceptable, not adjudicated ground truth. '
                   'Original validation images only; no training or test images opened. '
                   'Gallery is selected for inspection and is not representative of all validation images.</p>'
                   '<h2>Tied-case p(unacceptable)</h2>' + ''.join(chart) + '<div class="cards">' + ''.join(cards) + '</div>')
    (output_dir / 'visual_review.html').write_text(report_html, encoding='utf-8')
    metadata = {'reference_run': str(reference_run.resolve()), 'focus_run': focus_run.name,
                'threshold': .5, 'diagnostic_probability_band': [0.4, 0.6],
                'binary_vote_cross_entropy_log_clip': 1e-12,
                'run_count': len(run_dirs), 'validation_image_count': len(cases),
                'binary_tied_image_count': int(cases.binary_reference_tied.sum()),
                'unique_false_negative_images_across_runs': int(cases.runs_with_false_negative.gt(0).sum()),
                'review_queue_count': len(review),
                'limitation': 'Exploratory validation analysis; no threshold optimisation, independent adjudication or held-out test evaluation. Runs are correlated, not independent votes.'}
    (output_dir / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(summary[['run_name', 'binary_FN', 'binary_sensitivity', 'binary_specificity',
                   'tied_p_0p4_to_0p6_count', 'tied_p_above_0p6_count', 'tied_mean_p_unacceptable']].to_string(index=False))
    print(f'Reports: {output_dir}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, nargs='+', type=Path)
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--focus-run', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    analyse(args.run_dir, args.reference_run, args.focus_run, args.output_dir)
