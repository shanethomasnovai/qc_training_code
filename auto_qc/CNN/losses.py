"""Previously used objectives; explicit reductions and inference-only PAVA."""
import torch
from torch.nn import functional as F

PROJECTION = 'inference_only_unweighted_least_squares_descending_pava_v1'


def loss_components(outputs, targets, settings, weights=None):
    name = settings['name']
    logits, binary = outputs if isinstance(outputs, tuple) else (outputs, None)
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('Non-finite logits or targets')
    if name == 'hard_ce':
        primary = F.cross_entropy(logits, targets, weight=weights)
    elif name == 'soft_ce':
        terms = -targets * F.log_softmax(logits, dim=1)
        primary = (terms if weights is None else terms * weights).sum(1).mean()
    elif name == 'continuous_cdw_ce':
        alpha = settings['alpha']
        if not isinstance(alpha, (int, float)) or not 0 < alpha < float('inf') or targets.ndim != 1 or ((targets < 1) | (targets > 5)).any():
            raise ValueError('CDW requires positive alpha and continuous grades in [1,5]')
        distance = (torch.arange(1, 6, device=logits.device)[None, :] - targets[:, None]).abs().pow(alpha)
        primary = -(distance * torch.log((1-logits.softmax(1)).clamp(min=1e-7))).sum(1).mean()
    elif name in ('ordinal_bce', 'joint_bce', 'binary_bce'):
        if logits.shape != targets.shape or logits.shape[1] != 4 or ((targets < 0) | (targets > 1)).any() or (targets[:, 1:] > targets[:, :-1]).any():
            raise ValueError('Four descending cumulative targets required')
        primary = logits.new_zeros(()) if name == 'binary_bce' else F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(1).mean()
    else:
        raise ValueError(f'Unknown loss: {name}')
    auxiliary = F.binary_cross_entropy_with_logits(binary, targets[:, 1]) if binary is not None else primary.new_zeros(())
    if name == 'joint_bce':
        total = primary + settings['binary_loss_weight'] * auxiliary
    elif name == 'binary_bce':
        primary, total = auxiliary, auxiliary
    else:
        total = primary
    return primary, auxiliary, total


def isotonic_projection(q):
    """Our inference adaptation, not specified by the OR-Soft paper. No sorting."""
    if q.ndim != 2 or q.shape[1] != 4 or not torch.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Four finite sigmoid probabilities required')
    rows = []
    for row in q.detach().cpu().double().tolist():
        blocks = []
        for value in row:
            blocks.append((value, 1))
            while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
                right, nr = blocks.pop()
                left, nl = blocks.pop()
                blocks.append(((left*nl + right*nr)/(nl+nr), nl+nr))
        rows.append([value for value, n in blocks for _ in range(n)])
    return torch.tensor(rows, dtype=q.dtype, device=q.device).reshape(q.shape)


def grade_probabilities(logits, head):
    if head == 'five_logits':
        p = logits.softmax(1)
    else:
        q = logits.sigmoid()
        if head != 'coral':
            q = isotonic_projection(q)
        if (q[:, 1:] > q[:, :-1]).any():
            raise ValueError('Cumulative probabilities inconsistent')
        p = torch.cat((1-q[:, :1], q[:, :-1]-q[:, 1:], q[:, -1:]), 1)
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-6, rtol=0):
        raise ValueError('Invalid reconstructed grade probabilities')
    return p


def ordering_diagnostics(q):
    projected = isotonic_projection(q)
    gaps = (q[:, 1:]-q[:, :-1]).clamp_min(0)
    changed = (q != projected).any(1)
    return {'patch_count': len(q), 'violating_patch_count': int((gaps > 0).any(1).sum()),
            'violating_patch_fraction': float((gaps > 0).any(1).float().mean()),
            'violating_pair_fraction': float((gaps > 0).float().mean()), 'maximum_positive_gap': float(gaps.max()),
            'mean_positive_gap_all_pairs': float(gaps.mean()),
            'projection_changed_patch_count': int(changed.sum()), 'projection_changed_patch_fraction': float(changed.float().mean()),
            'projection_mean_absolute_change': float((q-projected).abs().mean()),
            'projection_policy': PROJECTION, 'projection_origin': 'Our inference-only adaptation, not specified by the paper'}
