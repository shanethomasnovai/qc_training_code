"""Selectable five-logit objectives; CDW-CE follows the supplied formulation."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')


def continuous_cdw_ce(logits, targets, alpha=1.0):
    validate_alpha(alpha)
    if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape[0] == 0:
        raise ValueError('Expected nonempty logits with shape [B, 5]')
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError('Expected continuous targets with shape [B]')
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise ValueError('Logits and continuous targets must be floating point')
    if logits.device != targets.device:
        raise ValueError('Continuous targets must be on the same device as logits')
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('Logits and continuous targets must be finite')
    if ((targets < 1) | (targets > 5)).any():
        raise ValueError('Continuous targets must be on the 1-5 grade scale')

    probs = F.softmax(logits, dim=1)
    classes = torch.arange(1, 6, device=logits.device, dtype=logits.dtype)
    distances = torch.abs(classes.unsqueeze(0) - targets.unsqueeze(1))
    weights = distances.pow(alpha)
    log_not_probs = torch.log((1 - probs).clamp(min=1e-7))
    loss = -(weights * log_not_probs).sum(dim=1)
    return loss.mean()


class ContinuousCDWCELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        validate_alpha(alpha)
        self.alpha = alpha

    def forward(self, logits, targets):
        return continuous_cdw_ce(logits, targets, self.alpha)


def make_criterion(loss='cross_entropy', class_weights=None, alpha=1.0):
    if loss == 'continuous_cdw_ce':
        if class_weights is not None:
            raise ValueError('continuous_cdw_ce uses distance weights only; class-balancing weights are incompatible')
        return ContinuousCDWCELoss(alpha)
    if loss != 'cross_entropy':
        raise ValueError(f'Unknown loss: {loss}')
    return nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
