"""Small policy helpers shared by native replay and CPU tests."""
from __future__ import annotations


def frozen_lengths(scores, threshold):
    import torch
    if scores.ndim != 2 or scores.shape[1] != 15:
        raise ValueError('Require 15 prefix scores per cycle')
    if threshold is None:
        return torch.full((len(scores),), 16, device=scores.device, dtype=torch.int32)
    # Preserve the calibration code's FP64 nextafter threshold exactly.
    return (scores.double() >= threshold).int().cumprod(1).sum(1).int()+1


def fixed_width(case):
    token = case.split('_')[0]
    if token not in ('fixed8', 'fixed12', 'fixed16'):
        raise ValueError('Unknown fixed replay case: ' + case)
    return int(token[5:])


def same_model_identity(left, right):
    """Snapshot paths may differ between PVC collection and workstation replay."""
    return (set(left) == set(right) == {'target', 'draft'} and
            all(left[k].get('repo') == right[k].get('repo') and
                left[k].get('revision') == right[k].get('revision') and
                bool(left[k].get('revision')) for k in ('target', 'draft')))
