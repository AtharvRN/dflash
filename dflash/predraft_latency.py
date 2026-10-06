"""Small policy helpers shared by native replay and CPU tests."""
from __future__ import annotations


def resolve_concurrencies(values=None, *, smoke=False):
    """Supported batch sizes; a full-cycle cohort may have one smaller tail batch."""
    values = list(values) if values is not None else ([4] if smoke else [64, 128])
    cohort_size = 4 if smoke else 128
    if (not values or len(set(values)) != len(values) or
            any(type(c) is not int or c <= 0 or c > cohort_size or cohort_size % c
                for c in values)):
        raise ValueError('Concurrencies must be unique positive divisors of the cohort size')
    if smoke and values != [4]:
        raise ValueError('Smoke concurrency must remain C4')
    return values


def validate_replay_cohort(rows, metadata):
    if metadata.get('cohort_mode') == 'all_assessment_cycles':
        if len(rows) != metadata['rows'] or not 4 <= len(rows) <= 4096:
            raise ValueError('Full assessment cohort count mismatch')
    elif len(rows) != 128:
        raise ValueError('Frozen legacy cohort must contain 128 assessment states')
    if any(r['group'] != 'assessment' for r in rows):
        raise ValueError('Replay must not contain training/calibration states')
    if len({(str(r['prompt_id']), r['cycle']) for r in rows}) != len(rows):
        raise ValueError('Duplicate replay cycle')


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
