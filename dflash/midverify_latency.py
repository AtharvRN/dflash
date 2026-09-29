"""Pure helpers for the bounded, same-state segmented-verification benchmark."""
from __future__ import annotations

import hashlib
import numpy as np


def select_states(rows, source_rows, count):
    """First eligible state of each assessment prompt; no label-based selection."""
    source = {(str(r['prompt_id']), int(r['cycle'])): r for r in source_rows}
    selected, seen = [], set()
    for row in rows:
        pid = str(row['prompt_id'])
        if row['group'] != 'assessment' or pid in seen:
            continue
        original = source[(pid, int(row['cycle']))]
        for key in ('prefix_sha256', 'prefix_length', 'draft_ids', 'group'):
            if original[key] != row[key]:
                raise ValueError('Source state changed: ' + key)
        tokens = original['prefix_token_ids']
        if len(tokens) != row['prefix_length'] + 1 or len(row['draft_ids']) != 15:
            raise ValueError('Invalid anchor/prefix alignment')
        if hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest() != row['prefix_sha256']:
            raise ValueError('Prefix checksum mismatch')
        selected.append({**row, 'prefix_token_ids': tokens})
        seen.add(pid)
        if len(selected) == count:
            return selected
    raise ValueError('Insufficient distinct assessment prompts')


def packed_indices(lengths, width=16):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not len(lengths) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError('Require a nonempty integer length vector')
    if np.any(lengths < 1) or np.any(lengths > width):
        raise ValueError('Invalid prefix length')
    return np.flatnonzero((np.arange(width)[None] < lengths[:, None]).reshape(-1))


def compact_indices(front, end):
    front, end = np.asarray(front), np.asarray(end)
    if front.shape != end.shape or np.any(end > front):
        raise ValueError('Cannot resurrect a removed suffix')
    a, b = packed_indices(front), packed_indices(end)
    out = np.searchsorted(a, b)
    if not np.array_equal(a[out], b):
        raise AssertionError('Compaction lost prefix membership')
    return out


def accepted_from_top1(blocks, top1, lengths):
    blocks, top1 = np.asarray(blocks), np.asarray(top1)
    lengths = np.asarray(lengths)
    if blocks.shape != top1.shape or blocks.shape != (len(lengths), 16):
        raise ValueError('Invalid padded block layout')
    packed_indices(lengths)
    matches = (blocks[:, 1:] == top1[:, :-1]) & (np.arange(15)[None] < lengths[:, None] - 1)
    accepted = np.cumprod(matches, axis=1).sum(1)
    return accepted, top1[np.arange(len(lengths)), accepted]
