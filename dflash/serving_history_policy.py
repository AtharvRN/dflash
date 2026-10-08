"""GPU-resident, frozen history-priced controller for greedy DFlash serving.

State is indexed by request-pool slot and reset on a new prompt's first prefill.
Only completed verification rows 0..A contribute entropy (including the
correction/bonus distribution). No rejected-prefix distribution enters history.
"""
import hashlib
import json
import math
import os
import logging
from pathlib import Path

import torch


def load_artifacts(table_path, profile_path, frozen_path):
    # Copied alongside this module by the auditable serving extension.
    from .dflash_history_table import HistoryValueTable
    table_bytes, profile_bytes = Path(table_path).read_bytes(), Path(profile_path).read_bytes()
    frozen = json.loads(Path(frozen_path).read_text())
    bindings = dict(table_sha256=hashlib.sha256(table_bytes).hexdigest(),
                    cost_profile_sha256=hashlib.sha256(profile_bytes).hexdigest())
    if frozen.get('schema_version') != 1 or frozen['bindings'] != bindings:
        raise ValueError('Frozen history policy artifact binding mismatch')
    table = HistoryValueTable(json.loads(table_bytes))
    profile = json.loads(profile_bytes)
    if table.payload['provenance']['model_identity'] != profile['model_identity']:
        raise ValueError('History policy model identity mismatch')
    c = frozen['concurrency']
    costs = {int(b): t for b, t in profile['costs_ms'][str(c)].items()}
    policy = frozen['policies']['history']
    if (policy['history_free'] or set(costs) != set(table.blocks)
            or table.blocks != (4, 8, 12, 16)
            or not math.isfinite(policy['rho']) or policy['rho'] < 0
            or any(not math.isfinite(t) or t <= 0 for t in costs.values())):
        raise ValueError('Expected frozen history-aware policy with all arms')
    return table, costs, policy['rho'], c


def action_lookup(table, costs, rho):
    """Compile the exact CPU score, including Python-float tie breaking, once."""
    from .dflash_history_table import select_block
    bins, width = len(table.edges) + 1, max(table.blocks) + 1
    result = torch.empty((3, bins, width, 2), dtype=torch.int64)
    cold = dict(entropy_count=0, certainty=None, previous_block=None, previous_full=None)
    result.fill_(select_block(table, cold, mode='priced', costs_ms=costs, rho=rho)[0])
    for n in (1, 2):
        for bi in range(bins):
            # Avoid reconstructing a representative floating-point certainty.
            for b in table.blocks:
                for full in (0, 1):
                    key = f'{n}:{bi}:{b}:{full}'
                    values = table.payload['values'].get(key, table.payload['global_progress'])
                    scores = [v - rho*costs[arm] for arm, v in zip(table.blocks, values)]
                    result[n, bi, b, full] = table.blocks[max(range(len(scores)), key=scores.__getitem__)]
    return result


def row_entropy(logits):
    """Stable FP32 entropy without materializing FP32 tokens x vocab matrices."""
    if logits.device.type != 'cuda':
        lp = logits.float().log_softmax(-1)
        return -(lp.exp()*lp).sum(-1)
    import triton
    from .dflash_history_entropy import entropy_partials
    rows, vocab = logits.shape
    chunks = triton.cdiv(vocab, 4096)
    partial = torch.empty((rows, chunks, 3), device=logits.device, dtype=torch.float32)
    entropy_partials[(rows, chunks)](logits, partial, logits.stride(0), vocab, chunks, K=4096)
    maxima, sums, weighted = partial.unbind(-1)
    peak = maxima.max(-1, keepdim=True).values
    scale = (maxima - peak).exp()
    z = (scale*sums).sum(-1)
    ex = (scale*weighted).sum(-1) / z
    return (peak.squeeze(-1) + z.log() - ex).clamp_min(0)


class DFlashHistoryPolicy:
    def __init__(self, table, costs, rho, *, capacity, device):
        self.arms = table.blocks
        self.actions = action_lookup(table, costs, rho).to(device)
        self.edges = torch.tensor(table.edges, dtype=torch.float64, device=device)
        self.entropies = torch.zeros((capacity, 2), dtype=torch.float64, device=device)
        self.count = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.previous_block = torch.zeros_like(self.count)
        self.previous_full = torch.zeros_like(self.count)
        self.check_acceptance = os.getenv('SGLANG_DFLASH_HISTORY_CHECK') == '1'
        self.checked_rows = 0

    def reset(self, indices):
        self.count[indices] = 0
        self.entropies[indices] = 0
        self.previous_block[indices] = 0
        self.previous_full[indices] = 0

    def select(self, indices):
        n = self.count[indices]
        certainty = -self.entropies[indices].sum(-1) / n.clamp_min(1)
        bins = torch.bucketize(certainty.contiguous(), self.edges, right=True)
        return self.actions[n, bins, self.previous_block[indices], self.previous_full[indices]]

    def observe(self, indices, blocks, accepted, logits, offsets, real_indices=None,
                candidates=None, output_tokens=None, packed=False):
        if self.check_acceptance:
            self.check_emission(blocks, accepted, logits, offsets, real_indices,
                                candidates, output_tokens, packed)
        ent = row_entropy(logits)
        if real_indices is not None:
            ent = ent.index_select(0, real_indices)
        # Fixed-width 16 temporary entries per request, independent of vocab size.
        pos = torch.arange(max(self.arms), device=indices.device)[None, :]
        valid = pos <= accepted[:, None]
        gather = offsets[:-1, None] + pos
        gathered = ent[gather.clamp_max(ent.numel()-1)]
        mean = (gathered * valid).sum(-1) / (accepted + 1)
        old = self.entropies[indices, 1].clone()
        self.entropies[indices, 0] = old
        self.entropies[indices, 1] = mean.double()
        self.count[indices] = (self.count[indices] + 1).clamp_max(2)
        self.previous_block[indices] = blocks
        self.previous_full[indices] = (accepted == blocks - 1).long()

    def check_emission(self, blocks, accepted, logits, offsets, real_indices,
                       candidates, output_tokens, packed):
        """Diagnostic only: independent first-rejection and emitted-token oracle."""
        pred = logits.argmax(-1)
        if real_indices is not None:
            pred = pred[real_indices]
        candidates = candidates.reshape(-1)
        emitted = output_tokens.reshape(-1)
        cursor = 0
        for i, (b, a, off) in enumerate(zip(blocks.tolist(), accepted.tolist(), offsets[:-1].tolist())):
            draft, target = candidates[off:off+b], pred[off:off+b]
            expected_a = int((draft[1:] == target[:-1]).long().cumprod(0).sum().item())
            if a != expected_a:
                raise AssertionError(f'First-rejection mismatch: {a} != {expected_a}')
            expected = torch.cat([draft[1:a+1], target[a:a+1]])
            start = cursor if packed else off
            if not torch.equal(emitted[start:start+a+1], expected):
                raise AssertionError('Emitted tokens do not match verified prefix + correction/bonus')
            cursor += a+1
        self.checked_rows += len(blocks)
        logging.getLogger(__name__).info('HISTORY_ACCEPTANCE_CHECK passed cumulative_rows=%s mixed=%s',
                                        self.checked_rows, len(set(blocks.tolist())) > 1)
