"""Small pre-draft response models with previous-cycle rejected trace memory.

The trace is strictly after the previous rejected anchor.  It is evidence for
choosing an actual new block size, never valid KV to commit to the target.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn


ARMS = ("control", "aligned", "shuffled")
COMMON_FIELDS = ("features", "anchor_embedding", "rejected_anchor_embedding", "previous_meta")
MEMORY_FIELDS = ("trace_draft", "trace_target", "trace_mask", "trace_offsets")


class RejectedTraceResponseModel(nn.Module):
    """One-query memory pooling followed by 15 independent bounded means.

All arms instantiate exactly this model, including unused control parameters.
An always-valid null memory prevents NaNs on first-cycle/empty-memory rows.
No dropout call or tensor shape depends on which arm is being evaluated.
"""

    def __init__(self, input_dim: int = 2560, width: int = 128, dropout: float = .05):
        super().__init__()
        self.model_config = {"input_dim": input_dim, "width": width, "dropout": dropout}
        self.feature_projection = nn.Linear(input_dim, width)
        self.token_projection = nn.Linear(input_dim, width)
        self.draft_projection = nn.Linear(input_dim, width)
        self.target_projection = nn.Linear(input_dim, width)
        self.offset_embedding = nn.Embedding(16, width, padding_idx=0)
        self.query = nn.Sequential(nn.Linear(3*width+3, width), nn.GELU(), nn.LayerNorm(width))
        self.memory = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.LayerNorm(width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, width))
        self.head = nn.Sequential(
            nn.Linear(2*width, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 15))
        self.register_buffer("budgets", torch.arange(1, 16, dtype=torch.float32))

    def forward(self, features, anchor_embedding, rejected_anchor_embedding,
                previous_meta, trace_draft, trace_target, trace_mask, trace_offsets):
        query = self.query(torch.cat((self.feature_projection(features.float()),
            self.token_projection(anchor_embedding.float()),
            self.token_projection(rejected_anchor_embedding.float()), previous_meta.float()), dim=-1))
        mask = trace_mask.bool()
        # Mask inputs before projection: masked values cannot influence output,
        # and pathological values in an unused memory are not propagated.
        draft = trace_draft.float().masked_fill(~mask[..., None], 0)
        target = trace_target.float().masked_fill(~mask[..., None], 0)
        offsets = trace_offsets.long().masked_fill(~mask, 0)
        memory = self.memory(torch.cat((self.draft_projection(draft),
            self.target_projection(target), self.offset_embedding(offsets)), dim=-1))
        memory = torch.cat((self.null_memory.expand(len(query), -1, -1), memory), dim=1)
        valid = torch.cat((torch.ones((len(query), 1), dtype=torch.bool, device=mask.device), mask), dim=1)
        score = (self.key(memory)*query[:, None]).sum(-1)/(query.shape[-1]**.5)
        attention = score.masked_fill(~valid, -torch.inf).softmax(-1)
        pooled = (attention[..., None]*self.value(memory)).sum(1)
        return self.head(torch.cat((query, pooled), dim=-1)).sigmoid()*self.budgets


def make_donor_mapping(rows, seed: int = 913):
    """Draw whole trace donors inside each partition and previous B/A stratum.

    Sampling is with replacement among valid rows in other prompts.  A literal
    permutation may not exist for imbalanced prompt strata; reporting a sampled
    donor map is more honest than silently allowing same-prompt donors.  Rows
    with no suffix or no different-prompt donor get -1.  They retain common
    features and labels and have empty memory in *every* arm.
    """
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        if row["has_previous"] and row["previous_B"]-row["previous_A"]-2 > 0:
            strata[(row["group"], int(row["previous_B"]), int(row["previous_A"]))].append(i)
    donors = np.full(len(rows), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _, indices in sorted(strata.items()):
        for i in indices:
            available = [j for j in indices if int(rows[j]["prompt_id"]) != int(rows[i]["prompt_id"])]
            if available:
                donors[i] = int(rng.choice(available))
    audit_donor_mapping(rows, donors)
    return donors


def audit_donor_mapping(rows, donors):
    donors = np.asarray(donors)
    if donors.shape != (len(rows),) or donors.dtype.kind not in "iu":
        raise ValueError("Invalid donor mapping dimensions or dtype")
    if ((donors < -1) | (donors >= len(rows))).any():
        raise ValueError("Donor index out of bounds")
    for i, j in enumerate(donors):
        if j < 0:
            continue
        row, donor = rows[i], rows[int(j)]
        if int(row["prompt_id"]) == int(donor["prompt_id"]):
            raise ValueError("Same-prompt memory donor")
        if any(row[k] != donor[k] for k in ("group", "previous_B", "previous_A")):
            raise ValueError("Donor partition or previous B/A mismatch")
        if not row["has_previous"] or not donor["has_previous"] or row["previous_B"]-row["previous_A"]-2 <= 0:
            raise ValueError("Donor requires a nonempty rejected suffix")


def model_batch(arrays, indices, arm, donors, device="cpu"):
    """Keep current inputs/targets fixed and select the paired memory source."""
    if arm not in ARMS:
        raise ValueError("Unknown trace ablation arm")
    indices = np.asarray(indices, dtype=np.int64)
    donor = np.asarray(donors)[indices]
    memory_indices = np.where(donor >= 0, donor, indices) if arm == "shuffled" else indices
    result = {k: torch.as_tensor(np.array(arrays[k][indices], copy=True), device=device) for k in COMMON_FIELDS}
    for key in MEMORY_FIELDS:
        result[key] = torch.as_tensor(np.array(arrays[key][memory_indices], copy=True), device=device)
    effective = donor >= 0
    if arm == "control":
        effective[:] = False
    result["trace_mask"] = result["trace_mask"].bool() & torch.as_tensor(effective, device=device)[:, None]
    return result
