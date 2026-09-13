"""Pre-draft acceptance heads over already-computed context token features."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContextReadBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor, valid: torch.Tensor):
        memory = self.context_norm(context)
        read, _ = self.attention(self.query_norm(query), memory, memory,
                                 key_padding_mask=~valid, need_weights=False)
        query = query + self.dropout(read)
        return query + self.dropout(self.ff(self.ff_norm(query)))


class ContextAcceptancePredictor(nn.Module):
    """One pooled query or one query per conditional acceptance position.

    Feature width is retained; only standard attention Q/K/V projections are used.
    Inputs must be pre-draft context, never current-cycle post-draft features.
    """
    def __init__(self, *, input_dim: int = 2560, num_slots: int = 15,
                 context_window: int = 16, num_queries: int = 15, num_layers: int = 2,
                 num_heads: int = 16, ff_width: int = 1024, dropout: float = 0.05):
        super().__init__()
        if num_queries not in (1, num_slots):
            raise ValueError("Use one pooled query or one query per slot")
        if min(input_dim, num_slots, context_window, num_layers, num_heads, ff_width) < 1:
            raise ValueError("Model dimensions must be positive")
        if input_dim % num_heads:
            raise ValueError("input_dim must be divisible by num_heads")
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.context_window = context_window
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, input_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.context_age = nn.Embedding(context_window, input_dim)
        nn.init.normal_(self.context_age.weight, std=0.02)
        self.blocks = nn.ModuleList([ContextReadBlock(input_dim, num_heads, ff_width, dropout)
                                     for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_slots if num_queries == 1 else 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                anchor_embedding: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[:2] != mask.shape or features.shape[-1] != self.input_dim:
            raise ValueError("Expected features [batch, context, width] and matching mask")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Context mask must be binary")
        valid = mask.bool()
        if features.shape[1] > self.context_window or not valid.any(dim=1).all():
            raise ValueError("Context exceeds configured window or contains an empty row")
        # Distance from the newest valid token, invariant to left/right padding.
        ages = valid.sum(1, keepdim=True) - valid.long().cumsum(1)
        ages = ages.clamp(0, self.context_window - 1)
        clean = features.float().masked_fill(~valid.unsqueeze(-1), 0)
        context = clean + self.context_age(ages)
        query = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        if anchor_embedding is not None:
            if anchor_embedding.shape != (features.shape[0], self.input_dim):
                raise ValueError("Anchor embedding must match batch and feature width")
            query = query + anchor_embedding.float().unsqueeze(1)
        for block in self.blocks:
            query = block(query, context, valid)
        logits = self.head(self.output_norm(query))
        return logits[:, 0] if self.num_queries == 1 else logits.squeeze(-1)


class ResidualContextAcceptancePredictor(nn.Module):
    """Frozen last-vector predictor plus a zero-initialized context correction."""
    def __init__(self, baseline: nn.Module, correction: ContextAcceptancePredictor, *, last_only: bool = False):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.correction = correction
        self.last_only = last_only
        nn.init.zeros_(self.correction.head.weight)
        nn.init.zeros_(self.correction.head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Context mask must be binary")
        with torch.no_grad():
            baseline = self.baseline(features, mask)
        correction_mask = mask
        if self.last_only:
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = positions.masked_fill(~mask.bool(), -1).max(1).values
            correction_mask = (positions == last[:, None]) & mask.bool()
        return baseline.float() + self.correction(features, correction_mask).float()


def acceptance_nll(logits: torch.Tensor, accepted: torch.Tensor,
                   reduction: str = "mean") -> torch.Tensor:
    """First-rejection likelihood; the maximum accepted length is right-censored."""
    if logits.ndim != 2 or accepted.shape != logits.shape[:1]:
        raise ValueError("Expected logits [batch, slots] and accepted [batch]")
    if not torch.isfinite(accepted).all() or ((accepted < 0) | (accepted > logits.shape[1])
                                              | (accepted != accepted.round())).any():
        raise ValueError("Accepted length must be an integer between zero and num_slots")
    k = torch.arange(1, logits.shape[1] + 1, device=logits.device)
    success = k <= accepted[:, None]
    reachable = k <= accepted[:, None] + 1
    terms = F.binary_cross_entropy_with_logits(logits.float(), success.float(), reduction="none")
    per_row = (terms * reachable).sum(1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    if reduction != "mean":
        raise ValueError("Unknown reduction")
    return per_row.mean()


def acceptance_survival(logits: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.logsigmoid(logits.float()).cumsum(-1))


def choose_budget(survival: torch.Tensor, alpha: float) -> torch.Tensor:
    """Smallest integer budget retaining alpha of the predicted expected length."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0,1]")
    if alpha == 1:
        return torch.full(survival.shape[:1], survival.shape[1], device=survival.device, dtype=torch.long)
    expected_prefix = survival.cumsum(-1)
    feasible = expected_prefix >= alpha * expected_prefix[:, -1:]
    return feasible.long().argmax(-1) + 1
