from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class _DSparkConfidenceHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        proj_dim: int,
        markov_dim: int,
        scalar_dim: int,
        scalar_proj_dim: int,
        head_hidden_size: int,
        vocab_size: int,
        dropout: float,
        use_hidden: bool,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_dim = int(markov_dim)
        self.hidden_proj = (
            nn.Sequential(
                nn.Linear(hidden_size, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
                nn.Dropout(dropout),
            )
            if use_hidden
            else None
        )
        self.prev_token_embed = (
            nn.Embedding(self.vocab_size, self.markov_dim)
            if self.markov_dim > 0
            else None
        )
        self.scalar_proj = (
            nn.Sequential(
                nn.Linear(scalar_dim, scalar_proj_dim),
                nn.GELU(),
                nn.LayerNorm(scalar_proj_dim),
                nn.Dropout(dropout),
            )
            if scalar_dim > 0 and scalar_proj_dim > 0
            else None
        )
        hidden_out = proj_dim if self.hidden_proj is not None else 0
        markov_out = self.markov_dim if self.prev_token_embed is not None else 0
        scalar_out = scalar_proj_dim if self.scalar_proj is not None else 0
        input_dim = hidden_out + markov_out + scalar_out
        if input_dim <= 0:
            raise ValueError(
                "post-draft confidence head needs hidden, token, or scalar input"
            )
        self.head = nn.Sequential(
            nn.Linear(input_dim, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size // 2, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        if self.hidden_proj is not None:
            parts.append(self.hidden_proj(hidden.float()))
        if self.prev_token_embed is not None:
            prev_token_ids = prev_token_ids.clamp(min=0, max=self.vocab_size - 1)
            parts.append(self.prev_token_embed(prev_token_ids))
        if self.scalar_proj is not None:
            parts.append(self.scalar_proj(scalar.float()))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


@dataclass
class DFlashPostdraftPolicy:
    checkpoint_path: str
    alpha: float
    arms: tuple[int, ...]
    device: torch.device

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"DFLASH post-draft checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        args = payload.get("args") or {}
        self.num_slots = int(payload.get("num_slots", 15))
        self.hidden_size = int(payload.get("hidden_size", 0))
        self.scalar_dim = int(payload.get("scalar_dim", 0))
        use_hidden = not bool(args.get("disable_hidden", False))
        self.markov_dim = int(args.get("markov_dim", 0))
        self.vocab_size = int(args.get("vocab_size", 200000))
        self.model = _DSparkConfidenceHead(
            hidden_size=self.hidden_size,
            proj_dim=int(args.get("proj_dim", 512)),
            markov_dim=self.markov_dim,
            scalar_dim=self.scalar_dim,
            scalar_proj_dim=int(args.get("scalar_proj_dim", 32)),
            head_hidden_size=int(args.get("head_hidden_size", 512)),
            vocab_size=self.vocab_size,
            dropout=0.0,
            use_hidden=use_hidden,
        ).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.arms = tuple(int(x) for x in self.arms)
        if not self.arms:
            raise ValueError("DFLASH post-draft policy needs at least one arm")
        logger.info(
            "Loaded DFLASH post-draft policy. path=%s hidden=%s markov=%s scalar=%s alpha=%s arms=%s",
            path,
            self.hidden_size,
            self.markov_dim,
            self.scalar_dim,
            self.alpha,
            self.arms,
        )

    @torch.inference_mode()
    def select_verify_lens(
        self,
        *,
        hidden: torch.Tensor,
        prev_token_ids: torch.Tensor | None = None,
        scalar: torch.Tensor,
        max_block_size: int,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"post-draft hidden must be [bs, slots, hidden], got {hidden.shape}")
        slots = min(int(hidden.shape[1]), self.num_slots, max(1, int(max_block_size) - 1))
        hidden = hidden[:, :slots, :]
        if self.markov_dim > 0:
            if prev_token_ids is None:
                raise ValueError("post-draft policy checkpoint expects previous token ids")
            prev_token_ids = prev_token_ids[:, :slots].to(device=hidden.device, dtype=torch.int64)
        else:
            prev_token_ids = torch.empty(
                (hidden.shape[0], slots),
                dtype=torch.int64,
                device=hidden.device,
            )
        scalar = scalar[:, :slots, :] if self.scalar_dim > 0 else scalar[:, :slots, :0]
        logits = self.model(hidden, prev_token_ids, scalar)
        cond_probs = torch.sigmoid(logits)
        survival = torch.cumprod(cond_probs, dim=1)
        budgets = torch.tensor(
            [max(1, min(int(arm), int(max_block_size))) - 1 for arm in self.arms],
            dtype=torch.int64,
            device=survival.device,
        ).clamp(min=1, max=slots)
        expected = torch.stack(
            [survival[:, : int(b.item())].sum(dim=1) for b in budgets], dim=1
        )
        expected_full = survival[:, :slots].sum(dim=1, keepdim=True)
        ok = expected >= float(self.alpha) * expected_full
        fallback = torch.full(
            (ok.shape[0],), ok.shape[1] - 1, dtype=torch.long, device=ok.device
        )
        chosen = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)
        verify_lens = budgets[chosen] + 1
        return verify_lens.to(dtype=torch.int64)


def dflash_confidence_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return [entropy, token_prob, top1-top2 prob margin, token_logprob]."""

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    token_logprob = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    token_prob = token_logprob.exp()
    return torch.stack(
        [entropy, token_prob, top2[:, 0] - top2[:, 1], token_logprob], dim=-1
    )
