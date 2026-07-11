from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn


def parse_arms(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        arms = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        arms = [int(x) for x in value]
    arms = sorted(set(arms))
    if not arms:
        raise ValueError("survival policy arms cannot be empty")
    if arms[0] < 2:
        raise ValueError(f"survival policy arms must be >= 2, got {arms}")
    return arms


@dataclass(frozen=True)
class Normalizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Normalizer":
        return cls(mean=tuple(payload.get("mean", ())), std=tuple(payload.get("std", ())))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x.float()
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
        return (x.float() - mean) / std


class TemporalSurvivalHead(nn.Module):
    def __init__(
        self,
        *,
        seq_dim: int,
        static_dim: int,
        num_slots: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size // 2, num_slots),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(seq)
        return self.head(torch.cat([h[-1], static], dim=-1))


class DFlashSurvivalBatchController:
    """Batch-level DFlash block selector from a survival checkpoint.

    DFlash serving currently uses one verify block size for the whole batch. This
    controller predicts per-request survival curves, averages expected accepted
    draft length over the batch for each arm, and chooses the smallest arm whose
    batch expectation preserves alpha of the largest-arm expectation.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        arms: Iterable[int],
        alpha: float,
        device: torch.device | str,
        draft_model: nn.Module,
        monotonicize_probs: bool = True,
        internal_feature_seed: int = 0,
    ) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config: dict[str, Any] = checkpoint["config"]
        if str(config.get("feature_set")) != "internal_window":
            raise ValueError(
                "SGLang DFlash survival controller currently expects feature_set='internal_window', "
                f"got {config.get('feature_set')!r}"
            )
        if str(config.get("architecture", "gru")) != "gru":
            raise ValueError("SGLang DFlash survival controller currently expects architecture='gru'")

        seq_len, seq_dim = [int(x) for x in config["seq_shape"]]
        static_dim = int(config.get("static_dim", 0))
        num_slots = int(config["num_slots"])
        hidden_size = int(config.get("hidden_size", 192))
        model = TemporalSurvivalHead(
            seq_dim=seq_dim,
            static_dim=static_dim,
            num_slots=num_slots,
            hidden_size=hidden_size,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.seq_normalizer = Normalizer.from_payload(config["seq_normalizer"])
        self.static_normalizer = Normalizer.from_payload(config["static_normalizer"])
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.num_slots = num_slots
        self.arms = parse_arms(tuple(arms))
        self.alpha = float(alpha)
        self.monotonicize_probs = bool(monotonicize_probs)
        self.internal_feature_dim = seq_dim - 1
        self.histories: dict[str, deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=self.seq_len)
        )
        max_budget = max(self.arms) - 1
        if max_budget > self.num_slots:
            raise ValueError(
                f"arms {self.arms} require {max_budget} slots but checkpoint has {self.num_slots}"
            )

        generator = torch.Generator(device="cpu").manual_seed(int(internal_feature_seed))
        projector = torch.randn(
            int(draft_model.config.hidden_size),
            self.internal_feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.projector = (projector / math.sqrt(int(draft_model.config.hidden_size))).to(self.device)

        self.last_block_size = self.arms[-1]
        self.last_mean_expected_by_arm: list[float] = []
        self.last_batch_size = 0

    @torch.inference_mode()
    def observe_context(
        self,
        *,
        draft_model: nn.Module,
        req_ids: Sequence[str],
        target_hidden: torch.Tensor,
        ctx_lens: torch.Tensor,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        fused = draft_model.hidden_norm(draft_model.fc(target_hidden))
        fused = fused.float() @ self.projector
        lens = ctx_lens.detach().to("cpu", dtype=torch.int64).tolist()
        start = 0
        for req_id, length in zip(req_ids, lens, strict=False):
            history = self.histories[str(req_id)]
            end = start + int(length)
            if end > start:
                for row in fused[start:end].detach():
                    history.append(row.float())
            start = end

    def _features_for_req(self, req_id: str) -> torch.Tensor:
        history = self.histories[str(req_id)]
        tail = list(history)[-self.seq_len :]
        pad = self.seq_len - len(tail)
        zero = torch.zeros(self.internal_feature_dim, dtype=torch.float32, device=self.device)
        rows = [zero for _ in range(pad)] + [row.to(self.device) for row in tail]
        mask = [0.0] * pad + [1.0] * len(tail)
        seq_rows = [
            torch.cat([row.float(), torch.tensor([mask_i], dtype=torch.float32, device=self.device)])
            for row, mask_i in zip(rows, mask, strict=True)
        ]
        return torch.stack(seq_rows)

    @torch.inference_mode()
    def select_block_size(self, req_ids: Sequence[str]) -> int:
        if not req_ids:
            self.last_block_size = self.arms[-1]
            self.last_batch_size = 0
            self.last_mean_expected_by_arm = []
            return self.last_block_size

        seq = torch.stack([self._features_for_req(str(req_id)) for req_id in req_ids], dim=0)
        static = torch.zeros((len(req_ids), self.static_dim), dtype=torch.float32, device=self.device)
        seq = self.seq_normalizer.apply(seq)
        static = self.static_normalizer.apply(static)
        probs = torch.sigmoid(self.model(seq, static))
        if self.monotonicize_probs:
            probs = torch.cummin(probs, dim=1).values

        budgets = torch.tensor([arm - 1 for arm in self.arms], dtype=torch.long, device=self.device)
        expected_by_arm = torch.stack(
            [probs[:, : int(budget.item())].sum(dim=1) for budget in budgets],
            dim=1,
        )
        mean_expected = expected_by_arm.mean(dim=0)
        expected_full = mean_expected[-1]
        ok = mean_expected >= self.alpha * expected_full
        chosen_idx = int(torch.argmax(ok.int()).item()) if bool(ok.any()) else len(self.arms) - 1
        self.last_block_size = int(self.arms[chosen_idx])
        self.last_batch_size = len(req_ids)
        self.last_mean_expected_by_arm = [float(x) for x in mean_expected.detach().cpu().tolist()]
        return self.last_block_size
