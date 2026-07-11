from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


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
        dropout: float = 0.0,
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
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_slots),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(seq)
        return self.head(torch.cat([h[-1], static], dim=-1))


class FlatSurvivalHead(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        seq_dim: int,
        static_dim: int,
        num_slots: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(seq_len * seq_dim + static_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_slots),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([seq.flatten(1), static], dim=-1))


class DFlashSurvivalBlockPolicy:
    """Runtime pre-draft block-size policy trained from DFlash trace rows."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        arms: Iterable[int] = (4, 8, 12, 16),
        alpha: float = 0.90,
        monotonicize_probs: bool = True,
        internal_feature_dim: int = 128,
        internal_feature_window: int = 16,
        internal_feature_seed: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.config: dict[str, Any] = checkpoint["config"]
        self.feature_set = str(self.config.get("feature_set", "all"))
        if self.feature_set not in {"internal_only", "internal_window"}:
            raise ValueError(
                f"runtime policy only supports internal_only/internal_window checkpoints, got {self.feature_set!r}"
            )

        seq_shape = list(self.config.get("seq_shape", []))
        if len(seq_shape) != 2:
            raise ValueError(f"checkpoint is missing seq_shape, got {seq_shape!r}")
        seq_len, seq_dim = int(seq_shape[0]), int(seq_shape[1])
        static_dim = int(self.config.get("static_dim", 0))
        num_slots = int(self.config["num_slots"])
        hidden_size = int(self.config.get("hidden_size", 192))
        architecture = str(self.config.get("architecture", "gru"))

        if architecture == "gru":
            model: nn.Module = TemporalSurvivalHead(
                seq_dim=seq_dim,
                static_dim=static_dim,
                num_slots=num_slots,
                hidden_size=hidden_size,
                dropout=0.0,
            )
        elif architecture == "mlp":
            model = FlatSurvivalHead(
                seq_len=seq_len,
                seq_dim=seq_dim,
                static_dim=static_dim,
                num_slots=num_slots,
                hidden_size=hidden_size,
                dropout=0.0,
            )
        else:
            raise ValueError(f"unsupported survival policy architecture {architecture!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.seq_normalizer = Normalizer.from_payload(self.config["seq_normalizer"])
        self.static_normalizer = Normalizer.from_payload(self.config["static_normalizer"])
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.num_slots = num_slots
        self.arms = tuple(int(x) for x in arms)
        if self.arms != tuple(sorted(set(self.arms))):
            raise ValueError(f"arms must be sorted and unique, got {self.arms}")
        max_budget = max(self.arms) - 1
        if max_budget > num_slots:
            raise ValueError(f"arms {self.arms} require {max_budget} survival slots, checkpoint has {num_slots}")
        self.alpha = float(alpha)
        self.monotonicize_probs = bool(monotonicize_probs)
        if self.feature_set == "internal_window":
            self.internal_feature_window = seq_len
            self.internal_feature_dim = seq_dim - 1
        elif self.feature_set == "internal_only":
            self.internal_feature_window = int(internal_feature_window)
            self.internal_feature_dim = static_dim
        else:
            self.internal_feature_window = int(internal_feature_window)
            self.internal_feature_dim = int(internal_feature_dim)
        self.internal_feature_seed = int(internal_feature_seed)

        self.device = torch.device("cpu")
        self.projector: torch.Tensor | None = None
        self.history: deque[torch.Tensor] = deque(maxlen=self.internal_feature_window)
        self.last_probs: torch.Tensor | None = None
        self.last_expected_by_arm: torch.Tensor | None = None

    def reset(self, draft_model: nn.Module, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device if device is not None else next(draft_model.parameters()).device)
        self.model.to(self.device).eval()
        self.history.clear()
        self.last_probs = None
        self.last_expected_by_arm = None

        if self.internal_feature_dim <= 0:
            self.projector = None
            self.internal_feature_dim = int(draft_model.config.hidden_size)
            return

        generator = torch.Generator(device="cpu").manual_seed(self.internal_feature_seed)
        projector = torch.randn(
            int(draft_model.config.hidden_size),
            self.internal_feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.projector = (projector / math.sqrt(int(draft_model.config.hidden_size))).to(self.device)

    @torch.inference_mode()
    def observe_context(self, draft_model: nn.Module, target_hidden: torch.Tensor) -> None:
        fused = draft_model.hidden_norm(draft_model.fc(target_hidden))
        if self.projector is not None:
            fused = fused.float() @ self.projector
        fused = fused[0].detach().float().cpu()
        for row in fused:
            self.history.append(row)

    @torch.inference_mode()
    def select_block_size(self) -> int:
        seq, static = self._make_features()
        seq = self.seq_normalizer.apply(seq.to(self.device))
        static = self.static_normalizer.apply(static.to(self.device))
        probs = torch.sigmoid(self.model(seq, static))[0]
        if self.monotonicize_probs:
            probs = torch.cummin(probs, dim=0).values
        budgets = torch.tensor([arm - 1 for arm in self.arms], device=probs.device, dtype=torch.long)
        expected_by_arm = torch.stack([probs[: int(budget.item())].sum() for budget in budgets])
        expected_full = expected_by_arm[-1]
        ok = expected_by_arm >= self.alpha * expected_full
        chosen_idx = int(torch.argmax(ok.int()).item()) if bool(ok.any()) else len(self.arms) - 1
        self.last_probs = probs.detach().cpu()
        self.last_expected_by_arm = expected_by_arm.detach().cpu()
        return self.arms[chosen_idx]

    def _make_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_set == "internal_only":
            if not self.history:
                static = torch.zeros((1, self.static_dim), dtype=torch.float32)
            else:
                static = self.history[-1].view(1, -1)
            seq = torch.zeros((1, self.seq_len, self.seq_dim), dtype=torch.float32)
            return seq, static

        tail = list(self.history)[-self.internal_feature_window :]
        pad = self.internal_feature_window - len(tail)
        rows = [torch.zeros(self.internal_feature_dim, dtype=torch.float32) for _ in range(pad)] + tail
        mask = [0.0] * pad + [1.0] * len(tail)
        seq_rows = [
            torch.cat([row.float(), torch.tensor([mask_i], dtype=torch.float32)])
            for row, mask_i in zip(rows, mask)
        ]
        seq = torch.stack(seq_rows).view(1, self.internal_feature_window, self.internal_feature_dim + 1)
        static = torch.zeros((1, self.static_dim), dtype=torch.float32)
        return seq, static


class DFlashOracleArmBlockPolicy:
    """Runtime block-size policy trained as an oracle-arm classifier."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        threshold: float | None = None,
        internal_feature_dim: int = 128,
        internal_feature_window: int = 16,
        internal_feature_seed: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config: dict[str, Any] = checkpoint["config"]
        self.feature_set = str(self.config.get("feature_set", "all"))
        if self.feature_set not in {"internal_only", "internal_window"}:
            raise ValueError(
                "runtime oracle-arm policy only supports internal_only/internal_window checkpoints, "
                f"got {self.feature_set!r}"
            )

        seq_shape = list(self.config.get("seq_shape", []))
        if len(seq_shape) != 2:
            raise ValueError(f"checkpoint is missing seq_shape, got {seq_shape!r}")
        seq_len, seq_dim = int(seq_shape[0]), int(seq_shape[1])
        static_dim = int(self.config.get("static_dim", 0))
        num_arms = int(self.config.get("num_arms", len(self.config.get("arms", []))))
        hidden_size = int(self.config.get("hidden_size", 192))
        architecture = str(self.config.get("architecture", "gru"))

        if architecture == "gru":
            model: nn.Module = TemporalSurvivalHead(
                seq_dim=seq_dim,
                static_dim=static_dim,
                num_slots=num_arms,
                hidden_size=hidden_size,
                dropout=0.0,
            )
        elif architecture == "mlp":
            model = FlatSurvivalHead(
                seq_len=seq_len,
                seq_dim=seq_dim,
                static_dim=static_dim,
                num_slots=num_arms,
                hidden_size=hidden_size,
                dropout=0.0,
            )
        else:
            raise ValueError(f"unsupported oracle-arm policy architecture {architecture!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.seq_normalizer = Normalizer.from_payload(self.config["seq_normalizer"])
        self.static_normalizer = Normalizer.from_payload(self.config["static_normalizer"])
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.arms = tuple(int(x) for x in self.config["arms"])
        if self.arms != tuple(sorted(set(self.arms))):
            raise ValueError(f"arms must be sorted and unique, got {self.arms}")
        self.threshold = None if threshold is None else float(threshold)
        if self.threshold is not None and not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

        if self.feature_set == "internal_window":
            self.internal_feature_window = seq_len
            self.internal_feature_dim = seq_dim - 1
        elif self.feature_set == "internal_only":
            self.internal_feature_window = int(internal_feature_window)
            self.internal_feature_dim = static_dim
        else:
            self.internal_feature_window = int(internal_feature_window)
            self.internal_feature_dim = int(internal_feature_dim)
        self.internal_feature_seed = int(internal_feature_seed)

        self.device = torch.device("cpu")
        self.projector: torch.Tensor | None = None
        self.history: deque[torch.Tensor] = deque(maxlen=self.internal_feature_window)
        self.last_probs: torch.Tensor | None = None
        self.last_expected_by_arm: torch.Tensor | None = None

    def reset(self, draft_model: nn.Module, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device if device is not None else next(draft_model.parameters()).device)
        self.model.to(self.device).eval()
        self.history.clear()
        self.last_probs = None
        self.last_expected_by_arm = None

        if self.internal_feature_dim <= 0:
            self.projector = None
            self.internal_feature_dim = int(draft_model.config.hidden_size)
            return

        generator = torch.Generator(device="cpu").manual_seed(self.internal_feature_seed)
        projector = torch.randn(
            int(draft_model.config.hidden_size),
            self.internal_feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.projector = (projector / math.sqrt(int(draft_model.config.hidden_size))).to(self.device)

    @torch.inference_mode()
    def observe_context(self, draft_model: nn.Module, target_hidden: torch.Tensor) -> None:
        fused = draft_model.hidden_norm(draft_model.fc(target_hidden))
        if self.projector is not None:
            fused = fused.float() @ self.projector
        fused = fused[0].detach().float().cpu()
        for row in fused:
            self.history.append(row)

    @torch.inference_mode()
    def select_block_size(self) -> int:
        seq, static = self._make_features()
        seq = self.seq_normalizer.apply(seq.to(self.device))
        static = self.static_normalizer.apply(static.to(self.device))
        probs = torch.softmax(self.model(seq, static), dim=-1)[0]
        if self.threshold is None:
            chosen_idx = int(torch.argmax(probs).item())
        else:
            cdf = torch.cumsum(probs, dim=0)
            ok = cdf >= self.threshold
            chosen_idx = int(torch.argmax(ok.int()).item()) if bool(ok.any()) else len(self.arms) - 1
            self.last_expected_by_arm = cdf.detach().cpu()
        self.last_probs = probs.detach().cpu()
        return int(self.arms[chosen_idx])

    def _make_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_set == "internal_only":
            if not self.history:
                static = torch.zeros((1, self.static_dim), dtype=torch.float32)
            else:
                static = self.history[-1].view(1, -1)
            seq = torch.zeros((1, self.seq_len, self.seq_dim), dtype=torch.float32)
            return seq, static

        tail = list(self.history)[-self.internal_feature_window :]
        pad = self.internal_feature_window - len(tail)
        rows = [torch.zeros(self.internal_feature_dim, dtype=torch.float32) for _ in range(pad)] + tail
        mask = [0.0] * pad + [1.0] * len(tail)
        seq_rows = [
            torch.cat([row.float(), torch.tensor([mask_i], dtype=torch.float32)])
            for row, mask_i in zip(rows, mask)
        ]
        seq = torch.stack(seq_rows).view(1, self.internal_feature_window, self.internal_feature_dim + 1)
        static = torch.zeros((1, self.static_dim), dtype=torch.float32)
        return seq, static
