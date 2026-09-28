from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class DFlashEntropyMLP(nn.Module):
    """Small MLP used by the offline pre-draft entropy predictor."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        proj_dim: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


class DFlashHorizonRuntime(nn.Module):
    """Runtime module for older fused-context horizon checkpoints."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        proj_dim: int,
        hidden_size: int,
        architecture: str,
        num_layers: int,
        context_window: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.context_window = int(context_window)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        if architecture == "last_mlp":
            self.encoder = None
            pooled_dim = proj_dim
        elif architecture == "gru":
            self.encoder = nn.GRU(
                input_size=proj_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            pooled_dim = hidden_size
        elif architecture == "transformer":
            nhead = 8
            if proj_dim % nhead != 0:
                raise ValueError(f"proj_dim={proj_dim} must be divisible by nhead={nhead}")
            self.pos_embed = nn.Parameter(torch.zeros(1, self.context_window, proj_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=nhead,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            pooled_dim = proj_dim
        else:
            raise ValueError(f"unsupported horizon architecture {architecture!r}")
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last_valid = (positions * (mask > 0.5)).max(dim=1).values.long()
        if self.architecture == "last_mlp":
            gather_idx = last_valid.view(-1, 1, 1).expand(-1, 1, features.shape[-1])
            pooled = self.input_proj(features.float().gather(dim=1, index=gather_idx).squeeze(1))
            return self.head(pooled)

        x = self.input_proj(features.float())
        x = x * mask.unsqueeze(-1)
        if self.architecture == "gru":
            encoded, _ = self.encoder(x)
        else:
            encoded = self.encoder(
                x + self.pos_embed[:, : x.shape[1], :],
                src_key_padding_mask=mask < 0.5,
            )
        gather_idx = last_valid.view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        return self.head(encoded.gather(dim=1, index=gather_idx).squeeze(1))


def _torch_load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a dict checkpoint at {path}, got {type(checkpoint)}.")
    return checkpoint


def _as_tensor(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    try:
        return torch.tensor(value, dtype=torch.float32, device=device)
    except Exception as e:
        raise ValueError(f"Could not convert checkpoint field {name!r} to tensor: {e}") from e


@dataclass
class DFlashPredraftEntropyPolicy:
    """Checkpoint-backed policy that maps fused context to a DFlash block size.

    The model predicts the draft-token entropy curve before drafting. A threshold
    policy converts that curve into a per-request draft budget; SGLang currently
    executes one rectangular block per batch, so per-request budgets are aggregated
    to one runtime block size.
    """

    checkpoint_path: str
    threshold: float = 2.60
    min_budget: int = 1
    max_budget: int = 15
    monotone: bool = True
    aggregate: str = "max"
    aggregate_percentile: float = 0.90
    block_arms: Sequence[int] | None = None
    device: torch.device | str = "cuda"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"DFLASH pre-draft entropy checkpoint not found: {path}")
        if self.min_budget < 1:
            raise ValueError(f"min_budget must be >= 1, got {self.min_budget}.")
        if self.max_budget < self.min_budget:
            raise ValueError(
                f"max_budget must be >= min_budget, got {self.max_budget} < {self.min_budget}."
            )
        if self.aggregate not in ("max", "mean", "median", "percentile", "min"):
            raise ValueError(
                "aggregate must be one of max, mean, median, percentile, min; "
                f"got {self.aggregate!r}."
            )
        if not (0.0 <= float(self.aggregate_percentile) <= 1.0):
            raise ValueError(
                "aggregate_percentile must be in [0, 1], "
                f"got {self.aggregate_percentile}."
            )

        checkpoint = _torch_load_checkpoint(str(path), self.device)
        config = checkpoint.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("DFLASH entropy checkpoint field 'config' must be a dict.")

        self.policy_kind = "entropy"
        self.horizon_objective = ""
        self.horizon_arms = (4, 8, 12, 16)
        if "model_state_dict" in checkpoint and "model" not in checkpoint:
            self._init_horizon_policy(path, checkpoint, config)
            return

        output_dim = int(config.get("num_slots", config.get("output_dim", 15)))
        self.model = DFlashEntropyMLP(
            input_dim=int(config["input_dim"]),
            output_dim=output_dim,
            proj_dim=int(config.get("proj_dim", 512)),
            hidden_size=int(config.get("hidden_size", 512)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.target_mean = _as_tensor(
            checkpoint.get("target_mean", torch.zeros(output_dim)),
            device=self.device,
            name="target_mean",
        ).view(1, -1)
        self.target_std = _as_tensor(
            checkpoint.get("target_std", torch.ones(output_dim)),
            device=self.device,
            name="target_std",
        ).view(1, -1)
        self.max_budget = min(int(self.max_budget), int(output_dim))

        logger.info(
            "Loaded DFLASH pre-draft entropy policy. path=%s input_dim=%s output_dim=%s "
            "threshold=%s budget=[%s,%s] aggregate=%s",
            path,
            int(config["input_dim"]),
            output_dim,
            self.threshold,
            self.min_budget,
            self.max_budget,
            self.aggregate,
        )

    def _init_horizon_policy(
        self, path: Path, checkpoint: dict[str, Any], config: dict[str, Any]
    ) -> None:
        required = (
            "input_dim",
            "proj_dim",
            "hidden_size",
            "num_slots",
            "architecture",
            "num_layers",
            "context_window",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"DFLASH horizon checkpoint is missing config keys: {missing}")
        output_dim = int(config["num_slots"])
        self.policy_kind = "horizon"
        self.horizon_objective = str(config.get("objective", "survival_bce"))
        if self.horizon_objective not in (
            "survival_bce",
            "censored_survival_bce",
            "hazard",
        ):
            raise ValueError(
                "DFLASH runtime horizon policy supports survival_bce, "
                f"censored_survival_bce, and hazard; got {self.horizon_objective!r}."
            )
        raw_arms = (
            self.block_arms
            if self.block_arms is not None
            else config.get("arms", self.horizon_arms)
        )
        self.horizon_arms = tuple(
            int(x) for x in raw_arms if 2 <= int(x) <= int(output_dim) + 1
        )
        if not self.horizon_arms:
            self.horizon_arms = tuple(
                arm for arm in (4, 8, 12, 16) if arm <= int(output_dim) + 1
            )
        self.model = DFlashHorizonRuntime(
            input_dim=int(config["input_dim"]),
            output_dim=output_dim,
            proj_dim=int(config["proj_dim"]),
            hidden_size=int(config["hidden_size"]),
            architecture=str(config["architecture"]),
            num_layers=int(config["num_layers"]),
            context_window=int(config["context_window"]),
            dropout=0.0,
        ).to(self.device)
        state_dict = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if not key.startswith("aux_arm_head.")
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.max_budget = min(int(self.max_budget), output_dim)
        logger.info(
            "Loaded DFLASH pre-draft horizon policy. path=%s input_dim=%s output_dim=%s "
            "objective=%s alpha=%s arms=%s budget=[%s,%s] aggregate=%s",
            path,
            int(config["input_dim"]),
            output_dim,
            self.horizon_objective,
            self.threshold,
            self.horizon_arms,
            self.min_budget,
            self.max_budget,
            self.aggregate,
        )

    @torch.inference_mode()
    def predict_entropy(self, fused_context: torch.Tensor) -> torch.Tensor:
        if self.policy_kind != "entropy":
            raise RuntimeError("predict_entropy is only available for entropy checkpoints")
        if fused_context.ndim == 1:
            fused_context = fused_context.unsqueeze(0)
        if fused_context.ndim != 2:
            raise ValueError(
                "DFLASH pre-draft policy expected fused_context with shape [bs, hidden], "
                f"got {tuple(fused_context.shape)}."
            )
        expected_dim = int(self.model.net[0].in_features)
        if int(fused_context.shape[-1]) != expected_dim:
            raise ValueError(
                "DFLASH pre-draft policy input dimension mismatch. "
                f"Expected {expected_dim}, got {int(fused_context.shape[-1])}."
            )
        features = fused_context.to(device=self.device, non_blocking=True)
        pred = self.model(features) * self.target_std + self.target_mean
        pred = pred[:, : int(self.max_budget)]
        if self.monotone and pred.numel() > 0:
            pred = torch.cummax(pred, dim=-1).values
        return pred

    @torch.inference_mode()
    def predict_survival(self, fused_context: torch.Tensor) -> torch.Tensor:
        if self.policy_kind != "horizon":
            raise RuntimeError("predict_survival is only available for horizon checkpoints")
        if fused_context.ndim == 1:
            fused_context = fused_context.unsqueeze(0)
        if fused_context.ndim != 2:
            raise ValueError(
                "DFLASH pre-draft horizon policy expected fused_context with shape [bs, hidden], "
                f"got {tuple(fused_context.shape)}."
            )
        expected_dim = int(self.model.input_proj[0].in_features)
        if int(fused_context.shape[-1]) != expected_dim:
            raise ValueError(
                "DFLASH pre-draft horizon policy input dimension mismatch. "
                f"Expected {expected_dim}, got {int(fused_context.shape[-1])}."
            )
        bs = int(fused_context.shape[0])
        context_window = int(self.model.context_window)
        features = torch.zeros(
            (bs, context_window, expected_dim),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((bs, context_window), dtype=torch.float32, device=self.device)
        features[:, -1, :] = fused_context.to(device=self.device, non_blocking=True).float()
        mask[:, -1] = 1.0
        logits = self.model(features, mask)[:, : int(self.max_budget)]
        if self.horizon_objective == "hazard":
            probs = torch.cumprod(torch.sigmoid(logits), dim=-1)
        else:
            probs = torch.sigmoid(logits)
            if self.monotone and probs.numel() > 0:
                probs = torch.cummin(probs, dim=-1).values
        return probs

    @torch.inference_mode()
    def request_budgets(self, fused_context: torch.Tensor) -> torch.Tensor:
        if self.policy_kind == "horizon":
            survival = self.predict_survival(fused_context)
            arms = torch.tensor(
                [min(int(arm) - 1, int(self.max_budget)) for arm in self.horizon_arms],
                device=survival.device,
                dtype=torch.long,
            )
            expected_by_arm = torch.stack(
                [survival[:, : int(budget.item())].sum(dim=1) for budget in arms],
                dim=1,
            )
            expected_full = expected_by_arm[:, -1:].clamp_min(1e-6)
            ok = expected_by_arm >= float(self.threshold) * expected_full
            fallback = torch.full(
                (survival.shape[0],),
                len(self.horizon_arms) - 1,
                device=survival.device,
                dtype=torch.long,
            )
            chosen = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)
            budget = arms[chosen]
            budget = torch.clamp(budget, min=int(self.min_budget), max=int(self.max_budget))
            return budget.to(torch.int64)

        entropy = self.predict_entropy(fused_context)
        ok = entropy <= float(self.threshold)
        first_bad = (~ok).to(torch.int64).argmax(dim=1)
        all_ok = ok.all(dim=1)
        budget = torch.where(
            all_ok,
            torch.full_like(first_bad, int(self.max_budget)),
            first_bad,
        )
        budget = torch.clamp(budget, min=int(self.min_budget), max=int(self.max_budget))
        return budget.to(torch.int64)

    @torch.inference_mode()
    def request_block_sizes(
        self,
        fused_context: torch.Tensor,
        *,
        max_block_size: int,
    ) -> torch.Tensor:
        budgets = self.request_budgets(fused_context)
        block_sizes = budgets + 1
        return torch.clamp(block_sizes, min=2, max=int(max_block_size)).to(torch.int64)

    @torch.inference_mode()
    def select_block_size(
        self,
        fused_context: torch.Tensor,
        *,
        max_block_size: int,
    ) -> int:
        budgets = self.request_budgets(fused_context)
        if budgets.numel() == 0:
            return int(max_block_size)

        if self.aggregate == "max":
            batch_budget = budgets.max()
        elif self.aggregate == "min":
            batch_budget = budgets.min()
        elif self.aggregate == "mean":
            batch_budget = torch.ceil(budgets.float().mean()).to(torch.int64)
        elif self.aggregate == "median":
            batch_budget = torch.ceil(budgets.float().median()).to(torch.int64)
        else:
            q = torch.quantile(budgets.float(), float(self.aggregate_percentile))
            batch_budget = torch.ceil(q).to(torch.int64)

        block_size = int(batch_budget.item()) + 1
        block_size = max(2, min(int(block_size), int(max_block_size)))
        return block_size
