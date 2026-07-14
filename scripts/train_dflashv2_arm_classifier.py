from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Shard:
    path: Path
    rows: int
    features: np.ndarray
    mask: np.ndarray
    accepted_len: np.ndarray


def _parse_ints(value: str) -> tuple[int, ...]:
    out = tuple(int(x) for x in value.split(",") if x)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _load_shards(trace_dirs: list[Path]) -> list[Shard]:
    shards: list[Shard] = []
    for trace_dir in trace_dirs:
        manifest_path = trace_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing trace manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["shards"]:
            rows = int(item["rows"])
            if rows <= 0:
                continue
            shard_dir = trace_dir / item["path"]
            shards.append(
                Shard(
                    path=shard_dir,
                    rows=rows,
                    features=np.load(shard_dir / "features.npy", mmap_mode="r"),
                    mask=np.load(shard_dir / "mask.npy", mmap_mode="r"),
                    accepted_len=np.load(shard_dir / "accepted_len.npy", mmap_mode="r"),
                )
            )
    if not shards:
        raise ValueError(f"no non-empty shards found in {trace_dirs}")
    return shards


class DFlashV2ArmDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        shards: list[Shard],
        indices: np.ndarray,
        *,
        arms: tuple[int, ...],
        last_only: bool = False,
    ) -> None:
        self.shards = shards
        self.indices = np.asarray(indices, dtype=np.int64)
        self.arms = tuple(sorted(arms))
        self.budgets = np.asarray([arm - 1 for arm in self.arms], dtype=np.float32)
        self.last_only = last_only
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += shard.rows
        self.total_rows = total
        first = self.shards[0]
        self.context_window = 1 if last_only else int(first.features.shape[1])
        self.input_dim = int(first.features.shape[2])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _locate(self, global_idx: int) -> tuple[Shard, int]:
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        return self.shards[shard_idx], global_idx - self.offsets[shard_idx]

    def _target_idx(self, accepted_len: float) -> int:
        ok = self.budgets >= float(accepted_len)
        if ok.any():
            return int(ok.argmax())
        return len(self.arms) - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard, local_idx = self._locate(global_idx)
        accepted_len = float(shard.accepted_len[local_idx])
        if self.last_only:
            raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
            valid = np.flatnonzero(raw_mask > 0.5)
            feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
            features = torch.from_numpy(
                np.asarray(shard.features[local_idx, feature_idx : feature_idx + 1], dtype=np.float16)
            )
            mask = torch.ones((1,), dtype=torch.float32) if valid.size else torch.zeros((1,), dtype=torch.float32)
        else:
            features = torch.from_numpy(np.asarray(shard.features[local_idx], dtype=np.float16))
            mask = torch.from_numpy(np.asarray(shard.mask[local_idx], dtype=np.float32))
        target_idx = torch.tensor(self._target_idx(accepted_len), dtype=torch.long)
        accepted = torch.tensor(accepted_len, dtype=torch.float32)
        return features, mask, target_idx, accepted


class CompactArmDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        target_idx: np.ndarray,
        accepted_len: np.ndarray,
    ) -> None:
        self.features = features
        self.mask = mask
        self.target_idx = target_idx
        self.accepted_len = accepted_len
        self.context_window = int(features.shape[1])
        self.input_dim = int(features.shape[2])

    def __len__(self) -> int:
        return int(self.target_idx.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(int(self.target_idx[idx]), dtype=torch.long),
            torch.tensor(float(self.accepted_len[idx]), dtype=torch.float32),
        )


def _target_idx_from_accepted(accepted_len: float, budgets: np.ndarray) -> int:
    ok = budgets >= float(accepted_len)
    if ok.any():
        return int(ok.argmax())
    return int(len(budgets) - 1)


def _materialize_last_feature_dataset(
    shards: list[Shard],
    indices: np.ndarray,
    *,
    arms: tuple[int, ...],
) -> CompactArmDataset:
    offsets: list[int] = []
    total = 0
    for shard in shards:
        offsets.append(total)
        total += shard.rows
    input_dim = int(shards[0].features.shape[2])
    budgets = np.asarray([arm - 1 for arm in sorted(arms)], dtype=np.float32)
    order = np.sort(np.asarray(indices, dtype=np.int64))
    features = np.empty((order.shape[0], 1, input_dim), dtype=np.float16)
    mask = np.empty((order.shape[0], 1), dtype=np.float32)
    target_idx = np.empty((order.shape[0],), dtype=np.int64)
    accepted = np.empty((order.shape[0],), dtype=np.float32)
    for out_idx, global_idx in enumerate(order):
        shard_idx = bisect.bisect_right(offsets, int(global_idx)) - 1
        local_idx = int(global_idx) - offsets[shard_idx]
        shard = shards[shard_idx]
        raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
        valid = np.flatnonzero(raw_mask > 0.5)
        feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
        features[out_idx, 0] = np.asarray(shard.features[local_idx, feature_idx], dtype=np.float16)
        mask[out_idx, 0] = 1.0 if valid.size else 0.0
        accepted_len = float(shard.accepted_len[local_idx])
        accepted[out_idx] = accepted_len
        target_idx[out_idx] = _target_idx_from_accepted(accepted_len, budgets)
    return CompactArmDataset(features, mask, target_idx, accepted)


class ArmClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        proj_dim: int,
        hidden_size: int,
        num_arms: int,
        architecture: str,
        num_layers: int,
        dropout: float,
        context_window: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
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
            self.pos_embed = nn.Parameter(torch.zeros(1, context_window, proj_dim))
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
            raise ValueError(f"unsupported architecture {architecture!r}")
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_arms),
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
        pooled = encoded.gather(dim=1, index=gather_idx).squeeze(1)
        return self.head(pooled)


def _make_splits(
    *,
    total_rows: int,
    max_total_rows: int | None,
    calibration_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    use_rows = total_rows if max_total_rows is None else min(total_rows, max_total_rows)
    selected = rng.choice(total_rows, size=use_rows, replace=False)
    rng.shuffle(selected)
    if calibration_rows >= use_rows:
        raise ValueError(f"calibration_rows={calibration_rows} must be smaller than selected rows={use_rows}")
    return selected[calibration_rows:], selected[:calibration_rows]


def _hist(values: torch.Tensor, n: int) -> list[int]:
    return torch.bincount(values.detach().cpu(), minlength=n).tolist()


def _choose_by_risk(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    cdf = probs.cumsum(dim=-1)
    ok = cdf >= threshold
    fallback = torch.full((probs.shape[0],), probs.shape[1] - 1, device=probs.device, dtype=torch.long)
    return torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)


def _decision_metrics(
    *,
    pred_idx: torch.Tensor,
    target_idx: torch.Tensor,
    accepted_len: torch.Tensor,
    arms: tuple[int, ...],
    prefix: str,
) -> dict[str, Any]:
    budgets = torch.tensor([arm - 1 for arm in arms], dtype=torch.float32, device=accepted_len.device)
    arm_tensor = torch.tensor(arms, dtype=torch.float32, device=accepted_len.device)
    pred_budget = budgets[pred_idx]
    target_budget = budgets[target_idx]
    fixed_budget = budgets[-1].expand_as(accepted_len)
    accepted_under_pred = torch.minimum(accepted_len, pred_budget)
    accepted_fixed = torch.minimum(accepted_len, fixed_budget)
    sufficient = pred_budget >= accepted_len
    wasted = torch.clamp(pred_budget - accepted_len, min=0.0)
    return {
        f"{prefix}_accuracy": (pred_idx == target_idx).float().mean().item(),
        f"{prefix}_within_one_arm": ((pred_idx - target_idx).abs() <= 1).float().mean().item(),
        f"{prefix}_under_rate": (~sufficient).float().mean().item(),
        f"{prefix}_sufficient_rate": sufficient.float().mean().item(),
        f"{prefix}_mean_block": arm_tensor[pred_idx].mean().item(),
        f"{prefix}_mean_target_block": arm_tensor[target_idx].mean().item(),
        f"{prefix}_mean_budget": pred_budget.mean().item(),
        f"{prefix}_mean_target_budget": target_budget.mean().item(),
        f"{prefix}_mean_accepted": accepted_under_pred.mean().item(),
        f"{prefix}_accept_retention": (accepted_under_pred.sum() / accepted_fixed.sum().clamp_min(1.0)).item(),
        f"{prefix}_accept_ratio": (accepted_under_pred / pred_budget.clamp_min(1.0)).mean().item(),
        f"{prefix}_mean_wasted_budget": wasted.mean().item(),
        f"{prefix}_pred_arm_hist": _hist(pred_idx, len(arms)),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    arms: tuple[int, ...],
    risk_thresholds: tuple[float, ...],
    temperature: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    accepted_chunks: list[torch.Tensor] = []
    loss_sum = 0.0
    rows = 0
    for features, mask, target_idx, accepted_len in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        logits = model(features, mask)
        loss = F.cross_entropy(logits, target_idx)
        batch = int(features.shape[0])
        loss_sum += float(loss.item()) * batch
        rows += batch
        logits_chunks.append(logits.detach().cpu())
        target_chunks.append(target_idx.detach().cpu())
        accepted_chunks.append(accepted_len.detach().cpu())
    logits = torch.cat(logits_chunks)
    target_idx = torch.cat(target_chunks)
    accepted_len = torch.cat(accepted_chunks)
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    pred_idx = probs.argmax(dim=-1)
    budgets = torch.tensor([arm - 1 for arm in arms], dtype=torch.float32)
    fixed_budget = budgets[-1].expand_as(accepted_len)
    fixed_accept = torch.minimum(accepted_len, fixed_budget)
    oracle_accept = torch.minimum(accepted_len, budgets[target_idx])
    metrics: dict[str, Any] = {
        "rows": rows,
        "ce": loss_sum / max(rows, 1),
        "mean_accepted_len": accepted_len.mean().item(),
        "target_arm_hist": _hist(target_idx, len(arms)),
        "fixed_b16_mean_accepted": fixed_accept.mean().item(),
        "fixed_b16_accept_ratio": (fixed_accept / fixed_budget.clamp_min(1.0)).mean().item(),
        "oracle_mean_block": torch.tensor(arms, dtype=torch.float32)[target_idx].mean().item(),
        "oracle_mean_accepted": oracle_accept.mean().item(),
        "oracle_accept_retention": (oracle_accept.sum() / fixed_accept.sum().clamp_min(1.0)).item(),
        "oracle_accept_ratio": (oracle_accept / budgets[target_idx].clamp_min(1.0)).mean().item(),
    }
    metrics.update(
        _decision_metrics(
            pred_idx=pred_idx,
            target_idx=target_idx,
            accepted_len=accepted_len,
            arms=arms,
            prefix="argmax",
        )
    )
    for threshold in risk_thresholds:
        risk_idx = _choose_by_risk(probs, threshold)
        metrics.update(
            _decision_metrics(
                pred_idx=risk_idx,
                target_idx=target_idx,
                accepted_len=accepted_len,
                arms=arms,
                prefix=f"risk{threshold:.2f}",
            )
        )
    return metrics


def _loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: tuple[int, ...],
    class_weights: torch.Tensor | None,
    ordinal_weight: float,
    under_weight: float,
    over_weight: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target_idx, weight=class_weights)
    if ordinal_weight <= 0:
        return ce
    probs = F.softmax(logits, dim=-1)
    budgets = torch.tensor([arm - 1 for arm in arms], dtype=torch.float32, device=logits.device)
    expected_budget = (probs * budgets.view(1, -1)).sum(dim=-1)
    under = torch.clamp(accepted_len - expected_budget, min=0.0)
    over = torch.clamp(expected_budget - accepted_len, min=0.0)
    return ce + ordinal_weight * (under_weight * under + over_weight * over).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    arms: tuple[int, ...],
    class_weights: torch.Tensor | None,
    ordinal_weight: float,
    under_weight: float,
    over_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    rows = 0
    for features, mask, target_idx, accepted_len in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        logits = model(features, mask)
        loss = _loss(
            logits,
            target_idx,
            accepted_len,
            arms=arms,
            class_weights=class_weights,
            ordinal_weight=ordinal_weight,
            under_weight=under_weight,
            over_weight=over_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch = int(features.shape[0])
        rows += batch
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target_idx).float().mean()
            ce = F.cross_entropy(logits, target_idx)
        for key, value in {"loss": loss, "ce": ce, "accuracy": acc}.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch
    return {key: value / rows for key, value in totals.items()}


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _collect_logits(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for features, mask, target_idx, _ in loader:
        logits_chunks.append(model(features.to(device), mask.to(device)).detach().cpu().clone())
        target_chunks.append(target_idx.detach().cpu().clone())
    return torch.cat(logits_chunks), torch.cat(target_chunks)


def fit_temperature(logits: torch.Tensor, target_idx: torch.Tensor) -> tuple[float, float]:
    log_temp = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = log_temp.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temp, target_idx)
        loss.backward()
        return loss

    before = F.cross_entropy(logits, target_idx).item()
    optimizer.step(closure)
    temp = float(log_temp.detach().exp().clamp(0.05, 20.0).item())
    after = F.cross_entropy(logits / temp, target_idx).item()
    return temp, before - after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFlashv2 direct arm classifier.")
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", type=_parse_ints, default=(4, 8, 12, 16))
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--architecture", choices=["last_mlp", "gru", "transformer"], default="gru")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ordinal-weight", type=float, default=0.02)
    parser.add_argument("--under-weight", type=float, default=1.0)
    parser.add_argument("--over-weight", type=float, default=0.25)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--risk-thresholds", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = tuple(sorted(args.arms))
    risk_thresholds = tuple(float(x) for x in args.risk_thresholds.split(",") if x)
    shards = _load_shards(args.trace_dir)
    total_rows = sum(shard.rows for shard in shards)
    train_indices, cal_indices = _make_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    last_only = args.architecture == "last_mlp"
    if last_only:
        print(
            json.dumps(
                {
                    "event": "materialize_last_features",
                    "train_rows": int(train_indices.shape[0]),
                    "calibration_rows": int(cal_indices.shape[0]),
                }
            ),
            flush=True,
        )
        train_ds = _materialize_last_feature_dataset(shards, train_indices, arms=arms)
        cal_ds = _materialize_last_feature_dataset(shards, cal_indices, arms=arms)
        print(json.dumps({"event": "materialize_last_features_done"}), flush=True)
    else:
        train_ds = DFlashV2ArmDataset(shards, train_indices, arms=arms, last_only=False)
        cal_ds = DFlashV2ArmDataset(shards, cal_indices, arms=arms, last_only=False)
    train_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    cal_loader = _make_loader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    class_weights = None
    if isinstance(cal_ds, CompactArmDataset):
        target_hist = np.bincount(cal_ds.target_idx, minlength=len(arms)).astype(np.int64)
    else:
        target_hist = np.zeros(len(arms), dtype=np.int64)
        for _, _, target_idx, _ in _make_loader(cal_ds, batch_size=args.batch_size, shuffle=False, num_workers=0):
            target_hist += np.bincount(target_idx.numpy(), minlength=len(arms))
    if args.class_weight_power > 0:
        freq = torch.tensor(target_hist, dtype=torch.float32).clamp_min(1.0)
        class_weights = (freq.sum() / freq).pow(args.class_weight_power)
        class_weights = (class_weights / class_weights.mean()).to(device)

    model = ArmClassifier(
        input_dim=train_ds.input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_arms=len(arms),
        architecture=args.architecture,
        num_layers=args.num_layers,
        dropout=args.dropout,
        context_window=train_ds.context_window,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = vars(args).copy()
    config.update(
        {
            "trace_dirs": [str(path) for path in args.trace_dir],
            "total_rows_available": total_rows,
            "train_rows": len(train_ds),
            "calibration_rows": len(cal_ds),
            "input_dim": train_ds.input_dim,
            "context_window": train_ds.context_window,
            "last_only": last_only,
            "arms": arms,
            "budgets": [arm - 1 for arm in arms],
            "calibration_target_hist": target_hist.tolist(),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best = math.inf
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            arms=arms,
            class_weights=class_weights,
            ordinal_weight=args.ordinal_weight,
            under_weight=args.under_weight,
            over_weight=args.over_weight,
        )
        cal_metrics = evaluate(
            model,
            cal_loader,
            device=device,
            arms=arms,
            risk_thresholds=risk_thresholds,
        )
        record = {"epoch": epoch, "train": train_metrics, "calibration": cal_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        score = float(cal_metrics["ce"])
        if score < best:
            best = score
            best_metrics = cal_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "calibration_metrics": cal_metrics,
                },
                args.output_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "calibration_metrics": best_metrics,
        },
        args.output_dir / "last.pt",
    )
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logits, target_idx = _collect_logits(model, cal_loader, device=device)
    temperature, nll_improvement = fit_temperature(logits, target_idx)
    calibrated_metrics = evaluate(
        model,
        cal_loader,
        device=device,
        arms=arms,
        risk_thresholds=risk_thresholds,
        temperature=temperature,
    )
    payload = {
        "temperature": temperature,
        "nll_improvement": nll_improvement,
        "metrics": calibrated_metrics,
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"calibrated": payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
