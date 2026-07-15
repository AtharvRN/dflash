from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from dataclasses import asdict, dataclass
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
    survival: np.ndarray
    accepted_len: np.ndarray


def _load_shards(trace_dir: Path) -> tuple[list[Shard], dict[str, Any]]:
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing trace manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    shards: list[Shard] = []
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
                survival=np.load(shard_dir / "survival.npy", mmap_mode="r"),
                accepted_len=np.load(shard_dir / "accepted_len.npy", mmap_mode="r"),
            )
        )
    if not shards:
        raise ValueError(f"no non-empty shards found in {trace_dir}")
    return shards, manifest


class HorizonTraceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, trace_dir: Path, *, max_rows: int | None = None) -> None:
        shards, manifest = _load_shards(trace_dir)
        self.trace_dir = trace_dir
        self.manifest = manifest
        self.shards = shards
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += shard.rows
        self.total_rows = total if max_rows is None else min(total, max_rows)
        first = self.shards[0]
        self.context_window = int(first.features.shape[1])
        self.hidden_size = int(first.features.shape[2])
        self.num_slots = int(first.survival.shape[1])

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.offsets, idx) - 1
        local_idx = idx - self.offsets[shard_idx]
        shard = self.shards[shard_idx]
        features = torch.from_numpy(np.asarray(shard.features[local_idx], dtype=np.float16))
        mask = torch.from_numpy(np.asarray(shard.mask[local_idx], dtype=np.float32))
        survival = torch.from_numpy(np.asarray(shard.survival[local_idx], dtype=np.float32))
        accepted_len = torch.tensor(float(shard.accepted_len[local_idx]), dtype=torch.float32)
        return features, mask, survival, accepted_len


class HorizonIndexedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        shards: list[Shard],
        indices: np.ndarray,
        *,
        last_only: bool = False,
    ) -> None:
        self.shards = shards
        self.indices = np.asarray(indices, dtype=np.int64)
        self.last_only = last_only
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += shard.rows
        self.total_rows = total
        first = self.shards[0]
        self.context_window = 1 if last_only else int(first.features.shape[1])
        self.hidden_size = int(first.features.shape[2])
        self.num_slots = int(first.survival.shape[1])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_idx = global_idx - self.offsets[shard_idx]
        shard = self.shards[shard_idx]
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
        survival = torch.from_numpy(np.asarray(shard.survival[local_idx], dtype=np.float32))
        accepted_len = torch.tensor(float(shard.accepted_len[local_idx]), dtype=torch.float32)
        return features, mask, survival, accepted_len


class CompactHorizonDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        survival: np.ndarray,
        accepted_len: np.ndarray,
    ) -> None:
        self.features = features
        self.mask = mask
        self.survival = survival
        self.accepted_len = accepted_len
        self.context_window = int(features.shape[1])
        self.hidden_size = int(features.shape[2])
        self.num_slots = int(survival.shape[1])

    def __len__(self) -> int:
        return int(self.accepted_len.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.mask[idx]),
            torch.from_numpy(self.survival[idx]),
            torch.tensor(float(self.accepted_len[idx]), dtype=torch.float32),
        )


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


def _load_shards_multi(trace_dirs: list[Path]) -> list[Shard]:
    shards: list[Shard] = []
    for trace_dir in trace_dirs:
        loaded, _ = _load_shards(trace_dir)
        shards.extend(loaded)
    if not shards:
        raise ValueError(f"no non-empty shards found in {trace_dirs}")
    return shards


def _materialize_last_feature_dataset(shards: list[Shard], indices: np.ndarray) -> CompactHorizonDataset:
    offsets: list[int] = []
    total = 0
    for shard in shards:
        offsets.append(total)
        total += shard.rows
    input_dim = int(shards[0].features.shape[2])
    num_slots = int(shards[0].survival.shape[1])
    order = np.sort(np.asarray(indices, dtype=np.int64))
    features = np.empty((order.shape[0], 1, input_dim), dtype=np.float16)
    mask = np.empty((order.shape[0], 1), dtype=np.float32)
    survival = np.empty((order.shape[0], num_slots), dtype=np.float32)
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
        survival[out_idx] = np.asarray(shard.survival[local_idx], dtype=np.float32)
        accepted[out_idx] = float(shard.accepted_len[local_idx])
    return CompactHorizonDataset(features, mask, survival, accepted)


def _compact_cache_ready(cache_dir: Path) -> bool:
    required = (
        cache_dir / "features.npy",
        cache_dir / "mask.npy",
        cache_dir / "survival.npy",
        cache_dir / "accepted_len.npy",
    )
    return all(path.exists() for path in required)


def _save_compact_dataset(dataset: CompactHorizonDataset, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", dataset.features)
    np.save(cache_dir / "mask.npy", dataset.mask)
    np.save(cache_dir / "survival.npy", dataset.survival)
    np.save(cache_dir / "accepted_len.npy", dataset.accepted_len)


def _load_compact_dataset(cache_dir: Path) -> CompactHorizonDataset:
    return CompactHorizonDataset(
        np.load(cache_dir / "features.npy", mmap_mode="r"),
        np.load(cache_dir / "mask.npy", mmap_mode="r"),
        np.load(cache_dir / "survival.npy", mmap_mode="r"),
        np.load(cache_dir / "accepted_len.npy", mmap_mode="r"),
    )


def _oracle_arm_idx(accepted_len: torch.Tensor, *, arms: tuple[int, ...]) -> torch.Tensor:
    budgets = torch.tensor([arm - 1 for arm in arms], dtype=torch.float32, device=accepted_len.device)
    ok = budgets.view(1, -1) >= accepted_len.view(-1, 1)
    fallback = torch.full((accepted_len.shape[0],), len(arms) - 1, dtype=torch.long, device=accepted_len.device)
    return torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)


def _survival_weights(num_slots: int, *, boundary_ks: tuple[int, ...], boundary_weight: float) -> torch.Tensor:
    weights = torch.ones((num_slots,), dtype=torch.float32)
    for k in boundary_ks:
        if 1 <= k <= num_slots:
            weights[k - 1] = float(boundary_weight)
    return weights / weights.mean().clamp_min(1e-6)


def _conditional_survival_mask(survival: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)


def _horizon_loss(
    logits: torch.Tensor,
    survival: torch.Tensor,
    *,
    objective: str,
    survival_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if objective == "survival_bce":
        if survival_weights is None:
            loss = F.binary_cross_entropy_with_logits(logits, survival)
        else:
            raw = F.binary_cross_entropy_with_logits(logits, survival, reduction="none")
            loss = (raw * survival_weights.view(1, -1)).mean()
        probs = torch.sigmoid(logits)
        return loss, probs, torch.ones((), dtype=logits.dtype, device=logits.device)

    if objective != "hazard":
        raise ValueError(f"unsupported objective {objective!r}")

    # CORN/discrete-time hazard objective:
    # q_k = P(H >= k | H >= k - 1, x).  Only train thresholds that are reachable:
    # all successes up to H and the first failure at H + 1.
    reachable = _conditional_survival_mask(survival)
    raw = F.binary_cross_entropy_with_logits(logits, survival, reduction="none")
    if survival_weights is not None:
        raw = raw * survival_weights.view(1, -1)
        reachable = reachable * survival_weights.view(1, -1)
    loss = (raw * reachable).sum() / reachable.sum().clamp_min(1.0)
    conditional_probs = torch.sigmoid(logits)
    survival_probs = torch.cumprod(conditional_probs, dim=1)
    return loss, survival_probs, conditional_probs.mean()


class HorizonPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        proj_dim: int,
        hidden_size: int,
        num_slots: int,
        architecture: str,
        num_layers: int,
        dropout: float,
        context_window: int,
        num_arms: int = 0,
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
            nn.Linear(hidden_size // 2, num_slots),
        )
        self.aux_arm_head = (
            nn.Sequential(
                nn.Linear(pooled_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_arms),
            )
            if num_arms > 0
            else None
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last_valid = (positions * (mask > 0.5)).max(dim=1).values.long()
        if self.architecture == "last_mlp":
            gather_idx = last_valid.view(-1, 1, 1).expand(-1, 1, features.shape[-1])
            return self.input_proj(features.float().gather(dim=1, index=gather_idx).squeeze(1))

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
        return encoded.gather(dim=1, index=gather_idx).squeeze(1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features, mask))

    def forward_with_aux(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = self.encode(features, mask)
        aux_logits = None if self.aux_arm_head is None else self.aux_arm_head(pooled)
        return self.head(pooled), aux_logits


def _monotonicize(probs: torch.Tensor) -> torch.Tensor:
    return torch.cummin(probs, dim=-1).values


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    pos = int(labels.sum().item())
    neg = int((~labels).sum().item())
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc.item())


def _policy_metrics(
    probs: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: tuple[int, ...],
    alpha: float,
) -> dict[str, float]:
    budgets = torch.tensor([arm - 1 for arm in arms], device=probs.device)
    expected = torch.stack([probs[:, : int(b.item())].sum(dim=1) for b in budgets], dim=1)
    expected_full = expected[:, -1:]
    ok = expected >= alpha * expected_full
    fallback = torch.full((ok.shape[0],), ok.shape[1] - 1, device=ok.device, dtype=torch.long)
    chosen = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)
    chosen_budget = budgets[chosen].float()
    true_accept = torch.minimum(accepted_len, chosen_budget)
    full_budget = float(arms[-1] - 1)
    full_accept = torch.minimum(accepted_len, torch.full_like(accepted_len, full_budget))
    return {
        f"alpha{alpha:.2f}_mean_block": (chosen_budget + 1).mean().item(),
        f"alpha{alpha:.2f}_mean_budget": chosen_budget.mean().item(),
        f"alpha{alpha:.2f}_mean_accepted": true_accept.mean().item(),
        f"alpha{alpha:.2f}_accept_retention": (true_accept.sum() / full_accept.sum().clamp_min(1)).item(),
        f"alpha{alpha:.2f}_accept_ratio": (true_accept / chosen_budget.clamp_min(1)).mean().item(),
    }


def _select_alpha_metrics(
    metrics: dict[str, float],
    *,
    alphas: tuple[float, ...],
    min_retention: float,
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for alpha in alphas:
        prefix = f"alpha{alpha:.2f}"
        candidates.append(
            {
                "alpha": float(alpha),
                "mean_block": float(metrics[f"{prefix}_mean_block"]),
                "mean_budget": float(metrics[f"{prefix}_mean_budget"]),
                "mean_accepted": float(metrics[f"{prefix}_mean_accepted"]),
                "accept_retention": float(metrics[f"{prefix}_accept_retention"]),
                "accept_ratio": float(metrics[f"{prefix}_accept_ratio"]),
            }
        )
    feasible = [item for item in candidates if item["accept_retention"] >= min_retention]
    if feasible:
        selected = max(feasible, key=lambda item: (item["accept_ratio"], item["mean_accepted"], -item["mean_block"]))
        feasible_flag = 1.0
    else:
        selected = max(candidates, key=lambda item: (item["accept_retention"], item["accept_ratio"]))
        feasible_flag = 0.0
    return {
        "selected_alpha": selected["alpha"],
        "selected_mean_block": selected["mean_block"],
        "selected_mean_budget": selected["mean_budget"],
        "selected_mean_accepted": selected["mean_accepted"],
        "selected_accept_retention": selected["accept_retention"],
        "selected_accept_ratio": selected["accept_ratio"],
        "selected_min_retention": float(min_retention),
        "selected_feasible": feasible_flag,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    objective: str = "survival_bce",
    monotonicize: bool,
    arms: tuple[int, ...],
    alphas: tuple[float, ...],
    selection_min_retention: float,
) -> dict[str, float]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    survival_all: list[torch.Tensor] = []
    accepted_all: list[torch.Tensor] = []
    losses: list[float] = []
    counts: list[int] = []
    for features, mask, survival, accepted_len in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        logits = model(features, mask)
        loss, probs_batch, _conditional_mean = _horizon_loss(
            logits,
            survival,
            objective=objective,
            survival_weights=None,
        )
        losses.append(loss.item())
        counts.append(features.shape[0])
        logits_all.append(probs_batch.detach().cpu())
        survival_all.append(survival.detach().cpu())
        accepted_all.append(accepted_len.detach().cpu())

    probs = torch.cat(logits_all)
    survival = torch.cat(survival_all)
    accepted_len = torch.cat(accepted_all)
    if monotonicize and objective == "survival_bce":
        probs = _monotonicize(probs)
    expected_len = probs.sum(dim=1)
    pred_len = (probs >= 0.5).sum(dim=1).float()
    rounded_len = expected_len.round().clamp(0, survival.shape[1])
    total = float(sum(counts))
    metrics = {
        "horizon_loss": float(sum(loss * count for loss, count in zip(losses, counts)) / total),
        "bce": float(sum(loss * count for loss, count in zip(losses, counts)) / total),
        "expected_len_mae": (expected_len - accepted_len).abs().mean().item(),
        "distance_mae": (expected_len - accepted_len).abs().mean().item(),
        "expected_len_rmse": torch.sqrt(torch.mean((expected_len - accepted_len) ** 2)).item(),
        "rounded_expected_len_mae": (rounded_len - accepted_len).abs().mean().item(),
        "rounded_expected_len_exact": (rounded_len == accepted_len).float().mean().item(),
        "threshold_len_mae": (pred_len - accepted_len).abs().mean().item(),
        "threshold_len_exact": (pred_len == accepted_len).float().mean().item(),
        "mean_expected_len": expected_len.mean().item(),
        "mean_pred_len": pred_len.mean().item(),
        "mean_target_len": accepted_len.mean().item(),
    }
    for k in (1, 4, 8, 12, 15):
        if k <= survival.shape[1]:
            metrics[f"auroc_h_ge_{k}"] = _roc_auc(probs[:, k - 1], survival[:, k - 1])
    for alpha in alphas:
        metrics.update(_policy_metrics(probs, accepted_len, arms=arms, alpha=alpha))
    metrics.update(
        _select_alpha_metrics(
            metrics,
            alphas=alphas,
            min_retention=selection_min_retention,
        )
    )
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    objective: str = "survival_bce",
    length_loss_weight: float,
    monotonic_weight: float,
    survival_weights: torch.Tensor | None,
    aux_arm_weight: float,
    arms: tuple[int, ...],
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_rows = 0
    for features, mask, survival, accepted_len in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        logits, aux_logits = model.forward_with_aux(features, mask)
        horizon_loss, probs, conditional_mean = _horizon_loss(
            logits,
            survival,
            objective=objective,
            survival_weights=survival_weights,
        )
        expected_len = probs.sum(dim=1)
        length_loss = F.smooth_l1_loss(expected_len, accepted_len)
        monotonic_loss = (
            torch.zeros((), dtype=logits.dtype, device=logits.device)
            if objective == "hazard"
            else F.relu(probs[:, 1:] - probs[:, :-1]).mean()
        )
        aux_arm_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        aux_arm_acc = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if aux_logits is not None and aux_arm_weight > 0:
            target_arm_idx = _oracle_arm_idx(accepted_len, arms=arms)
            aux_arm_loss = F.cross_entropy(aux_logits, target_arm_idx)
            aux_arm_acc = (aux_logits.argmax(dim=-1) == target_arm_idx).float().mean()
        loss = (
            horizon_loss
            + length_loss_weight * length_loss
            + monotonic_weight * monotonic_loss
            + aux_arm_weight * aux_arm_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch = features.shape[0]
        total_rows += batch
        for key, value in {
            "loss": loss,
            "horizon_loss": horizon_loss,
            "bce": horizon_loss,
            "length_loss": length_loss,
            "distance_loss": length_loss,
            "monotonic_loss": monotonic_loss,
            "aux_arm_loss": aux_arm_loss,
            "aux_arm_accuracy": aux_arm_acc,
            "expected_len_mae": (expected_len - accepted_len).abs().mean(),
            "conditional_mean": conditional_mean,
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch
    return {key: value / total_rows for key, value in totals.items()}


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a DFlashv2 horizon survival predictor.")
    parser.add_argument("--trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--architecture", choices=["last_mlp", "gru", "transformer"], default="gru")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--objective",
        choices=["survival_bce", "hazard"],
        default="survival_bce",
        help=(
            "survival_bce uses independent P(H>=k) logits; hazard uses conditional "
            "CORN/discrete-time survival logits whose cumulative product gives P(H>=k)."
        ),
    )
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--boundary-ks", default="4,8,12,15")
    parser.add_argument("--aux-arm-weight", type=float, default=0.0)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--compact-cache-dir", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monotonicize-eval", action="store_true")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default="0.85,0.90,0.95")
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["expected_len_mae", "selected_accept_ratio"],
        default="expected_len_mae",
    )
    args = parser.parse_args()
    if args.trace_dir is None and args.val_dir is None:
        parser.error("either --trace-dir or --val-dir is required")
    if not args.eval_only and args.trace_dir is None and args.train_dir is None:
        parser.error("--train-dir is required unless --eval-only is set when --trace-dir is not used")
    if args.eval_only and args.checkpoint is None:
        parser.error("--checkpoint is required with --eval-only")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_only = args.architecture == "last_mlp"
    if args.trace_dir is not None:
        cache_ready = (
            last_only
            and not args.eval_only
            and args.compact_cache_dir is not None
            and _compact_cache_ready(args.compact_cache_dir / "train")
            and _compact_cache_ready(args.compact_cache_dir / "val")
        )
        if cache_ready:
            print(
                json.dumps(
                    {
                        "event": "load_compact_cache",
                        "cache_dir": str(args.compact_cache_dir),
                    }
                ),
                flush=True,
            )
            train_ds = _load_compact_dataset(args.compact_cache_dir / "train")
            val_ds = _load_compact_dataset(args.compact_cache_dir / "val")
        else:
            print(
                json.dumps(
                    {
                        "event": "load_shards_start",
                        "trace_dirs": [str(path) for path in args.trace_dir],
                    }
                ),
                flush=True,
            )
            shards = _load_shards_multi(args.trace_dir)
            total_rows = sum(shard.rows for shard in shards)
            print(json.dumps({"event": "load_shards_done", "total_rows": total_rows}), flush=True)
        if args.eval_only:
            indices = np.arange(total_rows, dtype=np.int64)
            val_ds = HorizonIndexedDataset(shards, indices, last_only=last_only)
            train_ds = None
        elif not cache_ready:
            train_indices, val_indices = _make_splits(
                total_rows=total_rows,
                max_total_rows=args.max_total_rows,
                calibration_rows=args.calibration_rows,
                seed=args.seed,
            )
            if last_only:
                cache_meta = {
                    "trace_dirs": [str(path) for path in args.trace_dir],
                    "total_rows": int(total_rows),
                    "max_total_rows": args.max_total_rows,
                    "calibration_rows": int(args.calibration_rows),
                    "seed": int(args.seed),
                    "train_rows": int(train_indices.shape[0]),
                    "val_rows": int(val_indices.shape[0]),
                }
                if (
                    args.compact_cache_dir is not None
                    and _compact_cache_ready(args.compact_cache_dir / "train")
                    and _compact_cache_ready(args.compact_cache_dir / "val")
                ):
                    print(
                        json.dumps(
                            {
                                "event": "load_compact_cache",
                                "cache_dir": str(args.compact_cache_dir),
                            }
                        ),
                        flush=True,
                    )
                    train_ds = _load_compact_dataset(args.compact_cache_dir / "train")
                    val_ds = _load_compact_dataset(args.compact_cache_dir / "val")
                else:
                    print(
                        json.dumps(
                            {
                                "event": "materialize_last_features",
                                "train_rows": int(train_indices.shape[0]),
                                "val_rows": int(val_indices.shape[0]),
                            }
                        ),
                        flush=True,
                    )
                    train_ds = _materialize_last_feature_dataset(shards, train_indices)
                    val_ds = _materialize_last_feature_dataset(shards, val_indices)
                    if args.compact_cache_dir is not None:
                        print(
                            json.dumps(
                                {
                                    "event": "save_compact_cache",
                                    "cache_dir": str(args.compact_cache_dir),
                                }
                            ),
                            flush=True,
                        )
                        _save_compact_dataset(train_ds, args.compact_cache_dir / "train")
                        _save_compact_dataset(val_ds, args.compact_cache_dir / "val")
                        (args.compact_cache_dir / "meta.json").write_text(
                            json.dumps(cache_meta, indent=2, default=str) + "\n"
                        )
                    print(json.dumps({"event": "materialize_last_features_done"}), flush=True)
            else:
                train_ds = HorizonIndexedDataset(shards, train_indices, last_only=False)
                val_ds = HorizonIndexedDataset(shards, val_indices, last_only=False)
    else:
        assert args.val_dir is not None
        val_ds = HorizonTraceDataset(args.val_dir, max_rows=args.max_val_rows)
        train_ds = None if args.eval_only else HorizonTraceDataset(args.train_dir, max_rows=args.max_train_rows)
    reference_ds = train_ds or val_ds
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    boundary_ks = tuple(int(x) for x in args.boundary_ks.split(",") if x)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = HorizonPredictor(
        input_dim=reference_ds.hidden_size,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_slots=reference_ds.num_slots,
        architecture=args.architecture,
        num_layers=args.num_layers,
        dropout=args.dropout,
        context_window=reference_ds.context_window,
        num_arms=len(arms) if args.aux_arm_weight > 0 else 0,
    ).to(device)

    config = vars(args).copy()
    config.update(
        {
            "input_dim": reference_ds.hidden_size,
            "context_window": reference_ds.context_window,
            "num_slots": reference_ds.num_slots,
            "train_rows": len(train_ds) if train_ds is not None else None,
            "val_rows": len(val_ds),
            "last_only": last_only,
            "arms": arms,
            "alphas": alphas,
            "boundary_ks": boundary_ks,
        }
    )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_loader = _make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.eval_only:
        metrics = evaluate(
            model,
            val_loader,
            device=device,
            objective=args.objective,
            monotonicize=args.monotonicize_eval,
            arms=arms,
            alphas=alphas,
            selection_min_retention=args.selection_min_retention,
        )
        (args.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        return

    assert train_ds is not None
    train_loader = _make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    survival_weight_tensor = None
    if args.boundary_weight != 1.0:
        survival_weight_tensor = _survival_weights(
            reference_ds.num_slots,
            boundary_ks=boundary_ks,
            boundary_weight=args.boundary_weight,
        ).to(device)

    best = math.inf
    best_score = -math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            objective=args.objective,
            length_loss_weight=args.length_loss_weight,
            monotonic_weight=args.monotonic_weight,
            survival_weights=survival_weight_tensor,
            aux_arm_weight=args.aux_arm_weight,
            arms=arms,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            objective=args.objective,
            monotonicize=args.monotonicize_eval,
            arms=arms,
            alphas=alphas,
            selection_min_retention=args.selection_min_retention,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.checkpoint_selection == "expected_len_mae":
            is_best = val_metrics["expected_len_mae"] < best
            if is_best:
                best = val_metrics["expected_len_mae"]
                best_score = -best
        else:
            score = (
                val_metrics["selected_accept_ratio"]
                if val_metrics["selected_feasible"] > 0.5
                else -1.0 + val_metrics["selected_accept_retention"]
            )
            is_best = score > best_score
            if is_best:
                best_score = score
                best = val_metrics["expected_len_mae"]
        if is_best:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "checkpoint_selection": args.checkpoint_selection,
                    "checkpoint_score": best_score,
                },
                args.output_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "val_metrics": val_metrics,
        },
        args.output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
