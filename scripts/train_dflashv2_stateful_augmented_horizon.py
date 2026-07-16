from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dflashv2_horizon_predictor import (  # noqa: E402
    Shard,
    _horizon_loss,
    _policy_metrics,
    _roc_auc,
    _select_alpha_metrics,
)


DEFAULT_STATS_DIM = 8
SEQUENCE_CACHE_VERSION = 1


class AugmentedShard(Shard):
    prompt_index: np.ndarray
    cycle_id: np.ndarray
    predraft_stats: np.ndarray | None


def _load_augmented_shards(
    trace_dirs: list[Path],
    *,
    max_scan_rows: int | None,
) -> tuple[list[AugmentedShard], list[int]]:
    shards: list[AugmentedShard] = []
    trace_ids: list[int] = []
    loaded_rows = 0
    for trace_id, trace_dir in enumerate(trace_dirs):
        manifest_path = trace_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing trace manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["shards"]:
            rows = int(item["rows"])
            if rows <= 0:
                continue
            if max_scan_rows is not None and loaded_rows >= int(max_scan_rows):
                break
            shard_rows = rows
            if max_scan_rows is not None:
                shard_rows = min(shard_rows, int(max_scan_rows) - loaded_rows)
            shard_dir = trace_dir / item["path"]
            prompt_path = shard_dir / "prompt_index.npy"
            cycle_path = shard_dir / "cycle_id.npy"
            if not prompt_path.exists() or not cycle_path.exists():
                raise FileNotFoundError(
                    f"{shard_dir} is missing prompt_index.npy/cycle_id.npy; "
                    "stateful training requires request-cycle ordering"
                )
            aug = AugmentedShard(
                path=shard_dir,
                rows=shard_rows,
                features=np.load(shard_dir / "features.npy", mmap_mode="r"),
                mask=np.load(shard_dir / "mask.npy", mmap_mode="r"),
                survival=np.load(shard_dir / "survival.npy", mmap_mode="r"),
                accepted_len=np.load(shard_dir / "accepted_len.npy", mmap_mode="r"),
            )
            aug.prompt_index = np.load(prompt_path, mmap_mode="r")
            aug.cycle_id = np.load(cycle_path, mmap_mode="r")
            stats_path = shard_dir / "predraft_stats.npy"
            aug.predraft_stats = np.load(stats_path, mmap_mode="r") if stats_path.exists() else None
            shards.append(aug)
            trace_ids.append(trace_id)
            loaded_rows += shard_rows
        if max_scan_rows is not None and loaded_rows >= int(max_scan_rows):
            break
    if not shards:
        raise ValueError(f"no trace shards found in {trace_dirs}")
    return shards, trace_ids


def _build_sequences(
    shards: list[AugmentedShard],
    trace_ids: list[int],
    *,
    max_scan_rows: int | None,
    max_sequences: int | None,
    seed: int,
) -> list[np.ndarray]:
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    global_offset = 0
    scanned = 0
    for shard, trace_id in zip(shards, trace_ids):
        prompts = np.asarray(shard.prompt_index[: shard.rows])
        cycles = np.asarray(shard.cycle_id[: shard.rows])
        rows = shard.rows
        if max_scan_rows is not None:
            rows = min(rows, max(0, int(max_scan_rows) - scanned))
        for local_idx in range(rows):
            grouped[(trace_id, int(prompts[local_idx]))].append(
                (int(cycles[local_idx]), global_offset + local_idx)
            )
        scanned += rows
        global_offset += shard.rows
        if max_scan_rows is not None and scanned >= int(max_scan_rows):
            break
    sequences: list[np.ndarray] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        sequences.append(np.asarray([global_idx for _cycle, global_idx in rows], dtype=np.int64))
    rng = random.Random(seed)
    rng.shuffle(sequences)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    return sequences


def _split_sequences(
    sequences: list[np.ndarray],
    *,
    val_fraction: float,
    max_train_sequences: int | None,
    max_val_sequences: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    val_count = max(1, int(round(len(sequences) * val_fraction)))
    val_sequences = sequences[:val_count]
    train_sequences = sequences[val_count:]
    if max_train_sequences is not None:
        train_sequences = train_sequences[:max_train_sequences]
    if max_val_sequences is not None:
        val_sequences = val_sequences[:max_val_sequences]
    return train_sequences, val_sequences


class StatefulAugmentedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        shards: list[AugmentedShard],
        sequences: list[np.ndarray],
        *,
        stats_dim: int = DEFAULT_STATS_DIM,
    ) -> None:
        self.shards = shards
        self.sequences = sequences
        self.stats_dim = int(stats_dim)
        self.offsets: list[int] = []
        total = 0
        for shard in shards:
            self.offsets.append(total)
            total += shard.rows
        first = shards[0]
        self.hidden_size = int(first.features.shape[2])
        self.num_slots = int(first.survival.shape[1])
        self.context_window = 1

    def __len__(self) -> int:
        return len(self.sequences)

    def _row(self, global_idx: int, prev_accepted: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_idx = global_idx - self.offsets[shard_idx]
        shard = self.shards[shard_idx]
        raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
        valid = np.flatnonzero(raw_mask > 0.5)
        feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
        fused = np.asarray(shard.features[local_idx, feature_idx], dtype=np.float16)
        if shard.predraft_stats is not None:
            stats = np.asarray(shard.predraft_stats[local_idx], dtype=np.float32)
        else:
            stats = np.zeros((self.stats_dim,), dtype=np.float32)
            if prev_accepted is not None:
                norm = float(prev_accepted) / float(max(self.num_slots, 1))
                stats[4:8] = np.asarray([norm, norm, 1.0, 1.0], dtype=np.float32)
        survival = np.asarray(shard.survival[local_idx], dtype=np.float32)
        accepted = float(shard.accepted_len[local_idx])
        return fused, stats.astype(np.float32), survival, accepted

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        features = np.empty((len(seq), self.hidden_size), dtype=np.float16)
        stats = np.empty((len(seq), self.stats_dim), dtype=np.float32)
        survival = np.empty((len(seq), self.num_slots), dtype=np.float32)
        accepted = np.empty((len(seq),), dtype=np.float32)
        prev_accepted: float | None = None
        for out_idx, global_idx in enumerate(seq):
            fused, row_stats, row_survival, row_accepted = self._row(int(global_idx), prev_accepted)
            features[out_idx] = fused
            stats[out_idx] = row_stats
            survival[out_idx] = row_survival
            accepted[out_idx] = row_accepted
            prev_accepted = row_accepted
        return {
            "features": torch.from_numpy(features),
            "stats": torch.from_numpy(stats),
            "survival": torch.from_numpy(survival),
            "accepted_len": torch.from_numpy(accepted),
            "mask": torch.ones((len(seq),), dtype=torch.float32),
        }


class CachedStatefulDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache_dir: Path, sequence_indices: list[int] | None = None) -> None:
        self.cache_dir = cache_dir
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing sequence cache metadata: {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        self.features = np.load(cache_dir / "features.npy", mmap_mode="r")
        self.stats = np.load(cache_dir / "stats.npy", mmap_mode="r")
        self.survival = np.load(cache_dir / "survival.npy", mmap_mode="r")
        self.accepted_len = np.load(cache_dir / "accepted_len.npy", mmap_mode="r")
        self.seq_offsets = np.load(cache_dir / "seq_offsets.npy", mmap_mode="r")
        self.sequence_indices = (
            list(range(int(self.meta["num_sequences"])))
            if sequence_indices is None
            else list(sequence_indices)
        )
        self.hidden_size = int(self.features.shape[1])
        self.stats_dim = int(self.stats.shape[1])
        self.num_slots = int(self.survival.shape[1])
        self.context_window = 1

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx = int(self.sequence_indices[idx])
        start = int(self.seq_offsets[seq_idx])
        end = int(self.seq_offsets[seq_idx + 1])
        return {
            "features": torch.from_numpy(np.asarray(self.features[start:end]).copy()),
            "stats": torch.from_numpy(np.asarray(self.stats[start:end]).copy()),
            "survival": torch.from_numpy(np.asarray(self.survival[start:end]).copy()),
            "accepted_len": torch.from_numpy(np.asarray(self.accepted_len[start:end]).copy()),
            "mask": torch.ones((end - start,), dtype=torch.float32),
        }


def _sequence_cache_complete(cache_dir: Path) -> bool:
    if not (cache_dir / "meta.json").exists():
        return False
    required = ["features.npy", "stats.npy", "survival.npy", "accepted_len.npy", "seq_offsets.npy"]
    return all((cache_dir / name).exists() for name in required)


def _materialize_sequence_cache(
    cache_dir: Path,
    dataset: StatefulAugmentedDataset,
    *,
    trace_dirs: list[Path],
    max_scan_rows: int | None,
    seed: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not dataset.sequences:
        raise ValueError("cannot materialize empty sequence cache")
    total_rows = int(sum(len(seq) for seq in dataset.sequences))
    total_global_rows = int(sum(shard.rows for shard in dataset.shards))
    seq_offsets_arr = np.empty((len(dataset.sequences) + 1,), dtype=np.int64)
    seq_offsets_arr[0] = 0
    for idx, seq in enumerate(dataset.sequences, start=1):
        seq_offsets_arr[idx] = seq_offsets_arr[idx - 1] + len(seq)

    accepted_by_global = np.empty((total_global_rows,), dtype=np.float32)
    global_offset = 0
    for shard in dataset.shards:
        accepted_by_global[global_offset : global_offset + shard.rows] = np.asarray(
            shard.accepted_len[: shard.rows],
            dtype=np.float32,
        )
        global_offset += shard.rows

    global_to_write = np.full((total_global_rows,), -1, dtype=np.int64)
    previous_accepted = np.full((total_global_rows,), np.nan, dtype=np.float32)
    write_idx = 0
    for seq in dataset.sequences:
        prev_global_idx: int | None = None
        for global_idx_raw in seq:
            global_idx = int(global_idx_raw)
            global_to_write[global_idx] = write_idx
            if prev_global_idx is not None:
                previous_accepted[global_idx] = accepted_by_global[prev_global_idx]
            prev_global_idx = global_idx
            write_idx += 1

    features = np.lib.format.open_memmap(
        cache_dir / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, dataset.hidden_size),
    )
    stats = np.lib.format.open_memmap(
        cache_dir / "stats.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, dataset.stats_dim),
    )
    survival = np.lib.format.open_memmap(
        cache_dir / "survival.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, dataset.num_slots),
    )
    accepted_len = np.lib.format.open_memmap(
        cache_dir / "accepted_len.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows,),
    )

    global_offset = 0
    next_log = 50_000
    written = 0
    for shard in dataset.shards:
        for local_idx in range(shard.rows):
            global_idx = global_offset + local_idx
            out_idx = int(global_to_write[global_idx])
            if out_idx < 0:
                continue
            raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
            valid = np.flatnonzero(raw_mask > 0.5)
            feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
            features[out_idx] = np.asarray(shard.features[local_idx, feature_idx], dtype=np.float16)
            if shard.predraft_stats is not None:
                stats[out_idx] = np.asarray(shard.predraft_stats[local_idx], dtype=np.float32)
            else:
                row_stats = np.zeros((dataset.stats_dim,), dtype=np.float32)
                prev_accepted = float(previous_accepted[global_idx])
                if not math.isnan(prev_accepted):
                    norm = prev_accepted / float(max(dataset.num_slots, 1))
                    row_stats[4:8] = np.asarray([norm, norm, 1.0, 1.0], dtype=np.float32)
                stats[out_idx] = row_stats
            survival[out_idx] = np.asarray(shard.survival[local_idx], dtype=np.float32)
            accepted_len[out_idx] = accepted_by_global[global_idx]
            written += 1
            if written >= next_log:
                print(
                    json.dumps(
                        {
                            "event": "sequence_cache_write_progress",
                            "rows": written,
                            "total_rows": total_rows,
                        }
                    ),
                    flush=True,
                )
                next_log += 50_000
        global_offset += shard.rows
        features.flush()
        stats.flush()
        survival.flush()
        accepted_len.flush()

    if written != total_rows:
        raise RuntimeError(f"sequence cache wrote {written} rows, expected {total_rows}")

    features.flush()
    stats.flush()
    survival.flush()
    accepted_len.flush()
    np.save(cache_dir / "seq_offsets.npy", seq_offsets_arr)
    meta = {
        "version": SEQUENCE_CACHE_VERSION,
        "trace_dirs": [str(path) for path in trace_dirs],
        "max_scan_rows": max_scan_rows,
        "seed": seed,
        "num_sequences": len(dataset.sequences),
        "num_rows": total_rows,
        "hidden_size": dataset.hidden_size,
        "stats_dim": dataset.stats_dim,
        "num_slots": dataset.num_slots,
        "dtype": {
            "features": "float16",
            "stats": "float32",
            "survival": "float32",
            "accepted_len": "float32",
        },
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "sequence_cache_write_done", **meta}), flush=True)


def _split_sequence_indices(
    num_sequences: int,
    *,
    val_fraction: float,
    max_train_sequences: int | None,
    max_val_sequences: int | None,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    val_count = max(1, int(round(num_sequences * val_fraction)))
    val_indices = list(range(val_count))
    train_indices = list(range(val_count, num_sequences))
    if max_train_sequences is not None:
        train_indices = train_indices[:max_train_sequences]
    if max_val_sequences is not None:
        val_indices = val_indices[:max_val_sequences]
    return train_indices, val_indices


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["features"].shape[0]) for item in batch)
    hidden_size = int(batch[0]["features"].shape[1])
    stats_dim = int(batch[0]["stats"].shape[1])
    num_slots = int(batch[0]["survival"].shape[1])
    features = torch.zeros((len(batch), max_len, hidden_size), dtype=torch.float16)
    stats = torch.zeros((len(batch), max_len, stats_dim), dtype=torch.float32)
    survival = torch.zeros((len(batch), max_len, num_slots), dtype=torch.float32)
    accepted = torch.zeros((len(batch), max_len), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for idx, item in enumerate(batch):
        length = int(item["features"].shape[0])
        features[idx, :length] = item["features"]
        stats[idx, :length] = item["stats"]
        survival[idx, :length] = item["survival"]
        accepted[idx, :length] = item["accepted_len"]
        mask[idx, :length] = 1.0
    return {
        "features": features,
        "stats": stats,
        "survival": survival,
        "accepted_len": accepted,
        "mask": mask,
    }


class StatefulAugmentedHorizonPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        stats_dim: int,
        proj_dim: int,
        stats_proj_dim: int,
        hidden_size: int,
        num_layers: int,
        num_slots: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fused_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, stats_proj_dim),
            nn.GELU(),
            nn.LayerNorm(stats_proj_dim),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=proj_dim + stats_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_slots),
        )

    def forward(self, features: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.fused_proj(features.float()), self.stats_proj(stats.float())], dim=-1)
        x = x * mask.unsqueeze(-1)
        encoded, _state = self.encoder(x)
        return self.head(encoded)


def _flatten_valid(
    logits: torch.Tensor,
    survival: torch.Tensor,
    accepted_len: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask > 0.5
    return logits[valid], survival[valid], accepted_len[valid]


def _monotonicize(probs: torch.Tensor) -> torch.Tensor:
    return torch.cummin(probs, dim=-1).values


def _arm_utility_loss(
    probs: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: tuple[int, ...],
    utility_lambda: float,
) -> torch.Tensor:
    budgets = torch.tensor([arm - 1 for arm in arms], dtype=probs.dtype, device=probs.device)
    expected = torch.stack([probs[:, : int(b.item())].sum(dim=1) for b in budgets], dim=1)
    predicted_utility = expected - utility_lambda * budgets.view(1, -1)
    true_accept = torch.minimum(accepted_len.view(-1, 1), budgets.view(1, -1))
    true_utility = true_accept - utility_lambda * budgets.view(1, -1)
    target = true_utility.argmax(dim=1)
    return F.cross_entropy(predicted_utility, target)


def train_epoch(
    model: StatefulAugmentedHorizonPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    objective: str,
    length_loss_weight: float,
    arm_utility_weight: float,
    arm_utility_lambda: float,
    monotonic_weight: float,
    arms: tuple[int, ...],
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_rows = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        stats = batch["stats"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        survival = batch["survival"].to(device, non_blocking=True)
        accepted_len = batch["accepted_len"].to(device, non_blocking=True)
        logits = model(features, stats, mask)
        logits_flat, survival_flat, accepted_flat = _flatten_valid(logits, survival, accepted_len, mask)
        horizon_loss, probs, _conditional_mean = _horizon_loss(
            logits_flat,
            survival_flat,
            objective=objective,
            survival_weights=None,
        )
        expected_len = probs.sum(dim=1)
        length_loss = F.smooth_l1_loss(expected_len, accepted_flat)
        monotonic_loss = (
            torch.zeros((), dtype=logits.dtype, device=logits.device)
            if objective == "hazard"
            else F.relu(probs[:, 1:] - probs[:, :-1]).mean()
        )
        arm_loss = _arm_utility_loss(
            probs,
            accepted_flat,
            arms=arms,
            utility_lambda=arm_utility_lambda,
        )
        loss = (
            horizon_loss
            + length_loss_weight * length_loss
            + monotonic_weight * monotonic_loss
            + arm_utility_weight * arm_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows = int(accepted_flat.shape[0])
        total_rows += rows
        for key, value in {
            "loss": loss,
            "horizon_loss": horizon_loss,
            "length_loss": length_loss,
            "arm_utility_loss": arm_loss,
            "monotonic_loss": monotonic_loss,
            "expected_len_mae": (expected_len - accepted_flat).abs().mean(),
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * rows
    return {key: value / max(total_rows, 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: StatefulAugmentedHorizonPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    objective: str,
    monotonicize: bool,
    arms: tuple[int, ...],
    alphas: tuple[float, ...],
    selection_min_retention: float,
) -> dict[str, float]:
    model.eval()
    probs_all: list[torch.Tensor] = []
    survival_all: list[torch.Tensor] = []
    accepted_all: list[torch.Tensor] = []
    losses: list[float] = []
    counts: list[int] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        stats = batch["stats"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        survival = batch["survival"].to(device, non_blocking=True)
        accepted_len = batch["accepted_len"].to(device, non_blocking=True)
        logits = model(features, stats, mask)
        logits_flat, survival_flat, accepted_flat = _flatten_valid(logits, survival, accepted_len, mask)
        loss, probs, _conditional_mean = _horizon_loss(
            logits_flat,
            survival_flat,
            objective=objective,
            survival_weights=None,
        )
        if monotonicize and objective == "survival_bce":
            probs = _monotonicize(probs)
        probs_all.append(probs.cpu())
        survival_all.append(survival_flat.cpu())
        accepted_all.append(accepted_flat.cpu())
        losses.append(float(loss.item()))
        counts.append(int(accepted_flat.shape[0]))

    probs = torch.cat(probs_all)
    survival = torch.cat(survival_all)
    accepted_len = torch.cat(accepted_all)
    expected_len = probs.sum(dim=1)
    pred_len = (probs >= 0.5).sum(dim=1).float()
    rounded_len = expected_len.round().clamp(0, survival.shape[1])
    total = float(sum(counts))
    metrics = {
        "horizon_loss": float(sum(loss * count for loss, count in zip(losses, counts)) / total),
        "expected_len_mae": (expected_len - accepted_len).abs().mean().item(),
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
        _select_alpha_metrics(metrics, alphas=alphas, min_retention=selection_min_retention)
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stateful DFlashv2 horizon predictor with fused context plus confidence stats."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--stats-proj-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--objective", choices=["survival_bce", "hazard"], default="hazard")
    parser.add_argument("--length-loss-weight", type=float, default=0.10)
    parser.add_argument("--arm-utility-weight", type=float, default=0.10)
    parser.add_argument("--arm-utility-lambda", type=float, default=0.02)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-scan-rows", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument(
        "--sequence-cache-dir",
        type=Path,
        default=None,
        help="Optional contiguous cache for stateful sequence training. Built if missing, reused if present.",
    )
    parser.add_argument("--rebuild-sequence-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default="0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.98")
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument("--monotonicize-eval", action="store_true")
    parser.add_argument(
        "--checkpoint-selection",
        choices=["expected_len_mae", "selected_accept_ratio"],
        default="selected_accept_ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_cache_dir is not None and args.rebuild_sequence_cache:
        for path in (
            "features.npy",
            "stats.npy",
            "survival.npy",
            "accepted_len.npy",
            "seq_offsets.npy",
            "meta.json",
        ):
            target = args.sequence_cache_dir / path
            if target.exists():
                target.unlink()

    if args.sequence_cache_dir is not None and _sequence_cache_complete(args.sequence_cache_dir):
        print(
            json.dumps(
                {
                    "event": "sequence_cache_reuse",
                    "cache_dir": str(args.sequence_cache_dir),
                }
            ),
            flush=True,
        )
        if args.build_cache_only:
            return
        full_cache_ds = CachedStatefulDataset(args.sequence_cache_dir)
        train_indices, val_indices = _split_sequence_indices(
            len(full_cache_ds),
            val_fraction=args.val_fraction,
            max_train_sequences=args.max_train_sequences,
            max_val_sequences=args.max_val_sequences,
        )
        train_ds = CachedStatefulDataset(args.sequence_cache_dir, train_indices)
        val_ds = CachedStatefulDataset(args.sequence_cache_dir, val_indices)
        train_rows = int(sum(int(full_cache_ds.seq_offsets[i + 1] - full_cache_ds.seq_offsets[i]) for i in train_indices))
        val_rows = int(sum(int(full_cache_ds.seq_offsets[i + 1] - full_cache_ds.seq_offsets[i]) for i in val_indices))
    else:
        print(json.dumps({"event": "load_augmented_shards_start", "trace_dirs": [str(p) for p in args.trace_dir]}), flush=True)
        shards, trace_ids = _load_augmented_shards(
            args.trace_dir,
            max_scan_rows=args.max_scan_rows,
        )
        print(json.dumps({"event": "load_augmented_shards_done", "shards": len(shards), "rows": sum(s.rows for s in shards)}), flush=True)
        print(json.dumps({"event": "build_sequences_start", "max_scan_rows": args.max_scan_rows}), flush=True)
        sequences = _build_sequences(
            shards,
            trace_ids,
            max_scan_rows=args.max_scan_rows,
            max_sequences=args.max_sequences,
            seed=args.seed,
        )
        print(json.dumps({"event": "build_sequences_done", "sequences": len(sequences), "rows": int(sum(len(s) for s in sequences))}), flush=True)
        if args.sequence_cache_dir is not None:
            cache_ds = StatefulAugmentedDataset(shards, sequences)
            print(
                json.dumps(
                    {
                        "event": "sequence_cache_write_start",
                        "cache_dir": str(args.sequence_cache_dir),
                        "rows": int(sum(len(seq) for seq in sequences)),
                        "sequences": len(sequences),
                    }
                ),
                flush=True,
            )
            _materialize_sequence_cache(
                args.sequence_cache_dir,
                cache_ds,
                trace_dirs=args.trace_dir,
                max_scan_rows=args.max_scan_rows,
                seed=args.seed,
            )
            if args.build_cache_only:
                return
            full_cache_ds = CachedStatefulDataset(args.sequence_cache_dir)
            train_indices, val_indices = _split_sequence_indices(
                len(full_cache_ds),
                val_fraction=args.val_fraction,
                max_train_sequences=args.max_train_sequences,
                max_val_sequences=args.max_val_sequences,
            )
            train_ds = CachedStatefulDataset(args.sequence_cache_dir, train_indices)
            val_ds = CachedStatefulDataset(args.sequence_cache_dir, val_indices)
            train_rows = int(sum(int(full_cache_ds.seq_offsets[i + 1] - full_cache_ds.seq_offsets[i]) for i in train_indices))
            val_rows = int(sum(int(full_cache_ds.seq_offsets[i + 1] - full_cache_ds.seq_offsets[i]) for i in val_indices))
        else:
            train_sequences, val_sequences = _split_sequences(
                sequences,
                val_fraction=args.val_fraction,
                max_train_sequences=args.max_train_sequences,
                max_val_sequences=args.max_val_sequences,
            )
            train_ds = StatefulAugmentedDataset(shards, train_sequences)
            val_ds = StatefulAugmentedDataset(shards, val_sequences)
            train_rows = int(sum(len(seq) for seq in train_sequences))
            val_rows = int(sum(len(seq) for seq in val_sequences))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    model = StatefulAugmentedHorizonPredictor(
        input_dim=train_ds.hidden_size,
        stats_dim=DEFAULT_STATS_DIM,
        proj_dim=args.proj_dim,
        stats_proj_dim=args.stats_proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_slots=train_ds.num_slots,
        dropout=args.dropout,
    ).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    config = vars(args).copy()
    config.update(
        {
            "input_dim": train_ds.hidden_size,
            "stats_dim": DEFAULT_STATS_DIM,
            "num_slots": train_ds.num_slots,
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "arms": arms,
            "alphas": alphas,
            "stats_columns": [
                "verifier_entropy_latest",
                "verifier_token_prob_latest",
                "verifier_top1_top2_margin_latest",
                "verifier_token_logprob_latest",
                "prev_accepted_len_norm",
                "prev_accept_ratio",
                "prev_budget_norm",
                "has_prev_cycle",
            ],
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mae = math.inf
    best_score = -math.inf
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            objective=args.objective,
            length_loss_weight=args.length_loss_weight,
            arm_utility_weight=args.arm_utility_weight,
            arm_utility_lambda=args.arm_utility_lambda,
            monotonic_weight=args.monotonic_weight,
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
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.checkpoint_selection == "expected_len_mae":
            score = -float(val_metrics["expected_len_mae"])
        else:
            score = (
                float(val_metrics["selected_accept_ratio"])
                if float(val_metrics["selected_feasible"]) > 0.5
                else -1.0 + float(val_metrics["selected_accept_retention"])
            )
        is_best = score > best_score
        if is_best:
            best_score = score
            best_mae = float(val_metrics["expected_len_mae"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "checkpoint_selection": args.checkpoint_selection,
                    "checkpoint_score": best_score,
                    "best_mae": best_mae,
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
