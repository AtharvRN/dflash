from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_dflashv2_horizon_predictor import (
    _load_shards,
    _monotonicize,
    _oracle_arm_idx,
    _policy_metrics,
    _select_alpha_metrics,
    _survival_weights,
)


@dataclass
class SequenceIndex:
    sources: np.ndarray
    rows: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


class StatefulHorizonPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        proj_dim: int,
        hidden_size: int,
        num_slots: int,
        num_layers: int,
        dropout: float,
        num_arms: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=proj_dim,
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
        self.aux_arm_head = (
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_arms),
            )
            if num_arms > 0
            else None
        )

    def forward_with_aux(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.input_proj(features.float())
        out, _ = self.gru(x)
        logits = self.head(out)
        aux_logits = None if self.aux_arm_head is None else self.aux_arm_head(out)
        return logits, aux_logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(features)[0]


def _parse_ints(value: str) -> tuple[int, ...]:
    out = tuple(int(x) for x in value.split(",") if x)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _parse_floats(value: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in value.split(",") if x)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def _load_compact_sources(cache_dir: Path) -> tuple[list[dict[str, np.ndarray]], int, int]:
    sources: list[dict[str, np.ndarray]] = []
    for split in ("train", "val"):
        split_dir = cache_dir / split
        required = (
            "features.npy",
            "survival.npy",
            "accepted_len.npy",
            "sequence_id.npy",
            "cycle_id.npy",
        )
        missing = [name for name in required if not (split_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing files in {split_dir}: {missing}")
        sources.append(
            {
                "features": np.load(split_dir / "features.npy", mmap_mode="r"),
                "survival": np.load(split_dir / "survival.npy", mmap_mode="r"),
                "accepted_len": np.load(split_dir / "accepted_len.npy", mmap_mode="r"),
                "sequence_id": np.load(split_dir / "sequence_id.npy", mmap_mode="r"),
                "cycle_id": np.load(split_dir / "cycle_id.npy", mmap_mode="r"),
            }
        )
    input_dim = int(sources[0]["features"].shape[-1])
    num_slots = int(sources[0]["survival"].shape[-1])
    return sources, input_dim, num_slots


def _build_sequence_index(
    sources: list[dict[str, np.ndarray]],
    *,
    seed: int,
    val_fraction: float,
    max_train_sequences: int | None,
    max_val_sequences: int | None,
) -> tuple[SequenceIndex, SequenceIndex]:
    source_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    seq_parts: list[np.ndarray] = []
    cycle_parts: list[np.ndarray] = []
    for src_idx, source in enumerate(sources):
        rows = int(source["accepted_len"].shape[0])
        source_parts.append(np.full((rows,), src_idx, dtype=np.int8))
        row_parts.append(np.arange(rows, dtype=np.int64))
        seq_parts.append(np.asarray(source["sequence_id"], dtype=np.int64))
        cycle_parts.append(np.asarray(source["cycle_id"], dtype=np.int32))
    all_sources = np.concatenate(source_parts)
    all_rows = np.concatenate(row_parts)
    all_seq = np.concatenate(seq_parts)
    all_cycle = np.concatenate(cycle_parts)
    order = np.lexsort((all_cycle, all_seq))
    all_sources = all_sources[order]
    all_rows = all_rows[order]
    all_seq = all_seq[order]

    boundary = np.flatnonzero(all_seq[1:] != all_seq[:-1]) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), boundary.astype(np.int64)])
    ends = np.concatenate([boundary.astype(np.int64), np.asarray([all_seq.shape[0]], dtype=np.int64)])
    sequence_ids = all_seq[starts]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(sequence_ids.shape[0])
    val_count = max(1, int(round(sequence_ids.shape[0] * val_fraction)))
    val_seq_idx = np.sort(perm[:val_count])
    train_seq_idx = np.sort(perm[val_count:])
    if max_train_sequences is not None:
        train_seq_idx = np.sort(train_seq_idx[:max_train_sequences])
    if max_val_sequences is not None:
        val_seq_idx = np.sort(val_seq_idx[:max_val_sequences])

    def make(seq_idx: np.ndarray) -> SequenceIndex:
        return SequenceIndex(
            sources=all_sources,
            rows=all_rows,
            starts=starts[seq_idx],
            ends=ends[seq_idx],
        )

    return make(train_seq_idx), make(val_seq_idx)


class CompactSequenceDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, sources: list[dict[str, np.ndarray]], index: SequenceIndex, *, max_seq_len: int | None) -> None:
        self.sources_data = sources
        self.index = index
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return int(self.index.starts.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start = int(self.index.starts[idx])
        end = int(self.index.ends[idx])
        if self.max_seq_len is not None and end - start > self.max_seq_len:
            end = start + self.max_seq_len
        src = self.index.sources[start:end]
        row = self.index.rows[start:end]
        features = np.stack([self.sources_data[int(s)]["features"][int(r), 0] for s, r in zip(src, row)])
        survival = np.stack([self.sources_data[int(s)]["survival"][int(r)] for s, r in zip(src, row)]).astype(np.float32)
        accepted = np.asarray([self.sources_data[int(s)]["accepted_len"][int(r)] for s, r in zip(src, row)], dtype=np.float32)
        return {"features": features, "survival": survival, "accepted_len": accepted}


def _last_feature_from_row(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.flatnonzero(mask > 0.5)
    idx = int(valid[-1]) if valid.size else int(mask.shape[0] - 1)
    return np.asarray(features[idx], dtype=np.float16)


class OriginalTraceSequenceDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, trace_dirs: list[Path], *, max_seq_len: int | None = None) -> None:
        records: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
        for trace_idx, trace_dir in enumerate(trace_dirs):
            shards, _ = _load_shards(trace_dir)
            for shard in shards:
                prompt = np.load(shard.path / "prompt_index.npy", mmap_mode="r")
                cycle = np.load(shard.path / "cycle_id.npy", mmap_mode="r")
                for row in range(int(shard.rows)):
                    seq = int(trace_idx) * 1_000_000_000 + int(prompt[row])
                    feat = _last_feature_from_row(shard.features[row], shard.mask[row])
                    records.append(
                        (
                            seq,
                            int(cycle[row]),
                            feat,
                            np.asarray(shard.survival[row], dtype=np.float32),
                            float(shard.accepted_len[row]),
                        )
                    )
        records.sort(key=lambda item: (item[0], item[1]))
        self.sequences: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
        current_seq: int | None = None
        current: list[tuple[np.ndarray, np.ndarray, float]] = []
        for seq, _, feat, survival, accepted in records:
            if current_seq is None or seq != current_seq:
                if current:
                    self.sequences.append(current if max_seq_len is None else current[:max_seq_len])
                current_seq = seq
                current = []
            current.append((feat, survival, accepted))
        if current:
            self.sequences.append(current if max_seq_len is None else current[:max_seq_len])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        seq = self.sequences[idx]
        return {
            "features": np.stack([item[0] for item in seq]),
            "survival": np.stack([item[1] for item in seq]).astype(np.float32),
            "accepted_len": np.asarray([item[2] for item in seq], dtype=np.float32),
        }


def _collate_sequences(items: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(items)
    max_len = max(int(item["features"].shape[0]) for item in items)
    input_dim = int(items[0]["features"].shape[-1])
    num_slots = int(items[0]["survival"].shape[-1])
    features = np.zeros((batch, max_len, input_dim), dtype=np.float16)
    survival = np.zeros((batch, max_len, num_slots), dtype=np.float32)
    accepted = np.zeros((batch, max_len), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=np.float32)
    for i, item in enumerate(items):
        length = int(item["features"].shape[0])
        features[i, :length] = item["features"]
        survival[i, :length] = item["survival"]
        accepted[i, :length] = item["accepted_len"]
        mask[i, :length] = 1.0
    return (
        torch.from_numpy(features),
        torch.from_numpy(survival),
        torch.from_numpy(accepted),
        torch.from_numpy(mask),
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def train_epoch(
    model: StatefulHorizonPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    length_loss_weight: float,
    monotonic_weight: float,
    survival_weights: torch.Tensor | None,
    aux_arm_weight: float,
    arms: tuple[int, ...],
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_tokens = 0.0
    for features, survival, accepted, mask in loader:
        features = features.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        accepted = accepted.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, aux_logits = model.forward_with_aux(features)
        bce_raw = F.binary_cross_entropy_with_logits(logits, survival, reduction="none")
        if survival_weights is not None:
            bce_raw = bce_raw * survival_weights.view(1, 1, -1)
        bce = (bce_raw * mask.unsqueeze(-1)).sum() / (mask.sum().clamp_min(1.0) * survival.shape[-1])
        probs = torch.sigmoid(logits)
        expected_len = probs.sum(dim=-1)
        length_loss = _masked_mean(F.smooth_l1_loss(expected_len, accepted, reduction="none"), mask)
        monotonic_loss = ((F.relu(probs[:, :, 1:] - probs[:, :, :-1]).mean(dim=-1)) * mask).sum() / mask.sum().clamp_min(1.0)
        aux_arm_loss = torch.zeros((), dtype=logits.dtype, device=device)
        aux_arm_acc = torch.zeros((), dtype=logits.dtype, device=device)
        if aux_logits is not None and aux_arm_weight > 0:
            target = _oracle_arm_idx(accepted.reshape(-1), arms=arms).view_as(accepted).long()
            ce = F.cross_entropy(aux_logits.reshape(-1, aux_logits.shape[-1]), target.reshape(-1), reduction="none").view_as(mask)
            aux_arm_loss = _masked_mean(ce, mask)
            aux_arm_acc = _masked_mean((aux_logits.argmax(dim=-1) == target).float(), mask)
        loss = bce + length_loss_weight * length_loss + monotonic_weight * monotonic_loss + aux_arm_weight * aux_arm_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count = float(mask.sum().item())
        total_tokens += count
        for key, value in {
            "loss": loss,
            "bce": bce,
            "length_loss": length_loss,
            "monotonic_loss": monotonic_loss,
            "aux_arm_loss": aux_arm_loss,
            "aux_arm_accuracy": aux_arm_acc,
            "expected_len_mae": _masked_mean((expected_len - accepted).abs(), mask),
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * count
    return {key: value / total_tokens for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: StatefulHorizonPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    monotonicize: bool,
    arms: tuple[int, ...],
    alphas: tuple[float, ...],
    selection_min_retention: float,
) -> dict[str, float]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    survival_all: list[torch.Tensor] = []
    accepted_all: list[torch.Tensor] = []
    for features, survival, accepted, mask in loader:
        features = features.to(device, non_blocking=True)
        logits = model(features).cpu()
        keep = mask.bool().reshape(-1)
        logits_all.append(logits.reshape(-1, logits.shape[-1])[keep])
        survival_all.append(survival.reshape(-1, survival.shape[-1])[keep])
        accepted_all.append(accepted.reshape(-1)[keep])
    logits = torch.cat(logits_all)
    survival = torch.cat(survival_all).float()
    accepted = torch.cat(accepted_all).float()
    probs = torch.sigmoid(logits)
    if monotonicize:
        probs = _monotonicize(probs)
    expected_len = probs.sum(dim=1)
    pred_len = (probs >= 0.5).sum(dim=1).float()
    bce = F.binary_cross_entropy_with_logits(logits, survival).item()
    metrics = {
        "bce": bce,
        "expected_len_mae": (expected_len - accepted).abs().mean().item(),
        "threshold_len_mae": (pred_len - accepted).abs().mean().item(),
        "threshold_len_exact": (pred_len == accepted).float().mean().item(),
        "mean_expected_len": expected_len.mean().item(),
        "mean_pred_len": pred_len.mean().item(),
        "mean_target_len": accepted.mean().item(),
    }
    for alpha in alphas:
        metrics.update(_policy_metrics(probs, accepted, arms=arms, alpha=alpha))
    metrics.update(_select_alpha_metrics(metrics, alphas=alphas, min_retention=selection_min_retention))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stateful GRUCell-style DFlashv2 horizon predictor.")
    parser.add_argument("--compact-cache-dir", type=Path, required=True)
    parser.add_argument("--math500-trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--boundary-ks", type=_parse_ints, default=(4, 8, 12, 15))
    parser.add_argument("--aux-arm-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--arms", type=_parse_ints, default=(4, 8, 12, 16))
    parser.add_argument("--alphas", type=_parse_floats, default=(0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98))
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monotonicize-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    sources, input_dim, num_slots = _load_compact_sources(args.compact_cache_dir)
    train_index, val_index = _build_sequence_index(
        sources,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_sequences=args.max_train_sequences,
        max_val_sequences=args.max_val_sequences,
    )
    train_ds = CompactSequenceDataset(sources, train_index, max_seq_len=args.max_seq_len)
    val_ds = CompactSequenceDataset(sources, val_index, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_sequences,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_sequences,
        pin_memory=device.type == "cuda",
    )
    model = StatefulHorizonPredictor(
        input_dim=input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_slots=num_slots,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_arms=len(args.arms) if args.aux_arm_weight > 0 else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = _survival_weights(num_slots, boundary_ks=args.boundary_ks, boundary_weight=args.boundary_weight).to(device)

    config = vars(args).copy()
    config.update(
        {
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "input_dim": input_dim,
            "num_slots": num_slots,
            "model_parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    best_score = -float("inf")
    metrics_file = (args.output_dir / "metrics.jsonl").open("w")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            length_loss_weight=args.length_loss_weight,
            monotonic_weight=args.monotonic_weight,
            survival_weights=weights,
            aux_arm_weight=args.aux_arm_weight,
            arms=args.arms,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            monotonicize=args.monotonicize_eval,
            arms=args.arms,
            alphas=args.alphas,
            selection_min_retention=args.selection_min_retention,
        )
        payload = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(payload), flush=True)
        metrics_file.write(json.dumps(payload) + "\n")
        metrics_file.flush()
        score = (
            val_metrics["selected_accept_ratio"]
            if val_metrics["selected_feasible"] >= 0.5
            else -1.0 + val_metrics["selected_accept_retention"]
        )
        state = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(state, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(state, args.output_dir / "best.pt")
    metrics_file.close()

    if args.math500_trace_dir:
        print(json.dumps({"event": "load_math500_sequences_start"}), flush=True)
        math_ds = OriginalTraceSequenceDataset(args.math500_trace_dir, max_seq_len=args.max_seq_len)
        math_loader = DataLoader(
            math_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_sequences,
            pin_memory=device.type == "cuda",
        )
        ckpt = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        math_metrics = evaluate(
            model,
            math_loader,
            device=device,
            monotonicize=args.monotonicize_eval,
            arms=args.arms,
            alphas=args.alphas,
            selection_min_retention=args.selection_min_retention,
        )
        out = {
            "trace_dirs": [str(path) for path in args.math500_trace_dir],
            "checkpoint": str(args.output_dir / "best.pt"),
            "sequence_count": len(math_ds),
            "metrics": math_metrics,
        }
        (args.output_dir / "offline_eval_math500_best.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"event": "math500_eval_done", **out}), flush=True)


if __name__ == "__main__":
    main()
