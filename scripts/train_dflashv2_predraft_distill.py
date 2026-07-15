from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dflashv2_horizon_predictor import (  # noqa: E402
    HorizonPredictor,
    _load_shards_multi,
    _make_loader,
    _make_splits,
    evaluate,
)


DEFAULT_ALPHAS = "0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.98"


class DistillHorizonDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        shards,
        indices: np.ndarray,
        teacher_survival: np.ndarray,
        *,
        last_only: bool,
    ) -> None:
        self.shards = shards
        self.indices = np.asarray(indices, dtype=np.int64)
        self.teacher_survival = teacher_survival
        self.last_only = bool(last_only)
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += shard.rows
        if int(teacher_survival.shape[0]) != total:
            raise ValueError(
                f"teacher rows={teacher_survival.shape[0]} do not match trace rows={total}"
            )
        first = self.shards[0]
        self.context_window = 1 if self.last_only else int(first.features.shape[1])
        self.hidden_size = int(first.features.shape[2])
        self.num_slots = int(first.survival.shape[1])
        if int(teacher_survival.shape[1]) != self.num_slots:
            raise ValueError(
                f"teacher slots={teacher_survival.shape[1]} do not match trace slots={self.num_slots}"
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        hard_survival = torch.from_numpy(np.asarray(shard.survival[local_idx], dtype=np.float32))
        accepted_len = torch.tensor(float(shard.accepted_len[local_idx]), dtype=torch.float32)
        teacher = torch.from_numpy(np.asarray(self.teacher_survival[global_idx], dtype=np.float32))
        return features, mask, hard_survival, accepted_len, teacher


def _make_eval_dataset(train_dataset: DistillHorizonDataset, indices: np.ndarray):
    from scripts.train_dflashv2_horizon_predictor import HorizonIndexedDataset

    return HorizonIndexedDataset(
        train_dataset.shards,
        indices,
        last_only=train_dataset.last_only,
    )


def _train_epoch(
    model: HorizonPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    teacher_weight: float,
    hard_weight: float,
    teacher_length_weight: float,
    hard_length_weight: float,
    monotonic_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_rows = 0
    for features, mask, hard_survival, accepted_len, teacher in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        hard_survival = hard_survival.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True).clamp(1e-4, 1.0 - 1e-4)
        logits = model(features, mask)
        probs = torch.sigmoid(logits)
        teacher_bce = F.binary_cross_entropy_with_logits(logits, teacher)
        hard_bce = F.binary_cross_entropy_with_logits(logits, hard_survival)
        expected_len = probs.sum(dim=1)
        teacher_len = teacher.sum(dim=1)
        teacher_length = F.smooth_l1_loss(expected_len, teacher_len)
        hard_length = F.smooth_l1_loss(expected_len, accepted_len)
        monotonic_loss = F.relu(probs[:, 1:] - probs[:, :-1]).mean()
        loss = (
            teacher_weight * teacher_bce
            + hard_weight * hard_bce
            + teacher_length_weight * teacher_length
            + hard_length_weight * hard_length
            + monotonic_weight * monotonic_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch = features.shape[0]
        total_rows += batch
        for key, value in {
            "loss": loss,
            "teacher_bce": teacher_bce,
            "hard_bce": hard_bce,
            "teacher_length_loss": teacher_length,
            "hard_length_loss": hard_length,
            "monotonic_loss": monotonic_loss,
            "expected_len_mae": (expected_len - accepted_len).abs().mean(),
            "teacher_len_mae": (expected_len - teacher_len).abs().mean(),
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch
    return {key: value / total_rows for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pre-draft DFlashv2 horizon predictor from DSPARK teacher curves."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--teacher-survival", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["last_mlp", "gru", "transformer"], default="last_mlp")
    parser.add_argument("--last-only", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=0.7)
    parser.add_argument("--hard-weight", type=float, default=0.3)
    parser.add_argument("--teacher-length-weight", type=float, default=0.05)
    parser.add_argument("--hard-length-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["expected_len_mae", "selected_accept_ratio"],
        default="selected_accept_ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shards = _load_shards_multi(args.trace_dir)
    total_rows = sum(shard.rows for shard in shards)
    train_idx, val_idx = _make_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    teacher = np.load(args.teacher_survival, mmap_mode="r")
    train_dataset = DistillHorizonDataset(
        shards,
        train_idx,
        teacher,
        last_only=args.last_only or args.architecture == "last_mlp",
    )
    val_dataset = DistillHorizonDataset(
        shards,
        val_idx,
        teacher,
        last_only=args.last_only or args.architecture == "last_mlp",
    )
    eval_dataset = _make_eval_dataset(val_dataset, val_idx)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    model = HorizonPredictor(
        input_dim=train_dataset.hidden_size,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_slots=train_dataset.num_slots,
        architecture=args.architecture,
        num_layers=args.num_layers,
        dropout=args.dropout,
        context_window=train_dataset.context_window,
        num_arms=0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = _make_loader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    config = vars(args).copy()
    config.update(
        {
            "input_dim": train_dataset.hidden_size,
            "context_window": train_dataset.context_window,
            "num_slots": train_dataset.num_slots,
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset),
            "objective": "survival_bce",
            "distillation_teacher": str(args.teacher_survival),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_score = -float("inf")
    best_mae = float("inf")
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            teacher_weight=args.teacher_weight,
            hard_weight=args.hard_weight,
            teacher_length_weight=args.teacher_length_weight,
            hard_length_weight=args.hard_length_weight,
            monotonic_weight=args.monotonic_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            objective="survival_bce",
            monotonicize=True,
            arms=arms,
            alphas=alphas,
            selection_min_retention=args.selection_min_retention,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.checkpoint_selection == "expected_len_mae":
            is_best = val_metrics["expected_len_mae"] < best_mae
        else:
            score = (
                val_metrics["selected_accept_ratio"]
                if val_metrics["selected_feasible"] > 0.5
                else -1.0 + val_metrics["selected_accept_retention"]
            )
            is_best = score > best_score
        if is_best:
            best_mae = val_metrics["expected_len_mae"]
            best_score = (
                val_metrics["selected_accept_ratio"]
                if val_metrics["selected_feasible"] > 0.5
                else -1.0 + val_metrics["selected_accept_retention"]
            )
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
