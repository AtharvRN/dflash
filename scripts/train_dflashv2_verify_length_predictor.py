from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dflashv2_horizon_predictor import (  # noqa: E402
    HorizonPredictor,
    _make_splits,
    _select_alpha_metrics,
    _survival_weights,
    evaluate,
    train_epoch,
)


DEFAULT_ALPHAS = "0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.98"


@dataclass
class VerifyShard:
    path: Path
    rows: int
    features: np.ndarray
    mask: np.ndarray
    postdraft_confidence: np.ndarray
    survival: np.ndarray
    accepted_len: np.ndarray


def _load_shards(trace_dirs: list[Path]) -> list[VerifyShard]:
    shards: list[VerifyShard] = []
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
            postdraft_path = shard_dir / "postdraft_confidence.npy"
            if not postdraft_path.exists():
                raise FileNotFoundError(
                    f"missing {postdraft_path}; recollect traces with --log-postdraft-confidence"
                )
            shards.append(
                VerifyShard(
                    path=shard_dir,
                    rows=rows,
                    features=np.load(shard_dir / "features.npy", mmap_mode="r"),
                    mask=np.load(shard_dir / "mask.npy", mmap_mode="r"),
                    postdraft_confidence=np.load(postdraft_path, mmap_mode="r"),
                    survival=np.load(shard_dir / "survival.npy", mmap_mode="r"),
                    accepted_len=np.load(shard_dir / "accepted_len.npy", mmap_mode="r"),
                )
            )
    if not shards:
        raise ValueError("no non-empty shards found")
    return shards


class VerifyIndexedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, shards: list[VerifyShard], indices: np.ndarray | None = None) -> None:
        self.shards = shards
        self.offsets: list[int] = []
        total = 0
        for shard in shards:
            self.offsets.append(total)
            total += shard.rows
        self.total_rows = total
        self.indices = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        first = shards[0]
        self.context_window = 1
        self.hidden_size = int(first.features.shape[2]) + int(np.prod(first.postdraft_confidence.shape[1:]))
        self.num_slots = int(first.survival.shape[1])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_idx = global_idx - self.offsets[shard_idx]
        shard = self.shards[shard_idx]

        raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
        valid = np.flatnonzero(raw_mask > 0.5)
        feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
        fused = np.asarray(shard.features[local_idx, feature_idx], dtype=np.float32)
        postdraft = np.asarray(shard.postdraft_confidence[local_idx], dtype=np.float32).reshape(-1)
        feature = np.concatenate([fused, postdraft], axis=0)[None, :]
        survival = np.asarray(shard.survival[local_idx], dtype=np.float32)
        accepted_len = float(shard.accepted_len[local_idx])
        return (
            torch.from_numpy(feature),
            torch.ones((1,), dtype=torch.float32),
            torch.from_numpy(survival),
            torch.tensor(accepted_len, dtype=torch.float32),
        )


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _load_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a post-draft DFlashv2 verification-length survival predictor."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--boundary-ks", default="4,8,12,15")
    parser.add_argument("--aux-arm-weight", type=float, default=0.1)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument("--monotonicize-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    boundary_ks = tuple(int(x) for x in args.boundary_ks.split(",") if x)

    shards = _load_shards(args.trace_dir)
    base_dataset = VerifyIndexedDataset(shards)
    train_idx, val_idx = _make_splits(
        total_rows=base_dataset.total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    train_dataset = VerifyIndexedDataset(shards, train_idx)
    val_dataset = VerifyIndexedDataset(shards, val_idx)

    model = HorizonPredictor(
        input_dim=train_dataset.hidden_size,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_slots=train_dataset.num_slots,
        architecture="last_mlp",
        num_layers=1,
        dropout=args.dropout,
        context_window=1,
        num_arms=len(arms) if args.aux_arm_weight > 0 else 0,
    ).to(device)

    if args.checkpoint is not None:
        _load_checkpoint(args.checkpoint, model)

    val_loader = _make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    best_metrics: dict[str, Any] = {}
    if not args.eval_only:
        train_loader = _make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        weights = _survival_weights(
            train_dataset.num_slots,
            boundary_ks=boundary_ks,
            boundary_weight=args.boundary_weight,
        ).to(device)
        metrics_path = args.output_dir / "metrics.jsonl"
        best_score = -float("inf")
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
                arms=arms,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                monotonicize=args.monotonicize_eval,
                arms=arms,
                alphas=alphas,
                selection_min_retention=args.selection_min_retention,
            )
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

            selected = _select_alpha_metrics(
                val_metrics,
                alphas=alphas,
                min_retention=args.selection_min_retention,
            )
            score = selected["selected_accept_ratio"] if selected["selected_feasible"] else -1.0
            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(args),
                        "input_dim": train_dataset.hidden_size,
                        "num_slots": train_dataset.num_slots,
                        "feature_columns": {
                            "prefix": "latest DFlash fused context vector",
                            "postdraft_confidence": [
                                "draft_entropy",
                                "draft_token_prob",
                                "draft_top1_top2_margin",
                                "draft_token_logprob",
                            ],
                        },
                    },
                    args.output_dir / "best.pt",
                )
        torch.save({"model": model.state_dict(), "val_metrics": best_metrics, "args": vars(args)}, args.output_dir / "last.pt")

    config = {
        "model_kind": "postdraft_verify_length_survival",
        "train_rows": int(len(train_dataset)),
        "val_rows": int(len(val_dataset)),
        "input_dim": int(train_dataset.hidden_size),
        "num_slots": int(train_dataset.num_slots),
        "args": vars(args),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    checkpoint = args.checkpoint if args.eval_only else args.output_dir / "best.pt"
    if checkpoint.exists():
        _load_checkpoint(checkpoint, model)
    eval_metrics = evaluate(
        model,
        val_loader,
        device=device,
        monotonicize=args.monotonicize_eval,
        arms=arms,
        alphas=alphas,
        selection_min_retention=args.selection_min_retention,
    )
    (args.output_dir / "offline_eval_val.json").write_text(json.dumps(eval_metrics, indent=2) + "\n")

    if args.eval_trace_dir:
        eval_shards = _load_shards(args.eval_trace_dir)
        eval_dataset = VerifyIndexedDataset(eval_shards)
        eval_loader = _make_loader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        eval_metrics = evaluate(
            model,
            eval_loader,
            device=device,
            monotonicize=args.monotonicize_eval,
            arms=arms,
            alphas=alphas,
            selection_min_retention=args.selection_min_retention,
        )
        payload = {
            "trace_dirs": [str(path) for path in args.eval_trace_dir],
            "checkpoint": str(checkpoint),
            "rows": int(len(eval_dataset)),
            "metrics": eval_metrics,
        }
        (args.output_dir / "offline_eval_external.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"event": "external_eval_done", **payload}), flush=True)


if __name__ == "__main__":
    main()
