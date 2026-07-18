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

from train_dflashv2_horizon_predictor import _load_shards, _load_shards_multi, _make_splits


@dataclass
class IntegerBatch:
    features: torch.Tensor
    mask: torch.Tensor
    accepted_len: torch.Tensor
    token_ids: torch.Tensor
    token_mask: torch.Tensor


class CompactIntegerHorizonDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, split_dir: Path, *, max_rows: int | None = None) -> None:
        self.split_dir = split_dir
        self.features = np.load(split_dir / "features.npy", mmap_mode="r")
        self.mask = np.load(split_dir / "mask.npy", mmap_mode="r")
        self.accepted_len = np.load(split_dir / "accepted_len.npy", mmap_mode="r")
        self.survival = np.load(split_dir / "survival.npy", mmap_mode="r")
        self.token_ids = (
            np.load(split_dir / "predraft_token_ids.npy", mmap_mode="r")
            if (split_dir / "predraft_token_ids.npy").exists()
            else None
        )
        self.token_mask = (
            np.load(split_dir / "predraft_token_mask.npy", mmap_mode="r")
            if (split_dir / "predraft_token_mask.npy").exists()
            else None
        )
        self.rows = int(self.accepted_len.shape[0]) if max_rows is None else min(int(self.accepted_len.shape[0]), max_rows)
        self.context_window = int(self.features.shape[1])
        self.hidden_size = int(self.features.shape[2])
        self.num_classes = int(self.survival.shape[1]) + 1
        self.token_window = int(self.token_ids.shape[1]) if self.token_ids is not None else 1

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.token_ids is None:
            token_ids = torch.zeros((self.token_window,), dtype=torch.long)
            token_mask = torch.zeros((self.token_window,), dtype=torch.float32)
        else:
            token_ids = torch.from_numpy(np.asarray(self.token_ids[idx], dtype=np.int64).copy())
            assert self.token_mask is not None
            token_mask = torch.from_numpy(np.asarray(self.token_mask[idx], dtype=np.float32).copy())
        return (
            torch.from_numpy(np.asarray(self.features[idx]).copy()),
            torch.from_numpy(np.asarray(self.mask[idx]).copy()),
            torch.tensor(float(self.accepted_len[idx]), dtype=torch.float32),
            token_ids,
            token_mask,
        )


class TraceIntegerHorizonDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        trace_dirs: list[Path],
        *,
        indices: np.ndarray | None = None,
        max_rows: int | None = None,
        last_only: bool = True,
    ) -> None:
        self.shards = _load_shards_multi(trace_dirs)
        self.offsets: list[int] = []
        total = 0
        for shard in self.shards:
            self.offsets.append(total)
            total += int(shard.rows)
        if indices is None:
            use_rows = total if max_rows is None else min(total, max_rows)
            self.indices = np.arange(use_rows, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if max_rows is not None:
                self.indices = self.indices[:max_rows]
        first = self.shards[0]
        self.last_only = last_only
        self.context_window = 1 if last_only else int(first.features.shape[1])
        self.hidden_size = int(first.features.shape[2])
        self.num_classes = int(first.survival.shape[1]) + 1
        self.has_tokens = all((shard.path / "predraft_token_ids.npy").exists() for shard in self.shards)
        self.token_window = (
            int(np.load(self.shards[0].path / "predraft_token_ids.npy", mmap_mode="r").shape[1])
            if self.has_tokens
            else 1
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
                np.asarray(shard.features[local_idx, feature_idx : feature_idx + 1], dtype=np.float16).copy()
            )
            mask = torch.ones((1,), dtype=torch.float32) if valid.size else torch.zeros((1,), dtype=torch.float32)
        else:
            features = torch.from_numpy(np.asarray(shard.features[local_idx], dtype=np.float16).copy())
            mask = torch.from_numpy(np.asarray(shard.mask[local_idx], dtype=np.float32).copy())
        if self.has_tokens:
            token_ids_np = np.load(shard.path / "predraft_token_ids.npy", mmap_mode="r")
            token_mask_np = np.load(shard.path / "predraft_token_mask.npy", mmap_mode="r")
            token_ids = torch.from_numpy(np.asarray(token_ids_np[local_idx], dtype=np.int64).copy())
            token_mask = torch.from_numpy(np.asarray(token_mask_np[local_idx], dtype=np.float32).copy())
        else:
            token_ids = torch.zeros((self.token_window,), dtype=torch.long)
            token_mask = torch.zeros((self.token_window,), dtype=torch.float32)
        accepted_len = torch.tensor(float(shard.accepted_len[local_idx]), dtype=torch.float32)
        return features, mask, accepted_len, token_ids, token_mask


class IntegerHorizonHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        proj_dim: int,
        num_classes: int,
        dropout: float,
        use_token_tower: bool,
        vocab_size: int,
        token_embed_dim: int,
        token_hidden_size: int,
    ) -> None:
        super().__init__()
        self.use_token_tower = use_token_tower
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        token_dim = 0
        if use_token_tower:
            self.token_embed = nn.Embedding(vocab_size, token_embed_dim, padding_idx=0)
            self.token_proj = nn.Sequential(
                nn.Linear(token_embed_dim, token_hidden_size),
                nn.GELU(),
                nn.LayerNorm(token_hidden_size),
                nn.Dropout(dropout),
            )
            token_dim = token_hidden_size
        else:
            self.token_embed = None
            self.token_proj = None
        head_in = proj_dim + token_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def _last_fused(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last_valid = (positions * (mask > 0.5)).max(dim=1).values.long()
        gather_idx = last_valid.view(-1, 1, 1).expand(-1, 1, features.shape[-1])
        return features.float().gather(dim=1, index=gather_idx).squeeze(1)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.input_proj(self._last_fused(features, mask))
        if not self.use_token_tower:
            return self.head(fused)
        assert self.token_embed is not None
        assert self.token_proj is not None
        ids = token_ids.clamp(min=0, max=self.token_embed.num_embeddings - 1)
        tok = self.token_embed(ids)
        weights = token_mask.unsqueeze(-1).float()
        pooled = (tok * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        token_features = self.token_proj(pooled)
        return self.head(torch.cat([fused, token_features], dim=-1))


def _class_grid(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_classes, device=device, dtype=torch.float32)


def _soft_targets(target: torch.Tensor, *, num_classes: int, tau: float) -> torch.Tensor:
    grid = _class_grid(num_classes, target.device).view(1, -1)
    dist = torch.exp(-(grid - target.view(-1, 1)).abs() / max(tau, 1e-6))
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _emd_loss(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Survival/CDF-style ordinal distance: compare P(H >= k) for every threshold.
    num_classes = probs.shape[1]
    pred_survival = torch.flip(torch.cumsum(torch.flip(probs, dims=[1]), dim=1), dims=[1])[:, 1:]
    grid = torch.arange(1, num_classes, device=probs.device).view(1, -1)
    target_survival = (target.view(-1, 1) >= grid).float()
    return (pred_survival - target_survival).abs().mean()


def _loss_and_predictions(
    logits: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    ce_weight: float,
    soft_ce_weight: float,
    soft_tau: float,
    distance_weight: float,
    emd_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_classes = logits.shape[1]
    target = accepted_len.long().clamp(min=0, max=num_classes - 1)
    probs = torch.softmax(logits, dim=-1)
    grid = _class_grid(num_classes, logits.device).view(1, -1)
    expected = (probs * grid).sum(dim=-1)
    ce = F.cross_entropy(logits, target)
    soft_target = _soft_targets(accepted_len.clamp(0, num_classes - 1), num_classes=num_classes, tau=soft_tau)
    soft_ce = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    distance = (probs * (grid - accepted_len.view(-1, 1)).abs()).sum(dim=-1).mean()
    emd = _emd_loss(probs, accepted_len.clamp(0, num_classes - 1))
    loss = ce_weight * ce + soft_ce_weight * soft_ce + distance_weight * distance + emd_weight * emd
    return loss, {
        "ce": ce,
        "soft_ce": soft_ce,
        "distance_loss": distance,
        "emd_loss": emd,
        "expected_len": expected,
        "argmax_len": probs.argmax(dim=-1).float(),
        "rounded_len": expected.round().clamp(0, num_classes - 1),
    }


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _batch_to_device(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> IntegerBatch:
    features, mask, accepted_len, token_ids, token_mask = batch
    return IntegerBatch(
        features=features.to(device, non_blocking=True),
        mask=mask.to(device, non_blocking=True),
        accepted_len=accepted_len.to(device, non_blocking=True),
        token_ids=token_ids.to(device, non_blocking=True),
        token_mask=token_mask.to(device, non_blocking=True),
    )


def _metric_summary(predictions: dict[str, torch.Tensor], accepted_len: torch.Tensor) -> dict[str, float]:
    expected = predictions["expected_len"]
    rounded = predictions["rounded_len"]
    argmax = predictions["argmax_len"]
    return {
        "expected_len_mae": (expected - accepted_len).abs().mean().item(),
        "expected_len_rmse": torch.sqrt(torch.mean((expected - accepted_len) ** 2)).item(),
        "rounded_expected_len_mae": (rounded - accepted_len).abs().mean().item(),
        "rounded_expected_len_exact": (rounded == accepted_len).float().mean().item(),
        "argmax_len_mae": (argmax - accepted_len).abs().mean().item(),
        "argmax_len_exact": (argmax == accepted_len).float().mean().item(),
        "mean_expected_len": expected.mean().item(),
        "mean_argmax_len": argmax.mean().item(),
        "mean_target_len": accepted_len.mean().item(),
    }


@torch.inference_mode()
def evaluate(model: IntegerHorizonHead, loader: DataLoader, *, device: torch.device, loss_args: dict[str, float]) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    rows = 0
    all_expected: list[torch.Tensor] = []
    all_accepted: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        logits = model(batch.features, batch.mask, batch.token_ids, batch.token_mask)
        loss, preds = _loss_and_predictions(logits, batch.accepted_len, **loss_args)
        metrics = _metric_summary(preds, batch.accepted_len)
        metrics.update({key: value.detach().item() for key, value in preds.items() if key.endswith("_loss")})
        metrics["loss"] = loss.detach().item()
        metrics["ce"] = preds.get("ce", torch.tensor(float("nan"), device=device)).detach().item()
        batch_rows = int(batch.accepted_len.shape[0])
        rows += batch_rows
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_rows
        all_expected.append(preds["expected_len"].detach().cpu())
        all_accepted.append(batch.accepted_len.detach().cpu())
    out = {key: value / max(rows, 1) for key, value in totals.items()}
    expected = torch.cat(all_expected)
    accepted = torch.cat(all_accepted)
    for constant in (4.0, 5.0, 6.0, 8.0):
        out[f"constant_{int(constant)}_mae"] = (accepted - constant).abs().mean().item()
    out["constant_mean_target_mae"] = (accepted - accepted.mean()).abs().mean().item()
    out["mean_target_len_check"] = accepted.mean().item()
    return out


def train_epoch(
    model: IntegerHorizonHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_args: dict[str, float],
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    rows = 0
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        logits = model(batch.features, batch.mask, batch.token_ids, batch.token_mask)
        loss, preds = _loss_and_predictions(logits, batch.accepted_len, **loss_args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = _metric_summary(preds, batch.accepted_len)
        metrics.update(
            {
                "loss": loss.detach().item(),
                "ce": preds["ce"].detach().item(),
                "soft_ce": preds["soft_ce"].detach().item(),
                "distance_loss": preds["distance_loss"].detach().item(),
                "emd_loss": preds["emd_loss"].detach().item(),
            }
        )
        batch_rows = int(batch.accepted_len.shape[0])
        rows += batch_rows
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_rows
    return {key: value / max(rows, 1) for key, value in totals.items()}


def _build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.train_dir is not None and args.val_dir is not None:
        return (
            CompactIntegerHorizonDataset(args.train_dir, max_rows=args.max_train_rows),
            CompactIntegerHorizonDataset(args.val_dir, max_rows=args.max_val_rows),
        )
    if args.trace_dir is None:
        raise ValueError("provide either --train-dir/--val-dir or --trace-dir")
    shards, _manifest = _load_shards(args.trace_dir[0])
    total_rows = sum(int(shard.rows) for shard in shards)
    train_idx, val_idx = _make_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    return (
        TraceIntegerHorizonDataset(args.trace_dir, indices=train_idx, last_only=args.last_only),
        TraceIntegerHorizonDataset(args.trace_dir, indices=val_idx, last_only=args.last_only),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MAE-focused integer DFlashv2 horizon predictor.")
    parser.add_argument("--trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--last-only", action="store_true", default=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-token-tower", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--token-embed-dim", type=int, default=128)
    parser.add_argument("--token-hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--soft-ce-weight", type=float, default=0.0)
    parser.add_argument("--soft-tau", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.2)
    parser.add_argument("--emd-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("expected_len_mae", "rounded_expected_len_mae", "argmax_len_mae"),
        default="rounded_expected_len_mae",
    )
    args = parser.parse_args()
    if args.eval_only and args.checkpoint is None:
        parser.error("--checkpoint is required with --eval-only")
    if args.eval_only and args.val_dir is None and args.trace_dir is None:
        parser.error("--eval-only requires --val-dir or --trace-dir")
    if not args.eval_only and not ((args.train_dir is not None and args.val_dir is not None) or args.trace_dir is not None):
        parser.error("training requires --train-dir/--val-dir or --trace-dir")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        assert args.checkpoint is not None
        if args.val_dir is not None:
            val_ds: Dataset = CompactIntegerHorizonDataset(args.val_dir, max_rows=args.max_val_rows)
        else:
            assert args.trace_dir is not None
            val_ds = TraceIntegerHorizonDataset(args.trace_dir, max_rows=args.max_val_rows, last_only=args.last_only)
        train_ds = None
    else:
        train_ds, val_ds = _build_datasets(args)

    reference = train_ds or val_ds
    model = IntegerHorizonHead(
        input_dim=reference.hidden_size,
        hidden_size=args.hidden_size,
        proj_dim=args.proj_dim,
        num_classes=reference.num_classes,
        dropout=args.dropout,
        use_token_tower=args.use_token_tower,
        vocab_size=args.vocab_size,
        token_embed_dim=args.token_embed_dim,
        token_hidden_size=args.token_hidden_size,
    ).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    loss_args = {
        "ce_weight": float(args.ce_weight),
        "soft_ce_weight": float(args.soft_ce_weight),
        "soft_tau": float(args.soft_tau),
        "distance_weight": float(args.distance_weight),
        "emd_weight": float(args.emd_weight),
    }
    config = vars(args).copy()
    config.update(
        {
            "input_dim": reference.hidden_size,
            "num_classes": reference.num_classes,
            "context_window": reference.context_window,
            "token_window": reference.token_window,
            "train_rows": len(train_ds) if train_ds is not None else None,
            "val_rows": len(val_ds),
            "loss_args": loss_args,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    val_loader = _make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.eval_only:
        metrics = evaluate(model, val_loader, device=device, loss_args=loss_args)
        (args.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        return

    assert train_ds is not None
    train_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device=device, loss_args=loss_args)
        val_metrics = evaluate(model, val_loader, device=device, loss_args=loss_args)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        score = float(val_metrics[args.checkpoint_selection])
        if score < best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_metric": args.checkpoint_selection,
                    "best_score": best,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best.pt",
            )
    torch.save({"model_state_dict": model.state_dict(), "config": config}, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
