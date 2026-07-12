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
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        if architecture == "gru":
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

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features.float())
        x = x * mask.unsqueeze(-1)
        if self.architecture == "gru":
            encoded, _ = self.encoder(x)
        else:
            encoded = self.encoder(
                x + self.pos_embed[:, : x.shape[1], :],
                src_key_padding_mask=mask < 0.5,
            )
        positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last_valid = (positions * (mask > 0.5)).max(dim=1).values.long()
        gather_idx = last_valid.view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        pooled = encoded.gather(dim=1, index=gather_idx).squeeze(1)
        return self.head(pooled)


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


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    monotonicize: bool,
    arms: tuple[int, ...],
    alphas: tuple[float, ...],
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
        loss = F.binary_cross_entropy_with_logits(logits, survival)
        losses.append(loss.item())
        counts.append(features.shape[0])
        logits_all.append(logits.detach().cpu())
        survival_all.append(survival.detach().cpu())
        accepted_all.append(accepted_len.detach().cpu())

    logits = torch.cat(logits_all)
    survival = torch.cat(survival_all)
    accepted_len = torch.cat(accepted_all)
    probs = torch.sigmoid(logits)
    if monotonicize:
        probs = _monotonicize(probs)
    expected_len = probs.sum(dim=1)
    pred_len = (probs >= 0.5).sum(dim=1).float()
    total = float(sum(counts))
    metrics = {
        "bce": float(sum(loss * count for loss, count in zip(losses, counts)) / total),
        "expected_len_mae": (expected_len - accepted_len).abs().mean().item(),
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
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    length_loss_weight: float,
    monotonic_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_rows = 0
    for features, mask, survival, accepted_len in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        logits = model(features, mask)
        bce = F.binary_cross_entropy_with_logits(logits, survival)
        probs = torch.sigmoid(logits)
        expected_len = probs.sum(dim=1)
        length_loss = F.smooth_l1_loss(expected_len, accepted_len)
        monotonic_loss = F.relu(probs[:, 1:] - probs[:, :-1]).mean()
        loss = bce + length_loss_weight * length_loss + monotonic_weight * monotonic_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch = features.shape[0]
        total_rows += batch
        for key, value in {
            "loss": loss,
            "bce": bce,
            "length_loss": length_loss,
            "monotonic_loss": monotonic_loss,
            "expected_len_mae": (expected_len - accepted_len).abs().mean(),
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
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--architecture", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monotonicize-eval", action="store_true")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default="0.85,0.90,0.95")
    args = parser.parse_args()
    if not args.eval_only and args.train_dir is None:
        parser.error("--train-dir is required unless --eval-only is set")
    if args.eval_only and args.checkpoint is None:
        parser.error("--checkpoint is required with --eval-only")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_ds = HorizonTraceDataset(args.val_dir, max_rows=args.max_val_rows)
    train_ds = None if args.eval_only else HorizonTraceDataset(args.train_dir, max_rows=args.max_train_rows)
    reference_ds = train_ds or val_ds
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
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
    ).to(device)

    config = vars(args).copy()
    config.update(
        {
            "input_dim": reference_ds.hidden_size,
            "context_window": reference_ds.context_window,
            "num_slots": reference_ds.num_slots,
            "train_rows": len(train_ds) if train_ds is not None else None,
            "val_rows": len(val_ds),
            "arms": arms,
            "alphas": alphas,
        }
    )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
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
            monotonicize=args.monotonicize_eval,
            arms=arms,
            alphas=alphas,
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

    best = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            length_loss_weight=args.length_loss_weight,
            monotonic_weight=args.monotonic_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            monotonicize=args.monotonicize_eval,
            arms=arms,
            alphas=alphas,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_metrics["expected_len_mae"] < best:
            best = val_metrics["expected_len_mae"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
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
