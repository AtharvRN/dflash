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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dflashv2_horizon_predictor import (  # noqa: E402
    _make_splits,
    _policy_metrics,
    _roc_auc,
    _select_alpha_metrics,
)


DEFAULT_ALPHAS = "0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.98"


@dataclass
class DSparkShard:
    path: Path
    rows: int
    postdraft_hidden: np.ndarray
    postdraft_token_ids: np.ndarray
    postdraft_confidence: np.ndarray | None
    survival: np.ndarray
    accepted_len: np.ndarray


def _load_shards(trace_dirs: list[Path], *, require_scalar_confidence: bool) -> list[DSparkShard]:
    shards: list[DSparkShard] = []
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
            hidden_path = shard_dir / "postdraft_hidden.npy"
            token_path = shard_dir / "postdraft_token_ids.npy"
            if not hidden_path.exists() or not token_path.exists():
                raise FileNotFoundError(
                    f"missing DSPARK-style fields in {shard_dir}; recollect with --log-postdraft-hidden"
                )
            scalar_path = shard_dir / "postdraft_confidence.npy"
            if require_scalar_confidence and not scalar_path.exists():
                raise FileNotFoundError(
                    f"missing {scalar_path}; recollect with --log-postdraft-confidence or disable scalar stats"
                )
            shards.append(
                DSparkShard(
                    path=shard_dir,
                    rows=rows,
                    postdraft_hidden=np.load(hidden_path, mmap_mode="r"),
                    postdraft_token_ids=np.load(token_path, mmap_mode="r"),
                    postdraft_confidence=np.load(scalar_path, mmap_mode="r") if scalar_path.exists() else None,
                    survival=np.load(shard_dir / "survival.npy", mmap_mode="r"),
                    accepted_len=np.load(shard_dir / "accepted_len.npy", mmap_mode="r"),
                )
            )
    if not shards:
        raise ValueError("no non-empty shards found")
    return shards


class DSparkConfidenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        shards: list[DSparkShard],
        indices: np.ndarray | None = None,
        *,
        use_scalar_confidence: bool,
    ) -> None:
        self.shards = shards
        self.use_scalar_confidence = use_scalar_confidence
        self.offsets: list[int] = []
        total = 0
        for shard in shards:
            self.offsets.append(total)
            total += shard.rows
        self.total_rows = total
        self.indices = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        first = shards[0]
        self.num_slots = int(first.survival.shape[1])
        self.hidden_size = int(first.postdraft_hidden.shape[2])
        self.scalar_dim = int(first.postdraft_confidence.shape[2]) if use_scalar_confidence else 0

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_idx = global_idx - self.offsets[shard_idx]
        shard = self.shards[shard_idx]
        hidden = np.asarray(shard.postdraft_hidden[local_idx], dtype=np.float16).copy()
        token_ids = np.asarray(shard.postdraft_token_ids[local_idx], dtype=np.int64).copy()
        prev_token_ids = token_ids[:-1]
        if self.use_scalar_confidence:
            assert shard.postdraft_confidence is not None
            scalar = np.asarray(shard.postdraft_confidence[local_idx], dtype=np.float32).copy()
        else:
            scalar = np.zeros((hidden.shape[0], 0), dtype=np.float32)
        survival = np.asarray(shard.survival[local_idx], dtype=np.float32).copy()
        accepted_len = float(shard.accepted_len[local_idx])
        return (
            torch.from_numpy(hidden),
            torch.from_numpy(prev_token_ids),
            torch.from_numpy(scalar),
            torch.from_numpy(survival),
            torch.tensor(accepted_len, dtype=torch.float32),
        )


class DSparkConfidenceHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        proj_dim: int,
        markov_dim: int,
        scalar_dim: int,
        scalar_proj_dim: int,
        head_hidden_size: int,
        vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.prev_token_embed = nn.Embedding(vocab_size, markov_dim)
        self.scalar_proj = (
            nn.Sequential(
                nn.Linear(scalar_dim, scalar_proj_dim),
                nn.GELU(),
                nn.LayerNorm(scalar_proj_dim),
                nn.Dropout(dropout),
            )
            if scalar_dim > 0 and scalar_proj_dim > 0
            else None
        )
        scalar_out = scalar_proj_dim if self.scalar_proj is not None else 0
        self.head = nn.Sequential(
            nn.Linear(proj_dim + markov_dim + scalar_out, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, prev_token_ids: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        hidden_feat = self.hidden_proj(hidden.float())
        prev_token_ids = prev_token_ids.clamp(min=0, max=self.vocab_size - 1)
        markov_feat = self.prev_token_embed(prev_token_ids)
        parts = [hidden_feat, markov_feat]
        if self.scalar_proj is not None:
            parts.append(self.scalar_proj(scalar.float()))
        x = torch.cat(parts, dim=-1)
        return self.head(x).squeeze(-1)


def _conditional_targets(survival: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prev_survived = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    return survival, prev_survived


def _weights(num_slots: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(num_slots, device=device, dtype=torch.float32)
    weights = torch.exp(-pos / float(num_slots))
    return weights / weights.mean().clamp_min(1e-6)


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _threshold_policy_metrics(
    probs: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: tuple[int, ...],
    threshold: float,
) -> dict[str, float]:
    budgets = torch.tensor([arm - 1 for arm in arms], device=probs.device)
    keep_len = (probs >= threshold).sum(dim=1)
    ok = budgets.view(1, -1) >= keep_len.view(-1, 1)
    fallback = torch.full((ok.shape[0],), ok.shape[1] - 1, device=ok.device, dtype=torch.long)
    chosen = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), fallback)
    chosen_budget = budgets[chosen].float()
    true_accept = torch.minimum(accepted_len, chosen_budget)
    full_budget = float(arms[-1] - 1)
    full_accept = torch.minimum(accepted_len, torch.full_like(accepted_len, full_budget))
    prefix = f"threshold{threshold:.2f}"
    return {
        f"{prefix}_mean_block": (chosen_budget + 1).mean().item(),
        f"{prefix}_mean_budget": chosen_budget.mean().item(),
        f"{prefix}_mean_accepted": true_accept.mean().item(),
        f"{prefix}_accept_retention": (true_accept.sum() / full_accept.sum().clamp_min(1)).item(),
        f"{prefix}_accept_ratio": (true_accept / chosen_budget.clamp_min(1)).mean().item(),
    }


def _select_threshold_metrics(
    metrics: dict[str, float],
    *,
    thresholds: tuple[float, ...],
    min_retention: float,
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        prefix = f"threshold{threshold:.2f}"
        candidates.append(
            {
                "threshold": float(threshold),
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
        "selected_threshold": selected["threshold"],
        "threshold_selected_mean_block": selected["mean_block"],
        "threshold_selected_mean_budget": selected["mean_budget"],
        "threshold_selected_mean_accepted": selected["mean_accepted"],
        "threshold_selected_accept_retention": selected["accept_retention"],
        "threshold_selected_accept_ratio": selected["accept_ratio"],
        "threshold_selected_min_retention": float(min_retention),
        "threshold_selected_feasible": feasible_flag,
    }


def _train_epoch(
    model: DSparkConfidenceHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    survival_loss_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_rows = 0
    for hidden, prev_ids, scalar, survival, _accepted_len in loader:
        hidden = hidden.to(device, non_blocking=True)
        prev_ids = prev_ids.to(device, non_blocking=True)
        scalar = scalar.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        logits = model(hidden, prev_ids, scalar)
        cond_target, cond_mask = _conditional_targets(survival)
        weights = _weights(survival.shape[1], device).view(1, -1)
        cond_raw = F.binary_cross_entropy_with_logits(logits, cond_target, reduction="none")
        cond_loss = (cond_raw * cond_mask * weights).sum() / (cond_mask * weights).sum().clamp_min(1.0)
        cond_probs = torch.sigmoid(logits)
        survival_probs = torch.cumprod(cond_probs, dim=1)
        survival_loss = F.binary_cross_entropy(survival_probs.clamp(1e-6, 1 - 1e-6), survival)
        loss = cond_loss + survival_loss_weight * survival_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch = hidden.shape[0]
        total_rows += batch
        expected_len = survival_probs.sum(dim=1)
        accepted_len = survival.sum(dim=1)
        for key, value in {
            "loss": loss,
            "conditional_bce": cond_loss,
            "survival_bce": survival_loss,
            "expected_len_mae": (expected_len - accepted_len).abs().mean(),
        }.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch
    return {key: value / total_rows for key, value in totals.items()}


@torch.inference_mode()
def _evaluate(
    model: DSparkConfidenceHead,
    loader: DataLoader,
    *,
    device: torch.device,
    arms: tuple[int, ...],
    alphas: tuple[float, ...],
    thresholds: tuple[float, ...],
    selection_min_retention: float,
) -> dict[str, float]:
    model.eval()
    probs_all: list[torch.Tensor] = []
    survival_all: list[torch.Tensor] = []
    accepted_all: list[torch.Tensor] = []
    cond_losses: list[float] = []
    surv_losses: list[float] = []
    counts: list[int] = []
    for hidden, prev_ids, scalar, survival, accepted_len in loader:
        hidden = hidden.to(device, non_blocking=True)
        prev_ids = prev_ids.to(device, non_blocking=True)
        scalar = scalar.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        logits = model(hidden, prev_ids, scalar)
        cond_target, cond_mask = _conditional_targets(survival)
        weights = _weights(survival.shape[1], device).view(1, -1)
        cond_raw = F.binary_cross_entropy_with_logits(logits, cond_target, reduction="none")
        cond_loss = (cond_raw * cond_mask * weights).sum() / (cond_mask * weights).sum().clamp_min(1.0)
        cond_probs = torch.sigmoid(logits)
        survival_probs = torch.cumprod(cond_probs, dim=1)
        surv_loss = F.binary_cross_entropy(survival_probs.clamp(1e-6, 1 - 1e-6), survival)
        batch = hidden.shape[0]
        cond_losses.append(float(cond_loss.item()))
        surv_losses.append(float(surv_loss.item()))
        counts.append(batch)
        probs_all.append(survival_probs.cpu())
        survival_all.append(survival.cpu())
        accepted_all.append(accepted_len.float())

    probs = torch.cat(probs_all)
    survival = torch.cat(survival_all)
    accepted_len = torch.cat(accepted_all)
    expected_len = probs.sum(dim=1)
    pred_len = (probs >= 0.5).sum(dim=1).float()
    total = float(sum(counts))
    metrics: dict[str, float] = {
        "conditional_bce": float(sum(x * c for x, c in zip(cond_losses, counts)) / total),
        "survival_bce": float(sum(x * c for x, c in zip(surv_losses, counts)) / total),
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
    metrics.update(
        _select_alpha_metrics(
            metrics,
            alphas=alphas,
            min_retention=selection_min_retention,
        )
    )
    for threshold in thresholds:
        metrics.update(_threshold_policy_metrics(probs, accepted_len, arms=arms, threshold=threshold))
    metrics.update(
        _select_threshold_metrics(
            metrics,
            thresholds=thresholds,
            min_retention=selection_min_retention,
        )
    )
    return metrics


def _load_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DSPARK-style DFlash confidence head.")
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--markov-dim", type=int, default=64)
    parser.add_argument("--scalar-proj-dim", type=int, default=32)
    parser.add_argument("--head-hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--use-scalar-confidence", action="store_true")
    parser.add_argument("--survival-loss-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    arms = tuple(int(x) for x in args.arms.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x)
    shards = _load_shards(args.trace_dir, require_scalar_confidence=args.use_scalar_confidence)
    base_dataset = DSparkConfidenceDataset(
        shards,
        use_scalar_confidence=args.use_scalar_confidence,
    )
    train_idx, val_idx = _make_splits(
        total_rows=base_dataset.total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    train_dataset = DSparkConfidenceDataset(
        shards,
        train_idx,
        use_scalar_confidence=args.use_scalar_confidence,
    )
    val_dataset = DSparkConfidenceDataset(
        shards,
        val_idx,
        use_scalar_confidence=args.use_scalar_confidence,
    )
    model = DSparkConfidenceHead(
        hidden_size=train_dataset.hidden_size,
        proj_dim=args.proj_dim,
        markov_dim=args.markov_dim,
        scalar_dim=train_dataset.scalar_dim,
        scalar_proj_dim=args.scalar_proj_dim,
        head_hidden_size=args.head_hidden_size,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
    ).to(device)
    if args.checkpoint is not None:
        _load_checkpoint(args.checkpoint, model)

    val_loader = _make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    best_metrics: dict[str, Any] = {}
    if not args.eval_only:
        train_loader = _make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        metrics_path = args.output_dir / "metrics.jsonl"
        best_score = -float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                survival_loss_weight=args.survival_loss_weight,
            )
            val_metrics = _evaluate(
                model,
                val_loader,
                device=device,
                arms=arms,
                alphas=alphas,
                thresholds=thresholds,
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
                        "hidden_size": train_dataset.hidden_size,
                        "scalar_dim": train_dataset.scalar_dim,
                        "num_slots": train_dataset.num_slots,
                        "model_kind": "dspark_confidence_head",
                    },
                    args.output_dir / "best.pt",
                )
        torch.save({"model": model.state_dict(), "val_metrics": best_metrics, "args": vars(args)}, args.output_dir / "last.pt")

    config = {
        "model_kind": "dspark_confidence_head",
        "train_rows": int(len(train_dataset)),
        "val_rows": int(len(val_dataset)),
        "hidden_size": int(train_dataset.hidden_size),
        "scalar_dim": int(train_dataset.scalar_dim),
        "num_slots": int(train_dataset.num_slots),
        "args": vars(args),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    checkpoint = args.checkpoint if args.eval_only else args.output_dir / "best.pt"
    if checkpoint.exists():
        _load_checkpoint(checkpoint, model)
    val_metrics = _evaluate(
        model,
        val_loader,
        device=device,
        arms=arms,
        alphas=alphas,
        thresholds=thresholds,
        selection_min_retention=args.selection_min_retention,
    )
    (args.output_dir / "offline_eval_val.json").write_text(json.dumps(val_metrics, indent=2) + "\n")

    if args.eval_trace_dir:
        eval_shards = _load_shards(args.eval_trace_dir, require_scalar_confidence=args.use_scalar_confidence)
        eval_dataset = DSparkConfidenceDataset(
            eval_shards,
            use_scalar_confidence=args.use_scalar_confidence,
        )
        eval_loader = _make_loader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        eval_metrics = _evaluate(
            model,
            eval_loader,
            device=device,
            arms=arms,
            alphas=alphas,
            thresholds=thresholds,
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
