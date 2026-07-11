from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_block_policy_head import _draft_budgets, _parse_arms
from train_predictor_head import FlatSurvivalHead, Normalizer, TemporalSurvivalHead, _load_jsonl


class OracleArmDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        seq: np.ndarray,
        static: np.ndarray,
        target_idx: np.ndarray,
        accepted_len: np.ndarray,
    ) -> None:
        self.seq = torch.from_numpy(seq)
        self.static = torch.from_numpy(static)
        self.target_idx = torch.from_numpy(target_idx.astype(np.int64))
        self.accepted_len = torch.from_numpy(accepted_len.astype(np.float32))

    def __len__(self) -> int:
        return int(self.target_idx.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq[idx], self.static[idx], self.target_idx[idx], self.accepted_len[idx]


def _take_rows(x: np.ndarray, *, max_rows: int | None, seed: int, sample: bool) -> np.ndarray:
    if max_rows is None or max_rows >= x.shape[0]:
        return np.asarray(x, dtype=np.float32)
    if not sample:
        return np.asarray(x[:max_rows], dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))
    return np.asarray(x[idx], dtype=np.float32)


def _load_cache(
    path: Path,
    *,
    feature_set: str,
    max_rows: int | None,
    seed: int,
    sample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted_len = _take_rows(
        np.load(path / "accepted_len.npy", mmap_mode="r"),
        max_rows=max_rows,
        seed=seed,
        sample=sample,
    )
    if feature_set == "internal_only":
        static = _take_rows(
            np.load(path / "dflash_context.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=seed,
            sample=sample,
        )
        seq = np.zeros((static.shape[0], 1, 1), dtype=np.float32)
    elif feature_set == "internal_window":
        seq = _take_rows(
            np.load(path / "dflash_context_window_seq.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=seed,
            sample=sample,
        )
        static = np.zeros((seq.shape[0], 0), dtype=np.float32)
    else:
        raise ValueError(f"unsupported cache feature set: {feature_set}")
    return seq, static, accepted_len


def _oracle_idx(accepted_len: np.ndarray, *, arms: list[int], arm_kind: str) -> np.ndarray:
    budgets = np.asarray(_draft_budgets(arms, arm_kind), dtype=np.float32)
    accepted = accepted_len.astype(np.float32)[:, None]
    ok = budgets[None, :] >= accepted
    first = ok.argmax(axis=1)
    first[~ok.any(axis=1)] = len(arms) - 1
    return first.astype(np.int64)


def _hist(idx: torch.Tensor, n: int) -> list[int]:
    return torch.bincount(idx.detach().cpu(), minlength=n).tolist()


def _metrics(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: list[int],
    arm_kind: str,
) -> dict[str, Any]:
    pred_idx = logits.argmax(dim=-1)
    arm_tensor = torch.tensor(arms, dtype=torch.float32, device=logits.device)
    budget_tensor = torch.tensor(_draft_budgets(arms, arm_kind), dtype=torch.float32, device=logits.device)
    pred_budget = budget_tensor[pred_idx]
    target_budget = budget_tensor[target_idx]
    pred_arm = arm_tensor[pred_idx]
    target_arm = arm_tensor[target_idx]
    accepted_under_pred = torch.minimum(accepted_len, pred_budget)
    wasted = torch.clamp(pred_budget - accepted_len, min=0.0)
    under = pred_budget < accepted_len
    over_or_exact = pred_budget >= accepted_len
    return {
        "ce": F.cross_entropy(logits, target_idx).item(),
        "accuracy": (pred_idx == target_idx).float().mean().item(),
        "within_one_arm": ((pred_idx - target_idx).abs() <= 1).float().mean().item(),
        "under_rate": under.float().mean().item(),
        "over_rate": (pred_budget > accepted_len).float().mean().item(),
        "sufficient_rate": over_or_exact.float().mean().item(),
        "mean_pred_arm": pred_arm.mean().item(),
        "mean_target_arm": target_arm.mean().item(),
        "mean_pred_budget": pred_budget.mean().item(),
        "mean_target_budget": target_budget.mean().item(),
        "mean_accepted_len": accepted_len.mean().item(),
        "mean_accepted_under_pred": accepted_under_pred.mean().item(),
        "accepted_retention": (
            accepted_under_pred.mean() / torch.clamp(accepted_len.mean(), min=1e-6)
        ).item(),
        "mean_wasted_drafts": wasted.mean().item(),
        "target_arm_hist": _hist(target_idx, len(arms)),
        "pred_arm_hist": _hist(pred_idx, len(arms)),
    }


def _fixed_metrics(
    accepted_len: torch.Tensor,
    *,
    arm_idx: int,
    arms: list[int],
    arm_kind: str,
) -> dict[str, Any]:
    idx = torch.full((accepted_len.shape[0],), arm_idx, dtype=torch.long, device=accepted_len.device)
    logits = torch.full((accepted_len.shape[0], len(arms)), -1000.0, device=accepted_len.device)
    logits[:, arm_idx] = 0.0
    target_np = _oracle_idx(
        accepted_len.detach().cpu().numpy(),
        arms=arms,
        arm_kind=arm_kind,
    )
    target_idx = torch.from_numpy(target_np).to(accepted_len.device)
    out = _metrics(logits, target_idx, accepted_len, arms=arms, arm_kind=arm_kind)
    out["pred_arm_hist"] = _hist(idx, len(arms))
    return out


def _loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    accepted_len: torch.Tensor,
    *,
    arms: list[int],
    arm_kind: str,
    class_weights: torch.Tensor | None,
    under_weight: float,
    over_weight: float,
    ordinal_weight: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target_idx, weight=class_weights)
    if ordinal_weight <= 0:
        return ce
    probs = F.softmax(logits, dim=-1)
    budgets = torch.tensor(_draft_budgets(arms, arm_kind), dtype=torch.float32, device=logits.device)
    expected_budget = (probs * budgets[None, :]).sum(dim=-1)
    under = torch.clamp(accepted_len - expected_budget, min=0.0)
    over = torch.clamp(expected_budget - accepted_len, min=0.0)
    return ce + ordinal_weight * (under_weight * under + over_weight * over).mean()


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    arms: list[int],
    arm_kind: str,
    class_weights: torch.Tensor | None,
    under_weight: float,
    over_weight: float,
    ordinal_weight: float,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    pred_hist = np.zeros(len(arms), dtype=np.int64)
    target_hist = np.zeros(len(arms), dtype=np.int64)
    count = 0
    for seq, static, target_idx, accepted_len in loader:
        seq = seq.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            logits = model(seq, static)
            loss = _loss(
                logits,
                target_idx,
                accepted_len,
                arms=arms,
                arm_kind=arm_kind,
                class_weights=class_weights,
                under_weight=under_weight,
                over_weight=over_weight,
                ordinal_weight=ordinal_weight,
            )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_size = int(seq.shape[0])
        metrics = _metrics(logits.detach(), target_idx, accepted_len, arms=arms, arm_kind=arm_kind)
        metrics["loss"] = loss.detach().item()
        pred_hist += np.asarray(metrics["pred_arm_hist"], dtype=np.int64)
        target_hist += np.asarray(metrics["target_arm_hist"], dtype=np.int64)
        for key, value in metrics.items():
            if isinstance(value, list):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    out: dict[str, Any] = {key: value / count for key, value in totals.items()}
    out["pred_arm_hist"] = pred_hist.tolist()
    out["target_arm_hist"] = target_hist.tolist()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a per-request oracle block-size classifier for DFlash.")
    parser.add_argument("--train-cache-dir", type=Path, default=None)
    parser.add_argument("--val-cache-dir", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", type=_parse_arms, default="4,8,12,16")
    parser.add_argument("--arm-kind", choices=["block_size", "drafted_tokens"], default="block_size")
    parser.add_argument(
        "--feature-set",
        choices=["all", "target_entropy_only", "internal_only", "internal_window", "internal_plus_all"],
        default="internal_window",
    )
    parser.add_argument("--architecture", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--sample-train-rows", action="store_true")
    parser.add_argument("--include-cycle-history", action="store_true")
    parser.add_argument("--history-cycles", type=int, default=8)
    parser.add_argument("--reconstruct-target-entropy-history", action="store_true")
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument("--balanced-ce", action="store_true")
    parser.add_argument("--ordinal-weight", type=float, default=0.0)
    parser.add_argument("--under-weight", type=float, default=2.0)
    parser.add_argument("--over-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if isinstance(args.arms, str):
        args.arms = _parse_arms(args.arms)
    cache_pair = args.train_cache_dir is not None or args.val_cache_dir is not None
    jsonl_pair = args.train_jsonl is not None or args.val_jsonl is not None
    if cache_pair == jsonl_pair:
        parser.error("provide exactly one input pair: --train-cache-dir/--val-cache-dir or --train-jsonl/--val-jsonl")
    if cache_pair and (args.train_cache_dir is None or args.val_cache_dir is None):
        parser.error("--train-cache-dir and --val-cache-dir must be provided together")
    if jsonl_pair and (args.train_jsonl is None or args.val_jsonl is None):
        parser.error("--train-jsonl and --val-jsonl must be provided together")
    if cache_pair and args.feature_set not in {"internal_only", "internal_window"}:
        parser.error("cache inputs only support --feature-set internal_only or internal_window")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train_jsonl is not None:
        train_seq, train_static, _train_survival, train_len = _load_jsonl(
            args.train_jsonl,
            max_rows=args.max_train_rows,
            seed=args.seed,
            sample=args.sample_train_rows,
            include_cycle_history=args.include_cycle_history,
            history_cycles=args.history_cycles,
            feature_set=args.feature_set,
            reconstruct_target_entropy_history=args.reconstruct_target_entropy_history,
            sequence_window=args.sequence_window,
        )
        val_seq, val_static, _val_survival, val_len = _load_jsonl(
            args.val_jsonl,
            max_rows=args.max_val_rows,
            seed=args.seed,
            sample=False,
            include_cycle_history=args.include_cycle_history,
            history_cycles=args.history_cycles,
            feature_set=args.feature_set,
            reconstruct_target_entropy_history=args.reconstruct_target_entropy_history,
            sequence_window=args.sequence_window,
        )
    else:
        train_seq, train_static, train_len = _load_cache(
            args.train_cache_dir,
            feature_set=args.feature_set,
            max_rows=args.max_train_rows,
            seed=args.seed,
            sample=args.sample_train_rows,
        )
        val_seq, val_static, val_len = _load_cache(
            args.val_cache_dir,
            feature_set=args.feature_set,
            max_rows=args.max_val_rows,
            seed=args.seed,
            sample=False,
        )
    train_target = _oracle_idx(train_len, arms=args.arms, arm_kind=args.arm_kind)
    val_target = _oracle_idx(val_len, arms=args.arms, arm_kind=args.arm_kind)

    seq_norm = Normalizer.fit(train_seq)
    static_norm = Normalizer.fit(train_static)
    train_seq = seq_norm.apply(train_seq)
    val_seq = seq_norm.apply(val_seq)
    train_static = static_norm.apply(train_static)
    val_static = static_norm.apply(val_static)

    train_ds = OracleArmDataset(train_seq, train_static, train_target, train_len)
    val_ds = OracleArmDataset(val_seq, val_static, val_target, val_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.architecture == "gru":
        model: nn.Module = TemporalSurvivalHead(
            seq_dim=int(train_seq.shape[-1]),
            static_dim=int(train_static.shape[-1]),
            num_slots=len(args.arms),
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    else:
        model = FlatSurvivalHead(
            seq_len=int(train_seq.shape[1]),
            seq_dim=int(train_seq.shape[-1]),
            static_dim=int(train_static.shape[-1]),
            num_slots=len(args.arms),
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    model.to(device)

    class_weights = None
    if args.balanced_ce:
        counts = np.bincount(train_target, minlength=len(args.arms)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "seq_shape": list(train_seq.shape[1:]),
            "static_dim": int(train_static.shape[-1]),
            "num_arms": len(args.arms),
            "train_target_hist": np.bincount(train_target, minlength=len(args.arms)).tolist(),
            "val_target_hist": np.bincount(val_target, minlength=len(args.arms)).tolist(),
            "seq_normalizer": asdict(seq_norm),
            "static_normalizer": asdict(static_norm),
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    accepted_tensor = torch.from_numpy(val_len.astype(np.float32)).to(device)
    fixed = {
        str(arm): _fixed_metrics(
            accepted_tensor,
            arm_idx=i,
            arms=args.arms,
            arm_kind=args.arm_kind,
        )
        for i, arm in enumerate(args.arms)
    }
    (args.output_dir / "fixed_baselines.json").write_text(json.dumps(fixed, indent=2) + "\n")

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            arms=args.arms,
            arm_kind=args.arm_kind,
            class_weights=class_weights,
            under_weight=args.under_weight,
            over_weight=args.over_weight,
            ordinal_weight=args.ordinal_weight,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            arms=args.arms,
            arm_kind=args.arm_kind,
            class_weights=class_weights,
            under_weight=args.under_weight,
            over_weight=args.over_weight,
            ordinal_weight=args.ordinal_weight,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        score = float(val_metrics["accuracy"]) - 0.25 * float(val_metrics["under_rate"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "fixed_baselines": fixed,
                },
                args.output_dir / "best.pt",
            )

    torch.save({"model_state_dict": model.state_dict(), "config": config}, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
