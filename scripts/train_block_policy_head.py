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

from train_predictor_head import (
    FlatSurvivalHead,
    Normalizer,
    TemporalSurvivalHead,
    _load_jsonl,
)


class BlockPolicyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        seq: np.ndarray,
        static: np.ndarray,
        rewards: np.ndarray,
        best_arm_idx: np.ndarray,
        accepted_len: np.ndarray,
    ) -> None:
        self.seq = torch.from_numpy(seq)
        self.static = torch.from_numpy(static)
        self.rewards = torch.from_numpy(rewards)
        self.best_arm_idx = torch.from_numpy(best_arm_idx)
        self.accepted_len = torch.from_numpy(accepted_len)

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.seq[idx],
            self.static[idx],
            self.rewards[idx],
            self.best_arm_idx[idx],
            self.accepted_len[idx],
        )


def _parse_arms(value: str) -> list[int]:
    arms = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not arms:
        raise ValueError("at least one arm is required")
    if any(arm <= 0 for arm in arms):
        raise ValueError(f"arms must be positive integers, got {arms}")
    if arms != sorted(set(arms)):
        raise ValueError(f"arms must be sorted and unique, got {arms}")
    return arms


def _draft_budgets(arms: list[int], arm_kind: str) -> list[int]:
    if arm_kind == "drafted_tokens":
        return arms
    if arm_kind == "block_size":
        budgets = [arm - 1 for arm in arms]
        if any(budget <= 0 for budget in budgets):
            raise ValueError(f"block_size arms must be >= 2, got {arms}")
        return budgets
    raise ValueError(f"unknown arm kind: {arm_kind}")


def _make_rewards(
    accepted_len: np.ndarray,
    *,
    arms: list[int],
    arm_kind: str,
    max_available_slots: int,
    reward_mode: str,
    draft_cost: float,
    verify_cost: float,
    waste_cost: float,
    base_cost: float,
    block_costs: dict[int, float] | None,
    length_penalty: float,
) -> np.ndarray:
    draft_budgets = _draft_budgets(arms, arm_kind)
    max_budget = max(draft_budgets)
    if max_budget > max_available_slots:
        raise ValueError(
            f"requested max draft budget {max_budget} from arms {arms}, "
            f"but traces only expose {max_available_slots} draft slots. "
            f"Collect with --block-size {max_budget + 1} or reduce --arms."
        )

    accepted = accepted_len.astype(np.float32)[:, None]
    budget_arr = np.asarray(draft_budgets, dtype=np.float32)[None, :]
    accepted_under_arm = np.minimum(accepted, budget_arr)
    committed = accepted_under_arm + 1.0
    wasted = np.maximum(0.0, budget_arr - accepted)
    acceptance_ratio = accepted_under_arm / np.maximum(budget_arr, 1.0)
    lost_acceptance = accepted - accepted_under_arm

    if reward_mode == "accepted":
        rewards = accepted_under_arm
    elif reward_mode == "committed":
        rewards = committed
    elif reward_mode == "utility":
        rewards = committed - draft_cost * budget_arr - verify_cost * (budget_arr + 1.0) - waste_cost * wasted
    elif reward_mode == "throughput_proxy":
        if block_costs is None:
            cost = base_cost + draft_cost * budget_arr + verify_cost * (budget_arr + 1.0)
        else:
            missing = [arm for arm in arms if arm not in block_costs]
            if missing:
                raise ValueError(f"block cost file is missing arms: {missing}")
            cost = np.asarray([block_costs[arm] for arm in arms], dtype=np.float32)[None, :]
        rewards = committed / np.maximum(cost, 1e-6)
    elif reward_mode == "ratio_preserve":
        rewards = acceptance_ratio - length_penalty * lost_acceptance
    else:
        raise ValueError(f"unknown reward mode: {reward_mode}")
    return rewards.astype(np.float32)


def _normalize_rewards(rewards: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return rewards.astype(np.float32)
    if mode == "global":
        mean = rewards.mean()
        std = rewards.std()
        return ((rewards - mean) / max(float(std), 1e-6)).astype(np.float32)
    if mode == "row":
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True)
        return ((rewards - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    raise ValueError(f"unknown reward normalization: {mode}")


def _arm_stats(
    chosen_idx: torch.Tensor,
    rewards: torch.Tensor,
    accepted_len: torch.Tensor,
    arms: list[int],
    arm_kind: str,
) -> dict[str, float | list[int]]:
    chosen_rewards = rewards.gather(1, chosen_idx[:, None]).squeeze(1)
    oracle_rewards, oracle_idx = rewards.max(dim=1)
    arm_tensor = torch.tensor(arms, device=chosen_idx.device, dtype=torch.float32)
    budget_tensor = torch.tensor(_draft_budgets(arms, arm_kind), device=chosen_idx.device, dtype=torch.float32)
    chosen_arms = arm_tensor[chosen_idx]
    chosen_budgets = budget_tensor[chosen_idx]
    accepted_under_arm = torch.minimum(accepted_len, chosen_budgets)
    committed = accepted_under_arm + 1.0
    wasted = torch.clamp(chosen_budgets - accepted_len, min=0.0)
    acceptance_ratio = accepted_under_arm / torch.clamp(chosen_budgets, min=1.0)
    hist = torch.bincount(chosen_idx.detach().cpu(), minlength=len(arms)).tolist()
    mean_target_accepted = accepted_len.mean()
    return {
        "reward": chosen_rewards.mean().item(),
        "oracle_reward": oracle_rewards.mean().item(),
        "regret": (oracle_rewards - chosen_rewards).mean().item(),
        "arm_accuracy": (chosen_idx == oracle_idx).float().mean().item(),
        "mean_arm": chosen_arms.mean().item(),
        "mean_draft_budget": chosen_budgets.mean().item(),
        "mean_accepted_draft_len": accepted_under_arm.mean().item(),
        "mean_target_accepted_draft_len": mean_target_accepted.item(),
        "accepted_len_retention": (accepted_under_arm.mean() / torch.clamp(mean_target_accepted, min=1e-6)).item(),
        "mean_acceptance_ratio": acceptance_ratio.mean().item(),
        "mean_committed": committed.mean().item(),
        "mean_wasted_drafts": wasted.mean().item(),
        "committed_per_drafted": (committed.sum() / torch.clamp(chosen_budgets.sum(), min=1e-6)).item(),
        "under_arm_rate": (chosen_budgets < accepted_len).float().mean().item(),
        "over_arm_rate": (chosen_budgets > accepted_len).float().mean().item(),
        "arm_hist": hist,
    }


def _metrics(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    best_idx: torch.Tensor,
    accepted_len: torch.Tensor,
    arms: list[int],
    arm_kind: str,
) -> dict[str, Any]:
    pred_idx = logits.argmax(dim=-1)
    out: dict[str, Any] = {
        "ce": F.cross_entropy(logits, best_idx).item(),
    }
    out.update({f"policy_{key}": value for key, value in _arm_stats(pred_idx, rewards, accepted_len, arms, arm_kind).items()})
    for arm_idx, arm in enumerate(arms):
        fixed = torch.full_like(pred_idx, arm_idx)
        fixed_stats = _arm_stats(fixed, rewards, accepted_len, arms, arm_kind)
        out[f"fixed_{arm}_reward"] = fixed_stats["reward"]
        out[f"fixed_{arm}_regret"] = fixed_stats["regret"]
    return out


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    arms: list[int],
    arm_kind: str,
    ce_weight: float,
    reward_weight: float,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    hist = np.zeros(len(arms), dtype=np.int64)
    count = 0
    for seq, static, rewards, best_idx, accepted_len in loader:
        seq = seq.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)
        best_idx = best_idx.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(seq, static)
            probs = F.softmax(logits, dim=-1)
            expected_reward = (probs * rewards).sum(dim=-1)
            oracle_reward = rewards.max(dim=-1).values
            regret_loss = (oracle_reward - expected_reward).mean()
            ce = F.cross_entropy(logits, best_idx)
            loss = ce_weight * ce + reward_weight * regret_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = int(seq.shape[0])
        metrics = _metrics(logits.detach(), rewards, best_idx, accepted_len, arms, arm_kind)
        metrics["loss"] = loss.detach().item()
        metrics["regret_loss"] = regret_loss.detach().item()
        for key, value in metrics.items():
            if isinstance(value, list):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        hist += np.asarray(metrics["policy_arm_hist"], dtype=np.int64)
        count += batch_size

    out: dict[str, Any] = {key: value / count for key, value in totals.items()}
    out["policy_arm_hist"] = hist.tolist()
    return out


def _load_bandit_data(args: argparse.Namespace, path: Path, *, max_rows: int | None, sample: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq, static, survival, accepted_len = _load_jsonl(
        path,
        max_rows=max_rows,
        seed=args.seed,
        sample=sample,
        include_cycle_history=args.include_cycle_history,
        history_cycles=args.history_cycles,
        feature_set=args.feature_set,
        reconstruct_target_entropy_history=args.reconstruct_target_entropy_history,
        sequence_window=args.sequence_window,
    )
    raw_rewards = _make_rewards(
        accepted_len,
        arms=args.arms,
        arm_kind=args.arm_kind,
        max_available_slots=int(survival.shape[1]),
        reward_mode=args.reward_mode,
        draft_cost=args.draft_cost,
        verify_cost=args.verify_cost,
        waste_cost=args.waste_cost,
        base_cost=args.base_cost,
        block_costs=args.block_costs,
        length_penalty=args.length_penalty,
    )
    rewards = _normalize_rewards(raw_rewards, args.reward_normalization)
    best_idx = raw_rewards.argmax(axis=1).astype(np.int64)
    return seq, static, rewards, raw_rewards, best_idx, accepted_len.astype(np.float32)


def _take_rows(x: np.ndarray, *, max_rows: int | None, seed: int, sample: bool) -> np.ndarray:
    if max_rows is None or max_rows >= x.shape[0]:
        return np.asarray(x, dtype=np.float32)
    if not sample:
        return np.asarray(x[:max_rows], dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))
    return np.asarray(x[indices], dtype=np.float32)


def _load_bandit_cache(
    args: argparse.Namespace,
    path: Path,
    *,
    max_rows: int | None,
    sample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.feature_set not in {"internal_only", "internal_window"}:
        raise ValueError("--train-cache-dir/--val-cache-dir only support internal_only and internal_window feature sets")

    accepted_len = _take_rows(
        np.load(path / "accepted_len.npy", mmap_mode="r"),
        max_rows=max_rows,
        seed=args.seed,
        sample=sample,
    )
    survival = _take_rows(
        np.load(path / "survival.npy", mmap_mode="r"),
        max_rows=max_rows,
        seed=args.seed,
        sample=sample,
    )
    if args.feature_set == "internal_only":
        static = _take_rows(
            np.load(path / "dflash_context.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=args.seed,
            sample=sample,
        )
        seq = np.zeros((static.shape[0], 1, 1), dtype=np.float32)
    else:
        seq = _take_rows(
            np.load(path / "dflash_context_window_seq.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=args.seed,
            sample=sample,
        )
        static = np.zeros((seq.shape[0], 0), dtype=np.float32)

    raw_rewards = _make_rewards(
        accepted_len,
        arms=args.arms,
        arm_kind=args.arm_kind,
        max_available_slots=int(survival.shape[1]),
        reward_mode=args.reward_mode,
        draft_cost=args.draft_cost,
        verify_cost=args.verify_cost,
        waste_cost=args.waste_cost,
        base_cost=args.base_cost,
        block_costs=args.block_costs,
        length_penalty=args.length_penalty,
    )
    rewards = _normalize_rewards(raw_rewards, args.reward_normalization)
    best_idx = raw_rewards.argmax(axis=1).astype(np.int64)
    return seq, static, rewards, raw_rewards, best_idx, accepted_len.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFlash pre-draft contextual bandit block-size policy.")
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--train-cache-dir", type=Path, default=None)
    parser.add_argument("--val-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", type=_parse_arms, default="4,8,12,16,20")
    parser.add_argument("--arm-kind", choices=["block_size", "drafted_tokens"], default="block_size")
    parser.add_argument(
        "--reward-mode",
        choices=["accepted", "committed", "utility", "throughput_proxy", "ratio_preserve"],
        default="throughput_proxy",
    )
    parser.add_argument("--reward-normalization", choices=["none", "global", "row"], default="row")
    parser.add_argument("--draft-cost", type=float, default=0.03)
    parser.add_argument("--verify-cost", type=float, default=0.05)
    parser.add_argument("--waste-cost", type=float, default=0.10)
    parser.add_argument("--base-cost", type=float, default=1.0)
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.5,
        help="Penalty per accepted token lost when reward-mode=ratio_preserve.",
    )
    parser.add_argument(
        "--block-cost-json",
        type=Path,
        default=None,
        help="Optional JSON from benchmark_block_costs.py. If set, throughput_proxy uses measured cycle_ms per arm.",
    )
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--architecture", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument(
        "--feature-set",
        choices=["all", "target_entropy_only", "internal_only", "internal_window", "internal_plus_all"],
        default="all",
    )
    parser.add_argument("--reconstruct-target-entropy-history", action="store_true")
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.train_jsonl is None) == (args.train_cache_dir is None):
        parser.error("provide exactly one of --train-jsonl or --train-cache-dir")
    if (args.val_jsonl is None) == (args.val_cache_dir is None):
        parser.error("provide exactly one of --val-jsonl or --val-cache-dir")
    if isinstance(args.arms, str):
        args.arms = _parse_arms(args.arms)
    args.block_costs = _load_block_costs(args.block_cost_json, args.arms) if args.block_cost_json else None
    return args


def _load_block_costs(path: Path, arms: list[int]) -> dict[int, float]:
    payload = json.loads(path.read_text())
    raw_blocks = payload.get("blocks", payload)
    costs: dict[int, float] = {}
    for arm in arms:
        value = raw_blocks.get(str(arm), raw_blocks.get(arm))
        if isinstance(value, dict):
            value = value.get("cycle_ms", value.get("mean_cycle_ms"))
        if value is None:
            raise ValueError(f"{path} does not include a cycle cost for arm {arm}")
        costs[arm] = float(value)
    return costs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_cache_dir is not None:
        print(f"loading train cache from {args.train_cache_dir}", flush=True)
        train_seq, train_static, train_rewards, train_raw_rewards, train_best, train_len = _load_bandit_cache(
            args,
            args.train_cache_dir,
            max_rows=args.max_train_rows,
            sample=args.sample_train_rows,
        )
    else:
        print(f"loading train rows from {args.train_jsonl}", flush=True)
        train_seq, train_static, train_rewards, train_raw_rewards, train_best, train_len = _load_bandit_data(
            args,
            args.train_jsonl,
            max_rows=args.max_train_rows,
            sample=args.sample_train_rows,
        )
    if args.val_cache_dir is not None:
        print(f"loading validation cache from {args.val_cache_dir}", flush=True)
        val_seq, val_static, val_rewards, val_raw_rewards, val_best, val_len = _load_bandit_cache(
            args,
            args.val_cache_dir,
            max_rows=args.max_val_rows,
            sample=False,
        )
    else:
        print(f"loading validation rows from {args.val_jsonl}", flush=True)
        val_seq, val_static, val_rewards, val_raw_rewards, val_best, val_len = _load_bandit_data(
            args,
            args.val_jsonl,
            max_rows=args.max_val_rows,
            sample=False,
        )

    seq_norm = Normalizer.fit(train_seq)
    static_norm = Normalizer.fit(train_static)
    train_seq = seq_norm.apply(train_seq)
    val_seq = seq_norm.apply(val_seq)
    train_static = static_norm.apply(train_static)
    val_static = static_norm.apply(val_static)

    train_ds = BlockPolicyDataset(train_seq, train_static, train_rewards, train_best, train_len)
    val_ds = BlockPolicyDataset(val_seq, val_static, val_rewards, val_best, val_len)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_oracle = train_raw_rewards.max(axis=1).mean().item()
    val_oracle = val_raw_rewards.max(axis=1).mean().item()
    config = vars(args).copy()
    config.update(
        {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "seq_shape": list(train_seq.shape[1:]),
            "static_dim": int(train_static.shape[-1]),
            "num_arms": len(args.arms),
            "seq_normalizer": asdict(seq_norm),
            "static_normalizer": asdict(static_norm),
            "train_raw_oracle_reward": train_oracle,
            "val_raw_oracle_reward": val_oracle,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            arms=args.arms,
            arm_kind=args.arm_kind,
            ce_weight=args.ce_weight,
            reward_weight=args.reward_weight,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            arms=args.arms,
            arm_kind=args.arm_kind,
            ce_weight=args.ce_weight,
            reward_weight=args.reward_weight,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        if float(val_metrics["policy_regret"]) < best_val:
            best_val = float(val_metrics["policy_regret"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best.pt",
            )

    torch.save({"model_state_dict": model.state_dict(), "config": config}, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
