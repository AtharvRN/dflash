from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_block_policy_head import (
    BlockPolicyDataset,
    _arm_stats,
    _load_bandit_data,
    _metrics,
)
from train_predictor_head import FlatSurvivalHead, Normalizer, TemporalSurvivalHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DFlash contextual bandit block policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config: dict[str, Any] = checkpoint["config"]

    ns = argparse.Namespace(**config)
    ns.seed = int(config.get("seed", 0))
    ns.include_cycle_history = bool(config.get("include_cycle_history", False))
    ns.history_cycles = int(config.get("history_cycles", 8))
    ns.feature_set = str(config.get("feature_set", "all"))
    ns.reconstruct_target_entropy_history = bool(config.get("reconstruct_target_entropy_history", False))
    ns.sequence_window = config.get("sequence_window")
    ns.reward_mode = str(config.get("reward_mode", "throughput_proxy"))
    ns.reward_normalization = str(config.get("reward_normalization", "row"))
    ns.draft_cost = float(config.get("draft_cost", 0.03))
    ns.verify_cost = float(config.get("verify_cost", 0.05))
    ns.waste_cost = float(config.get("waste_cost", 0.10))
    ns.base_cost = float(config.get("base_cost", 1.0))
    ns.length_penalty = float(config.get("length_penalty", 0.5))
    ns.arms = [int(x) for x in config["arms"]]
    ns.arm_kind = str(config.get("arm_kind", "block_size"))
    block_costs = config.get("block_costs")
    ns.block_costs = {int(k): float(v) for k, v in block_costs.items()} if block_costs else None

    seq, static, rewards, raw_rewards, best_idx, accepted_len = _load_bandit_data(
        ns,
        args.eval_jsonl,
        max_rows=None,
        sample=False,
    )
    seq_norm = Normalizer(**config["seq_normalizer"])
    static_norm = Normalizer(**config["static_normalizer"])
    seq = seq_norm.apply(seq)
    static = static_norm.apply(static)

    architecture = str(config.get("architecture", "gru"))
    if architecture == "gru":
        model = TemporalSurvivalHead(
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=len(ns.arms),
            hidden_size=int(config.get("hidden_size", 192)),
            dropout=0.0,
        )
    else:
        model = FlatSurvivalHead(
            seq_len=int(seq.shape[1]),
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=len(ns.arms),
            hidden_size=int(config.get("hidden_size", 192)),
            dropout=0.0,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = BlockPolicyDataset(seq, static, rewards, best_idx, accepted_len.astype(np.float32))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    totals: dict[str, float] = {}
    policy_hist = np.zeros(len(ns.arms), dtype=np.int64)
    pred_chunks: list[torch.Tensor] = []
    count = 0
    for seq_batch, static_batch, reward_batch, best_batch, accepted_batch in loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        static_batch = static_batch.to(device, non_blocking=True)
        reward_batch = reward_batch.to(device, non_blocking=True)
        best_batch = best_batch.to(device, non_blocking=True)
        accepted_batch = accepted_batch.to(device, non_blocking=True)
        logits = model(seq_batch, static_batch)
        pred_chunks.append(logits.argmax(dim=-1).detach().cpu())
        metrics = _metrics(logits, reward_batch, best_batch, accepted_batch, ns.arms, ns.arm_kind)
        batch_size = int(seq_batch.shape[0])
        for key, value in metrics.items():
            if isinstance(value, list):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        policy_hist += np.asarray(metrics["policy_arm_hist"], dtype=np.int64)
        count += batch_size

    raw_reward_tensor = torch.from_numpy(raw_rewards)
    accepted_tensor = torch.from_numpy(accepted_len.astype(np.float32))
    policy_idx = torch.cat(pred_chunks, dim=0)
    policy_raw = _arm_stats(policy_idx, raw_reward_tensor, accepted_tensor, ns.arms, ns.arm_kind)
    fixed_raw = {}
    for arm_idx, arm in enumerate(ns.arms):
        fixed_idx = torch.full((len(dataset),), arm_idx, dtype=torch.long)
        fixed_raw[str(arm)] = _arm_stats(fixed_idx, raw_reward_tensor, accepted_tensor, ns.arms, ns.arm_kind)
    oracle_raw = raw_rewards.max(axis=1)

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "rows": count,
        "arms": ns.arms,
        "arm_kind": ns.arm_kind,
        "reward_mode": ns.reward_mode,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "metrics": {key: value / count for key, value in totals.items()},
        "policy_arm_hist": policy_hist.tolist(),
        "raw_reward": {
            "oracle_mean": float(oracle_raw.mean()),
            "policy": policy_raw,
            "fixed": fixed_raw,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
