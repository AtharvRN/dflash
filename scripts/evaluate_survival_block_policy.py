from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_predictor_head import FlatSurvivalHead, Normalizer, TemporalSurvivalHead, _load_cache, _load_jsonl


class SurvivalEvalDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, seq: np.ndarray, static: np.ndarray, survival: np.ndarray, accepted_len: np.ndarray) -> None:
        self.seq = torch.from_numpy(seq)
        self.static = torch.from_numpy(static)
        self.survival = torch.from_numpy(survival)
        self.accepted_len = torch.from_numpy(accepted_len.astype(np.float32))

    def __len__(self) -> int:
        return int(self.accepted_len.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq[idx], self.static[idx], self.survival[idx], self.accepted_len[idx]


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if out != sorted(set(out)):
        raise ValueError(f"values must be sorted and unique, got {out}")
    return out


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _block_stats(chosen_idx: torch.Tensor, accepted_len: torch.Tensor, arms: list[int]) -> dict[str, Any]:
    arm_tensor = torch.tensor(arms, device=accepted_len.device, dtype=torch.float32)
    budget_tensor = arm_tensor - 1.0
    chosen_arms = arm_tensor[chosen_idx]
    chosen_budgets = budget_tensor[chosen_idx]
    accepted_under_arm = torch.minimum(accepted_len, chosen_budgets)
    target_mean = accepted_len.mean()
    acceptance_ratio = accepted_under_arm / torch.clamp(chosen_budgets, min=1.0)
    hist = torch.bincount(chosen_idx.detach().cpu(), minlength=len(arms)).tolist()
    return {
        "mean_arm": chosen_arms.mean().item(),
        "mean_draft_budget": chosen_budgets.mean().item(),
        "mean_accepted_draft_len": accepted_under_arm.mean().item(),
        "mean_target_accepted_draft_len": target_mean.item(),
        "accepted_len_retention": (accepted_under_arm.mean() / torch.clamp(target_mean, min=1e-6)).item(),
        "mean_acceptance_ratio": acceptance_ratio.mean().item(),
        "mean_wasted_drafts": torch.clamp(chosen_budgets - accepted_len, min=0.0).mean().item(),
        "under_arm_rate": (chosen_budgets < accepted_len).float().mean().item(),
        "over_arm_rate": (chosen_budgets > accepted_len).float().mean().item(),
        "arm_hist": hist,
    }


def _monotonicize(probs: torch.Tensor) -> torch.Tensor:
    return torch.cummin(probs, dim=1).values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre-draft survival curves as block-size policies.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--eval-cache-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--arms", type=_parse_ints, default="4,8,12,16")
    parser.add_argument("--alphas", type=_parse_floats, default="0.85,0.90,0.95,0.98,1.0")
    parser.add_argument("--monotonicize-probs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.eval_jsonl is None) == (args.eval_cache_dir is None):
        parser.error("provide exactly one of --eval-jsonl or --eval-cache-dir")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config: dict[str, Any] = checkpoint["config"]
    feature_set = str(config.get("feature_set", "all"))

    if args.eval_cache_dir is not None:
        seq, static, survival, accepted_len = _load_cache(
            args.eval_cache_dir,
            max_rows=None,
            seed=int(config.get("seed", 0)),
            sample=False,
            feature_set=feature_set,
        )
    else:
        seq, static, survival, accepted_len = _load_jsonl(
            args.eval_jsonl,
            max_rows=None,
            seed=int(config.get("seed", 0)),
            sample=False,
            include_cycle_history=bool(config.get("include_cycle_history", False)),
            history_cycles=int(config.get("history_cycles", 8)),
            feature_set=feature_set,
            reconstruct_target_entropy_history=bool(config.get("reconstruct_target_entropy_history", False)),
            sequence_window=config.get("sequence_window"),
        )

    seq = Normalizer(**config["seq_normalizer"]).apply(seq)
    static = Normalizer(**config["static_normalizer"]).apply(static)

    architecture = str(config.get("architecture", "gru"))
    if architecture == "gru":
        model = TemporalSurvivalHead(
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=int(config["num_slots"]),
            hidden_size=int(config.get("hidden_size", 192)),
            dropout=0.0,
        )
    else:
        model = FlatSurvivalHead(
            seq_len=int(seq.shape[1]),
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=int(config["num_slots"]),
            hidden_size=int(config.get("hidden_size", 192)),
            dropout=0.0,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = SurvivalEvalDataset(seq, static, survival, accepted_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    probs_chunks: list[torch.Tensor] = []
    accepted_chunks: list[torch.Tensor] = []
    for seq_batch, static_batch, _survival_batch, accepted_batch in loader:
        logits = model(seq_batch.to(device, non_blocking=True), static_batch.to(device, non_blocking=True))
        probs = torch.sigmoid(logits).detach().cpu()
        if args.monotonicize_probs:
            probs = _monotonicize(probs)
        probs_chunks.append(probs)
        accepted_chunks.append(accepted_batch.detach().cpu())

    probs_all = torch.cat(probs_chunks, dim=0)
    accepted_all = torch.cat(accepted_chunks, dim=0)
    max_budget = probs_all.shape[1]
    budgets = torch.tensor([arm - 1 for arm in args.arms], dtype=torch.long)
    if budgets.max().item() > max_budget:
        raise ValueError(f"arms {args.arms} require budget {budgets.max().item()}, but model predicts {max_budget} slots")

    expected_by_arm = torch.stack([probs_all[:, : int(budget.item())].sum(dim=1) for budget in budgets], dim=1)
    expected_full = expected_by_arm[:, -1]

    alpha_payload: dict[str, Any] = {}
    for alpha in args.alphas:
        ok = expected_by_arm >= (float(alpha) * expected_full[:, None])
        first_ok = ok.float().argmax(dim=1)
        no_ok = ~ok.any(dim=1)
        first_ok[no_ok] = len(args.arms) - 1
        alpha_payload[str(alpha)] = _block_stats(first_ok, accepted_all, args.arms)

    fixed_payload = {}
    for idx, arm in enumerate(args.arms):
        fixed_idx = torch.full((len(dataset),), idx, dtype=torch.long)
        fixed_payload[str(arm)] = _block_stats(fixed_idx, accepted_all, args.arms)

    smallest_sufficient = []
    for accepted in accepted_all:
        accepted_f = float(accepted.item())
        chosen = len(args.arms) - 1
        for idx, budget in enumerate(budgets.tolist()):
            if budget >= accepted_f:
                chosen = idx
                break
        smallest_sufficient.append(chosen)
    oracle_idx = torch.tensor(smallest_sufficient, dtype=torch.long)

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "eval_cache_dir": str(args.eval_cache_dir) if args.eval_cache_dir else None,
        "rows": len(dataset),
        "arms": args.arms,
        "alphas": args.alphas,
        "monotonicize_probs": args.monotonicize_probs,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "policy_by_alpha": alpha_payload,
        "fixed": fixed_payload,
        "oracle_smallest_sufficient": _block_stats(oracle_idx, accepted_all, args.arms),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
