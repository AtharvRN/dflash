from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dflashv2_horizon_predictor import (
    HorizonPredictor,
    HorizonTraceDataset,
    _make_loader,
    evaluate,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in value.split(",") if x)


@torch.inference_mode()
def _oracle_metrics(loader: DataLoader, *, arms: tuple[int, ...]) -> dict[str, float | dict[str, int]]:
    accepted_chunks: list[torch.Tensor] = []
    for _, _, _, accepted_len in loader:
        accepted_chunks.append(accepted_len)
    accepted = torch.cat(accepted_chunks).float()
    full_budget = float(max(arms) - 1)
    full_accept = torch.minimum(accepted, torch.full_like(accepted, full_budget))

    smallest = torch.full_like(accepted, float(max(arms) - 1))
    chosen_blocks = torch.full_like(accepted, float(max(arms)))
    for arm in sorted(arms):
        budget = float(arm - 1)
        take = (accepted <= budget) & (chosen_blocks == float(max(arms)))
        smallest[take] = budget
        chosen_blocks[take] = float(arm)

    oracle_accept = torch.minimum(accepted, smallest)
    hist = {str(arm): int((chosen_blocks == float(arm)).sum().item()) for arm in sorted(arms)}
    return {
        "fixed_b16_mean_accepted": full_accept.mean().item(),
        "fixed_b16_accept_ratio": (full_accept / full_budget).mean().item(),
        "oracle_mean_block": chosen_blocks.mean().item(),
        "oracle_mean_budget": smallest.mean().item(),
        "oracle_mean_accepted": oracle_accept.mean().item(),
        "oracle_accept_retention": (oracle_accept.sum() / full_accept.sum().clamp_min(1)).item(),
        "oracle_accept_ratio": (oracle_accept / smallest.clamp_min(1)).mean().item(),
        "oracle_block_hist": hist,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DFlashv2 horizon checkpoint on offline trace shards."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--monotonicize-eval", action="store_true")
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [HorizonTraceDataset(path) for path in args.trace_dir]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    reference = datasets[0]
    arms = _parse_csv_ints(args.arms)
    alphas = _parse_csv_floats(args.alphas)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    has_aux_arm_head = any(
        key.startswith("aux_arm_head.") for key in checkpoint["model_state_dict"]
    )
    model = HorizonPredictor(
        input_dim=int(config["input_dim"]),
        proj_dim=int(config["proj_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_slots=reference.num_slots,
        architecture=str(config["architecture"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        context_window=reference.context_window,
        num_arms=len(arms) if has_aux_arm_head else 0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate(
        model,
        loader,
        device=device,
        monotonicize=args.monotonicize_eval,
        arms=arms,
        alphas=alphas,
        selection_min_retention=args.selection_min_retention,
    )

    oracle_loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    payload = {
        "trace_dirs": [str(path) for path in args.trace_dir],
        "checkpoint": str(args.checkpoint),
        "rows": len(dataset),
        "arms": arms,
        "alphas": alphas,
        "selection_min_retention": args.selection_min_retention,
        "metrics": metrics,
        "oracle": _oracle_metrics(oracle_loader, arms=arms),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
