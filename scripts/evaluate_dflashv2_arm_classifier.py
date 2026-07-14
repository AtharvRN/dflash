from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dflashv2_arm_classifier import (  # noqa: E402
    ArmClassifier,
    DFlashV2ArmDataset,
    _load_shards,
    _make_loader,
    evaluate,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in value.split(",") if x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DFlashv2 direct arm classifier checkpoint.")
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--arms", type=_parse_ints, default=(4, 8, 12, 16))
    parser.add_argument("--risk-thresholds", type=_parse_floats, default=(0.80, 0.85, 0.90, 0.95))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    arms = tuple(sorted(args.arms))
    shards = _load_shards(args.trace_dir)
    total_rows = sum(shard.rows for shard in shards)
    indices = torch.arange(total_rows, dtype=torch.long).numpy()
    last_only = bool(config.get("last_only", False)) or str(config.get("architecture")) == "last_mlp"
    dataset = DFlashV2ArmDataset(shards, indices, arms=arms, last_only=last_only)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ArmClassifier(
        input_dim=int(config["input_dim"]),
        proj_dim=int(config["proj_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_arms=len(arms),
        architecture=str(config["architecture"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        context_window=dataset.context_window,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader: DataLoader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate(
        model,
        loader,
        device=device,
        arms=arms,
        risk_thresholds=tuple(args.risk_thresholds),
    )
    payload = {
        "trace_dirs": [str(path) for path in args.trace_dir],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rows": len(dataset),
        "arms": arms,
        "risk_thresholds": tuple(args.risk_thresholds),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
