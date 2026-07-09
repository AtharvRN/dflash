from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_predictor_head import (
    FlatSurvivalHead,
    Normalizer,
    TemporalSurvivalHead,
    TraceDataset,
    _load_jsonl,
    _metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DFlash predictor head checkpoint.")
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
    config = checkpoint["config"]

    seq, static, survival, accepted_len = _load_jsonl(
        args.eval_jsonl,
        max_rows=None,
        seed=int(config.get("seed", 0)),
        sample=False,
        include_cycle_history=bool(config.get("include_cycle_history", False)),
        history_cycles=int(config.get("history_cycles", 8)),
        feature_set=str(config.get("feature_set", "all")),
        reconstruct_target_entropy_history=bool(config.get("reconstruct_target_entropy_history", False)),
        sequence_window=config.get("sequence_window"),
    )
    seq_norm = Normalizer(**config["seq_normalizer"])
    static_norm = Normalizer(**config["static_normalizer"])
    seq = seq_norm.apply(seq)
    static = static_norm.apply(static)

    architecture = config.get("architecture", "gru")
    if architecture == "gru":
        model = TemporalSurvivalHead(
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=int(survival.shape[1]),
            hidden_size=int(config.get("hidden_size", 192)),
            dropout=0.0,
        )
    else:
        model = FlatSurvivalHead(
            seq_len=int(seq.shape[1]),
            seq_dim=int(seq.shape[-1]),
            static_dim=int(static.shape[-1]),
            num_slots=int(survival.shape[1]),
            hidden_size=int(config.get("hidden_size", 384)),
            dropout=0.0,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = TraceDataset(seq, static, survival, accepted_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    totals: dict[str, float] = {}
    count = 0
    hist_pred = np.zeros(int(survival.shape[1]) + 1, dtype=np.int64)
    hist_target = np.zeros(int(survival.shape[1]) + 1, dtype=np.int64)
    for seq_batch, static_batch, survival_batch, accepted_batch in loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        static_batch = static_batch.to(device, non_blocking=True)
        survival_batch = survival_batch.to(device, non_blocking=True)
        accepted_batch = accepted_batch.to(device, non_blocking=True)
        logits = model(seq_batch, static_batch)
        metrics = _metrics(logits, survival_batch, accepted_batch)
        batch_size = int(seq_batch.shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size
        pred_len = (torch.sigmoid(logits) >= 0.5).sum(dim=-1).cpu().numpy()
        target_len = accepted_batch.cpu().numpy().astype(np.int64)
        hist_pred += np.bincount(pred_len, minlength=len(hist_pred))[: len(hist_pred)]
        hist_target += np.bincount(target_len, minlength=len(hist_target))[: len(hist_target)]

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "rows": count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("val_metrics"),
        "metrics": {key: value / count for key, value in totals.items()},
        "pred_len_hist": hist_pred.tolist(),
        "target_len_hist": hist_target.tolist(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
