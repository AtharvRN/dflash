from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_dflashv2_dspark_confidence_head import (  # noqa: E402
    DSparkConfidenceDataset,
    DSparkConfidenceHead,
    _load_shards,
    _make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DSPARK-style post-draft teacher survival curves for DFlashv2 traces."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    use_scalar_confidence = bool(ckpt_args.get("use_scalar_confidence", False))
    shards = _load_shards(args.trace_dir, require_scalar_confidence=use_scalar_confidence)
    dataset = DSparkConfidenceDataset(
        shards,
        use_scalar_confidence=use_scalar_confidence,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DSparkConfidenceHead(
        hidden_size=int(checkpoint["hidden_size"]),
        proj_dim=int(ckpt_args.get("proj_dim", 512)),
        markov_dim=int(ckpt_args.get("markov_dim", 64)),
        scalar_dim=int(checkpoint.get("scalar_dim", 0)),
        scalar_proj_dim=int(ckpt_args.get("scalar_proj_dim", 32)),
        head_hidden_size=int(ckpt_args.get("head_hidden_size", 512)),
        vocab_size=int(ckpt_args.get("vocab_size", 200000)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    survival_out = np.lib.format.open_memmap(
        args.output_dir / "teacher_survival.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), dataset.num_slots),
    )
    conditional_out = np.lib.format.open_memmap(
        args.output_dir / "teacher_conditional.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), dataset.num_slots),
    )
    loader = _make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    offset = 0
    for hidden, prev_ids, scalar, _survival, _accepted_len in loader:
        hidden = hidden.to(device, non_blocking=True)
        prev_ids = prev_ids.to(device, non_blocking=True)
        scalar = scalar.to(device, non_blocking=True)
        logits = model(hidden, prev_ids, scalar)
        conditional = torch.sigmoid(logits)
        survival = torch.cumprod(conditional, dim=1)
        batch = int(hidden.shape[0])
        conditional_out[offset : offset + batch] = conditional.cpu().numpy().astype(np.float32)
        survival_out[offset : offset + batch] = survival.cpu().numpy().astype(np.float32)
        offset += batch
        if offset == batch or offset % max(args.batch_size * 10, 1) == 0 or offset == len(dataset):
            print(json.dumps({"event": "export_progress", "rows": offset, "total": len(dataset)}), flush=True)
    survival_out.flush()
    conditional_out.flush()

    meta = {
        "trace_dirs": [str(path) for path in args.trace_dir],
        "checkpoint": str(args.checkpoint),
        "rows": len(dataset),
        "num_slots": dataset.num_slots,
        "use_scalar_confidence": use_scalar_confidence,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
