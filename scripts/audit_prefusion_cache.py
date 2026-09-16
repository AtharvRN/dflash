"""Check paired-feature provenance and numerically replay the saved frozen fusion."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_context_attention import atomic_json, buffered_backup
from scripts.collect_prefusion_acceptance import sha256


def audit(cache, device="cpu"):
    info = json.loads((cache / "manifest.json").read_text())
    if info["format"] != "dflash_prefusion_cache_v1":
        raise ValueError("Wrong cache format")
    for path, expected in info["collection_config"]["hashes"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
    weights = torch.load(cache / "fusion.pt", map_location=device, weights_only=True)
    fc = weights["fc"]["weight"].float()
    norm = weights["hidden_norm"]["weight"].float()
    squared_error, squared_signal, count, worst_relative = 0.0, 0.0, 0, 0.0
    for shard in info["shards"]:
        if "path" not in shard:
            continue
        original = Path(shard["path"])
        path = cache / "shards" / original.name
        if not path.exists():
            path = original
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"Shard hash mismatch: {path}")
        with np.load(path) as data:
            for start, anchor, digest in zip(data["prefix_length"], data["anchor_id"], data["prefix_sha256"]):
                prefix = data["trajectory_token_ids"][:int(start)+1].astype(np.int64)
                if int(prefix[-1]) != int(anchor) or hashlib.sha256(prefix.tobytes()).hexdigest() != str(digest):
                    raise ValueError(f"Prefix/anchor provenance mismatch: {path}")
    for split in ("train", "val"):
        raw = np.load(cache / split / "raw_features.npy", mmap_mode="r")
        fused = np.load(cache / split / "features.npy", mmap_mode="r")
        for start in range(0, len(raw), 256):
            x = torch.from_numpy(raw[start:start+256, 0].copy()).to(device).float()
            expected = torch.from_numpy(fused[start:start+256, 0].copy()).to(device).float()
            projected = F.linear(x, fc)
            replay = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + weights["rms_norm_eps"]) * norm
            error = (replay-expected).square().sum(-1)
            signal = expected.square().sum(-1).clamp_min(1e-12)
            relative = (error/signal).sqrt()
            if not torch.isfinite(relative).all() or relative.max() > .015:
                raise ValueError("Frozen-fusion replay exceeds 1.5% per-row relative RMS tolerance")
            worst_relative = max(worst_relative, float(relative.max()))
            squared_error += float(error.sum())
            squared_signal += float(signal.sum())
            count += len(x)
    result = {"rows": count, "fusion_relative_rmse": (squared_error/squared_signal)**.5,
              "worst_row_relative_rmse": worst_relative, "per_row_tolerance": .015,
              "replay_dtype": "FP32 from saved FP16 raw features and original BF16 fusion weights",
              "stored_fused": "actual BF16 draft hidden_norm output, stored FP16",
              "source_and_shard_hashes_verified": True, "prefix_and_anchor_verified": True}
    atomic_json(cache / "audit.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--persistent-dir", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(audit(args.cache_dir, args.device)), flush=True)
    if args.persistent_dir:
        buffered_backup(args.cache_dir / "audit.json", args.persistent_dir / "audit.json")


if __name__ == "__main__":
    main()
