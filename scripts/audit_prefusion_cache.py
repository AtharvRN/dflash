"""Check paired-feature provenance and numerically replay the saved frozen fusion."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_context_attention import atomic_json, buffered_backup


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_binding(cache):
    paths = [cache / "manifest.json", cache / "fusion.pt"]
    for split in ("train", "val"):
        paths.extend(sorted((cache / split).glob("*.npy")))
    return {path.relative_to(cache).as_posix(): sha256(path) for path in paths}


def source_path(cache, original, mapping, *, source_root=None):
    if original in mapping:
        relative = Path(mapping[original])
        path = cache / relative
        if relative.is_absolute() or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError(f"Provenance mapping must stay inside cache: {original}")
        if not path.is_file():
            raise ValueError(f"Missing mapped provenance file: {path}")
        return path
    local = cache / "provenance" / Path(original).name
    if local.exists():
        return local
    path = Path(original)
    return source_root / path if source_root is not None and not path.is_absolute() else path


def validate_prefixes(data, count, path):
    fields = ("prefix_length", "anchor_id", "prefix_sha256", "cycle_id", "accepted_len")
    for name in fields:
        if name not in data or data[name].shape != (count,):
            raise ValueError(f"{name} must have exactly {count} per-row entries: {path}")
    for name in ("prefix_length", "anchor_id", "cycle_id", "accepted_len"):
        if not np.issubdtype(data[name].dtype, np.integer) or (data[name] < 0).any():
            raise ValueError(f"Invalid nonnegative integer {name}: {path}")
    if (data["accepted_len"] > 15).any():
        raise ValueError(f"accepted_len outside [0,15]: {path}")
    if "trajectory_token_ids" not in data:
        raise ValueError(f"Missing trajectory_token_ids: {path}")
    trajectory = data["trajectory_token_ids"]
    if (trajectory.ndim != 1 or not len(trajectory)
            or not np.issubdtype(trajectory.dtype, np.integer) or (trajectory < 0).any()):
        raise ValueError(f"Invalid trajectory_token_ids: {path}")
    starts, cycles, accepted = data["prefix_length"], data["cycle_id"], data["accepted_len"]
    if (starts >= len(trajectory)).any():
        raise ValueError(f"Prefix length outside trajectory bounds: {path}")
    for i in range(1, count):
        if int(cycles[i]) <= int(cycles[i-1]) or int(starts[i]) <= int(starts[i-1]):
            raise ValueError(f"Cycles and prefix lengths must be strictly increasing: {path}")
        if (int(cycles[i]) == int(cycles[i-1]) + 1
                and int(starts[i]) - int(starts[i-1]) != int(accepted[i-1]) + 1):
            raise ValueError(f"Consecutive prefix increment must equal accepted_len + 1: {path}")
    for i in range(count):
        prefix = trajectory[:int(starts[i])+1].astype(np.int64)
        digest = data["prefix_sha256"][i]
        if isinstance(digest, bytes):
            digest = digest.decode("ascii")
        if (int(prefix[-1]) != int(data["anchor_id"][i])
                or hashlib.sha256(prefix.tobytes()).hexdigest() != str(digest)):
            raise ValueError(f"Prefix/anchor provenance mismatch: {path}")


def audit(cache, device="cpu"):
    from scripts.train_prefusion_acceptance import validate_cache

    cache = Path(cache)
    binding = cache_binding(cache)
    auditor_source_sha256 = sha256(Path(__file__))
    info = json.loads((cache / "manifest.json").read_text())
    if info["format"] != "dflash_prefusion_cache_v1":
        raise ValueError("Wrong cache format")
    if not isinstance(info.get("shards"), list) or not info["shards"]:
        raise ValueError("Audit requires paired collection shards")
    config = info["collection_config"]
    if not isinstance(config.get("hashes"), dict) or not config["hashes"]:
        raise ValueError("Audit requires source hashes")
    mapping = config.get("provenance_files", {})
    for key in ("hashes", "source_sha256", "model_files_sha256"):
        signatures = config.get(key, {})
        for original, expected in signatures.items():
            path = source_path(cache, original, mapping,
                               source_root=Path(__file__).resolve().parents[1] if key == "source_sha256" else None)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Source hash mismatch or missing source: {original}")
    # Validate exact arrays, labels, cycles and groups against authenticated shards.
    validate_cache(cache)
    weights = torch.load(cache / "fusion.pt", map_location=device, weights_only=True)
    fc = weights["fc"]["weight"].float()
    norm = weights["hidden_norm"]["weight"].float()
    eps = float(weights["rms_norm_eps"])
    if (fc.shape != (info["fused_dim"], info["input_dim"])
            or norm.shape != (info["fused_dim"],) or not math.isfinite(eps) or eps <= 0
            or not torch.isfinite(fc).all() or not torch.isfinite(norm).all()):
        raise ValueError("Invalid frozen fusion weights or epsilon")
    squared_error, squared_signal, count, worst_relative = 0.0, 0.0, 0, 0.0
    for shard in info["shards"]:
        if "path" not in shard:
            continue
        original = Path(shard["path"])
        path = cache / "shards" / original.name
        if not path.exists():
            path = original if original.is_absolute() else cache / original
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"Shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            validate_prefixes(data, shard["rows"], path)
    for split in ("train", "val"):
        raw = np.load(cache / split / "raw_features.npy", mmap_mode="r")
        fused = np.load(cache / split / "features.npy", mmap_mode="r")
        for start in range(0, len(raw), 256):
            x = torch.from_numpy(raw[start:start+256, 0].copy()).to(device).float()
            expected = torch.from_numpy(fused[start:start+256, 0].copy()).to(device).float()
            projected = F.linear(x, fc)
            replay = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + eps) * norm
            error = (replay-expected).square().sum(-1)
            signal = expected.square().sum(-1).clamp_min(1e-12)
            relative = (error/signal).sqrt()
            if not torch.isfinite(relative).all() or relative.max() > .015:
                raise ValueError("Frozen-fusion replay exceeds 1.5% per-row relative RMS tolerance")
            worst_relative = max(worst_relative, float(relative.max()))
            squared_error += float(error.sum())
            squared_signal += float(signal.sum())
            count += len(x)
    if cache_binding(cache) != binding or sha256(Path(__file__)) != auditor_source_sha256:
        raise ValueError("Cache or auditor source changed during audit")
    result = {"binding": binding, "auditor_source_sha256": auditor_source_sha256,
              "rows": count, "fusion_relative_rmse": (squared_error/squared_signal)**.5,
              "worst_row_relative_rmse": worst_relative, "per_row_tolerance": .015,
              "replay_dtype": "FP32 from saved FP16 raw features and original BF16 fusion weights",
              "stored_fused": "actual BF16 draft hidden_norm output, stored FP16",
              "source_and_shard_hashes_verified": True, "prefix_and_anchor_verified": True,
              "exact_cache_shard_alignment_verified": True}
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
