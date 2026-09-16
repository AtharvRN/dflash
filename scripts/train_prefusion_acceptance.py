"""Train matched raw/fused last-MLP acceptance heads from a prefusion cache.

Only cached features are used; no target or draft model is loaded. Policy
metrics are clipped full-draft offline proxies, not short-draft measurements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_context_attention import (
    acceptance_nll, acceptance_survival, atomic_json, buffered_backup,
    calibrate, metrics, predict, proxy_metrics,
)
from scripts.train_dflashv2_horizon_predictor import HorizonPredictor


MODEL_NAMES = ("raw", "fused", "fused_parameter_matched")
BASE_PROJ_DIM = 512
HIDDEN_SIZE = 256


def parameter_matched_proj_dim(raw_dim, fused_dim, raw_proj_dim=BASE_PROJ_DIM):
    """Nearest integer width including projection bias, LayerNorm and head.

    With fixed downstream widths, P(D, p) = p * (D + 3 + 256) + C.
    Thus the constant head parameters cancel when matching counts.
    """
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1
           for v in (raw_dim, fused_dim, raw_proj_dim)):
        raise ValueError("Feature and projection dimensions must be positive integers")
    numerator = raw_proj_dim * (raw_dim + 3 + HIDDEN_SIZE)
    denominator = fused_dim + 3 + HIDDEN_SIZE
    lower = max(1, numerator // denominator)
    return min((lower, lower + 1), key=lambda p: (abs(p * denominator - numerator), p))


def model_spec(name, info, dropout):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name}")
    return {
        "input_dim": info["input_dim"] if name == "raw" else info["fused_dim"],
        "proj_dim": parameter_matched_proj_dim(info["input_dim"], info["fused_dim"])
                    if name == "fused_parameter_matched" else BASE_PROJ_DIM,
        "hidden_size": HIDDEN_SIZE, "num_slots": 15, "architecture": "last_mlp",
        "num_layers": 1, "dropout": dropout, "context_window": 1,
    }


def make_model(name, info, dropout=.05):
    return HorizonPredictor(**model_spec(name, info, dropout))


def validate_shard_alignment(cache_dir, info, indices, calibration):
    """Audit paired collection shards when the collector's receipts are present.

    Prefer cache-local shards so a PVC copy remains independently auditable
    even when the original absolute collection path no longer exists.
    """
    if "shards" not in info:
        return
    offsets = {"train": 0, "val": 0}
    arrays = {split: {name: np.load(cache_dir / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                      for name in ("raw_features", "features", "accepted_len")}
              for split in offsets}
    selected, content_groups = {}, {}
    for row in info.get("collection_config", {}).get("selected_prompts", []):
        pid, group = row["prompt_id"], row["group"]
        if pid in selected or group not in ("train", "calibration", "assessment"):
            raise ValueError("Invalid selected prompt metadata")
        selected[pid] = group
        if "content_sha256" in row:
            content_groups.setdefault(row["content_sha256"], set()).add(group)
    if any(len(groups) > 1 for groups in content_groups.values()):
        raise ValueError("Selected prompt content hash crosses experiment groups")
    seen_prompts = set()
    for record in info["shards"]:
        pid, group = record["prompt_id"], record["group"]
        if pid in seen_prompts or group not in ("train", "calibration", "assessment"):
            raise ValueError("Invalid shard prompt/group metadata")
        seen_prompts.add(pid)
        if selected and selected.get(pid) != group:
            raise ValueError("Shard group disagrees with selected prompt metadata")
        if "path" not in record:
            if record.get("rows", 0):
                raise ValueError("Nonempty shard record missing path")
            continue
        recorded_path = Path(record["path"])
        local_path = cache_dir / "shards" / recorded_path.name
        path = local_path if local_path.exists() else (
            recorded_path if recorded_path.is_absolute() else cache_dir / recorded_path)
        if not path.is_file():
            raise ValueError(f"Missing paired collection shard: {path}")
        if sha256(path) != record.get("sha256"):
            raise ValueError(f"Shard SHA-256 mismatch: {path}")
        receipt = path.with_suffix(".json")
        if receipt.exists() and json.loads(receipt.read_text()) != record:
            raise ValueError(f"Shard receipt metadata mismatch: {receipt}")
        count = record.get("rows")
        if type(count) is not int or count < 1:
            raise ValueError("Invalid shard row count")
        split = "train" if group == "train" else "val"
        start, stop = offsets[split], offsets[split] + count
        if stop > len(indices[split]):
            raise ValueError("Shard rows exceed materialized cache")
        rows = indices[split][start:stop]
        if not (rows[:, 2] == pid).all() or not np.array_equal(rows[:, 1], np.arange(start, stop)):
            raise ValueError("Shard/cache row identity alignment mismatch")
        if split == "val" and not (calibration[start:stop] == (group == "calibration")).all():
            raise ValueError("Shard/cache calibration group alignment mismatch")
        with np.load(path, allow_pickle=False) as shard:
            for name, cached in arrays[split].items():
                if name not in shard:
                    raise ValueError(f"Shard missing {name}: {path}")
                source = shard[name]
                if source.dtype != cached.dtype or not np.array_equal(source, cached[start:stop]):
                    raise ValueError(f"Shard/cache {name} alignment mismatch: {path}")
            if "cycle_id" not in shard or not np.array_equal(shard["cycle_id"], rows[:, 3]):
                raise ValueError(f"Shard/cache cycle alignment mismatch: {path}")
        offsets[split] = stop
    if selected and seen_prompts != set(selected):
        raise ValueError("Shard records do not cover selected prompt metadata")
    if any(offsets[split] != len(indices[split]) for split in offsets):
        raise ValueError("Shard rows do not cover materialized cache")


def validate_cache(cache_dir):
    """Validate shared row alignment and prompt-level train/cal/assessment splits.

    A single row_index is the cache's identity contract for both feature arrays.
    With shard receipts, both feature arrays are additionally compared against
    hash-verified paired source shards. Minimal caches without receipts permit
    structural/label checks only, not detection of arbitrary feature permutations.
    """
    cache_dir = Path(cache_dir)
    info = json.loads((cache_dir / "manifest.json").read_text())
    if info.get("format") != "dflash_prefusion_cache_v1":
        raise ValueError("Not a dflash_prefusion_cache_v1 cache")
    for key in ("input_dim", "fused_dim", "num_slots", "context_window"):
        value = info.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid manifest {key}")
    if info["num_slots"] != 15 or info["context_window"] != 1:
        raise ValueError("Expected num_slots=15 and context_window=1")
    indices = {}
    for split in ("train", "val"):
        path = cache_dir / split
        labels = np.load(path / "accepted_len.npy", mmap_mode="r", allow_pickle=False)
        if labels.ndim != 1 or labels.dtype != np.int64 or not len(labels):
            raise ValueError(f"{split}: accepted_len must be nonempty int64 [N]")
        n = len(labels)
        if ((labels < 0) | (labels > 15)).any():
            raise ValueError(f"{split}: accepted_len outside [0,15]")
        if f"{split}_rows" in info and info[f"{split}_rows"] != n:
            raise ValueError(f"{split}: manifest row count mismatch")
        for filename, width in (("raw_features", info["input_dim"]), ("features", info["fused_dim"])):
            features = np.load(path / f"{filename}.npy", mmap_mode="r", allow_pickle=False)
            if features.shape != (n, 1, width) or features.dtype != np.float16:
                raise ValueError(f"{split}: {filename} must be float16 [{n},1,{width}]")
            for start in range(0, n, 1024):
                if not np.isfinite(features[start:start + 1024]).all():
                    raise ValueError(f"{split}: {filename} contains nonfinite values")
        mask = np.load(path / "mask.npy", mmap_mode="r", allow_pickle=False)
        if mask.shape != (n, 1) or not (mask == 1).all():
            raise ValueError(f"{split}: mask must be all ones [N,1]")
        rows = np.load(path / "row_index.npy", allow_pickle=False)
        if rows.shape != (n, 5) or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"{split}: row_index must be integer [N,5]")
        if (rows < 0).any() or not (rows[:, 0] == 0).all():
            raise ValueError(f"{split}: invalid row_index identity")
        if not np.array_equal(rows[:, 4], labels):
            raise ValueError(f"{split}: row_index/accepted_len alignment mismatch")
        if len(np.unique(rows[:, 1])) != n or len(np.unique(rows[:, 2:4], axis=0)) != n:
            raise ValueError(f"{split}: duplicate row or prompt/cycle identity")
        indices[split] = rows
    calibration = np.load(cache_dir / "val" / "calibration.npy", allow_pickle=False)
    if calibration.dtype != np.bool_ or calibration.shape != (len(indices["val"]),):
        raise ValueError("calibration must be bool [N_val]")
    if not calibration.any() or calibration.all():
        raise ValueError("Need nonempty calibration and assessment partitions")
    train_prompts = set(indices["train"][:, 2])
    cal_prompts = set(indices["val"][calibration, 2])
    assessment_prompts = set(indices["val"][~calibration, 2])
    if train_prompts & (cal_prompts | assessment_prompts):
        raise ValueError("Training/validation prompt leakage")
    if cal_prompts & assessment_prompts:
        raise ValueError("Calibration/assessment prompt leakage")
    if not np.load(cache_dir / "val" / "accepted_len.npy")[calibration].sum():
        raise ValueError("Calibration requires positive accepted length for retention")
    validate_shard_alignment(cache_dir, info, indices, calibration)
    return info, calibration


class PrefusionDataset(Dataset):
    def __init__(self, path, feature_file, indices=None):
        if feature_file not in ("raw_features.npy", "features.npy"):
            raise ValueError("Unknown feature file")
        self.features = np.load(Path(path) / feature_file, mmap_mode="r", allow_pickle=False)
        self.mask = np.load(Path(path) / "mask.npy", mmap_mode="r", allow_pickle=False)
        self.accepted = np.load(Path(path) / "accepted_len.npy", mmap_mode="r", allow_pickle=False)
        self.indices = np.arange(len(self.accepted)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return (torch.from_numpy(self.features[i].copy()), torch.from_numpy(self.mask[i].copy()),
                torch.tensor(int(self.accepted[i]), dtype=torch.long))


def make_loader(dataset, args, device, *, training=False):
    # An independent, identically seeded generator per model keeps shuffles
    # independent of model initialization, dropout and validation iteration.
    return DataLoader(dataset, batch_size=args.batch_size if training else args.eval_batch_size,
                      shuffle=training, generator=torch.Generator().manual_seed(args.seed),
                      num_workers=args.workers, persistent_workers=args.workers > 0,
                      pin_memory=device.type == "cuda")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", "--cache", type=Path, required=True)
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--persistent-dir", "--persistent", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--retention", type=float, default=.96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", "--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.eval_batch_size, args.cpu_threads) < 1
            or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 <= args.dropout < 1 or not 0 < args.retention <= 1 or args.seed < 0):
        parser.error("Invalid training hyperparameters")
    if args.persistent_dir and args.persistent_dir.resolve() == args.output_dir.resolve():
        parser.error("Persistent and output directories must differ")
    return args


def main(argv=None):
    args = parse_args(argv)
    info, calibration_mask = validate_cache(args.cache_dir)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository,
                                             text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "cache_manifest": info, "cache_manifest_sha256": sha256(args.cache_dir / "manifest.json"),
        "cache_validation": {"structure_labels_prompt_groups_checked": True,
                             "paired_shards_hash_and_alignment_checked": "shards" in info,
                             "external_collection_inputs_rehashed": False},
        "git_commit": git_commit, "torch": str(torch.__version__), "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "models": {name: model_spec(name, info, args.dropout) for name in MODEL_NAMES},
        "source_sha256": {str(path.relative_to(repository)): sha256(path) for path in (
            Path(__file__).resolve(), repository / "scripts/train_context_attention.py",
            repository / "scripts/train_dflashv2_horizon_predictor.py",
            repository / "dflash/context_attention.py")},
        "cache_alignment_sha256": {f"{split}/{file}.npy": sha256(args.cache_dir / split / f"{file}.npy")
                                   for split in ("train", "val")
                                   for file in ("row_index", "accepted_len", "mask")},
        "calibration_sha256": sha256(args.cache_dir / "val/calibration.npy"),
        "calibration_rows": int(calibration_mask.sum()),
        "assessment_rows": int((~calibration_mask).sum()),
        "objective": "mean per-example first-rejection NLL",
        "selection_metric": "calibration aggregate_accept_ratio; first epoch wins ties",
        "calibration_rule": "minimum mean budget satisfying calibration retention on alpha grid [.5,1]",
        "selection_caveat": "Checkpoint and alpha selection share the calibration set; calibration metrics "
                            "are selection-biased. Only the disjoint assessment prompts are held out.",
        "collection_scope": "Fresh paired same-forward pilot subset preserving historical train/validation "
                            "and calibration prompt groups; not the full historical state distribution.",
        "optimizer": {"name": "AdamW", "weight_decay": .01, "gradient_clip": 1.0},
        "policy_metrics": "clipped B16 offline proxy, NOT actual short-draft acceptance",
        "target_or_draft_loaded": False, "anchor_used": False,
        "train_order": "identically seeded independent DataLoader generators for each model",
        "evaluation_order": "cache row order; calibration preserves relative row order",
    })
    atomic_json(args.output_dir / "config.json", config)
    results, futures = {}, []
    with ThreadPoolExecutor(max_workers=1) as pool:
        def backup(path, relative=None):
            if args.persistent_dir:
                destination = args.persistent_dir / (relative or path.relative_to(args.output_dir))
                futures.append(pool.submit(buffered_backup, path, destination))

        def check_backups():
            for future in futures:
                if future.done():
                    future.result()

        backup(args.output_dir / "config.json")
        for file in ("row_index", "accepted_len", "calibration"):
            path = args.output_dir / f"validation_{file}.npy"
            buffered_backup(args.cache_dir / "val" / f"{file}.npy", path)
            backup(path)
        for name in MODEL_NAMES:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed % (2**32))
            random.seed(args.seed)
            feature_file = "raw_features.npy" if name == "raw" else "features.npy"
            train_loader = make_loader(PrefusionDataset(args.cache_dir / "train", feature_file),
                                       args, device, training=True)
            val_loader = make_loader(PrefusionDataset(args.cache_dir / "val", feature_file), args, device)
            cal_loader = make_loader(PrefusionDataset(args.cache_dir / "val", feature_file,
                                                     np.flatnonzero(calibration_mask)), args, device)
            model = make_model(name, info, args.dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            directory = args.output_dir / name
            directory.mkdir()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            best_score, best_path, best_epoch = -float("inf"), None, None
            history, started = [], time.monotonic()
            for epoch in range(1, args.epochs + 1):
                check_backups()
                model.train()
                total_nll, total_mae, samples = 0.0, 0.0, 0
                for features, mask, labels in train_loader:
                    features = features.to(device, non_blocking=True)
                    mask, labels = mask.to(device, non_blocking=True), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        logits = model(features, mask)
                    loss = acceptance_nll(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    total_nll += float(loss.detach()) * len(labels)
                    expected = acceptance_survival(logits.detach()).sum(-1)
                    total_mae += float((expected - labels).abs().sum())
                    samples += len(labels)
                prediction, accepted, cal_nll = predict(model, cal_loader, device, amp)
                cal_metrics = metrics(prediction, accepted)
                policy = calibrate(prediction, accepted, args.retention)
                score = policy["aggregate_accept_ratio"]
                record = {"model": name, "epoch": epoch, "train_nll": total_nll / samples,
                          "train_mae": total_mae / samples, "calibration_nll": cal_nll,
                          "calibration_mae": cal_metrics["expected_mae"],
                          "calibration_proxy_policy": policy, "selection_score": score,
                          "elapsed_s": time.monotonic() - started}
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    best_path = directory / f"best_epoch_{epoch}.pt"
                    temporary = best_path.with_suffix(".pt.tmp")
                    torch.save({"model": model.state_dict(), "architecture": "last_mlp", "variant": name,
                                "epoch": epoch, "model_config": model_spec(name, info, args.dropout),
                                "run_config": config, "parameters": parameters,
                                "selection_score": score, "calibration_mae": cal_metrics["expected_mae"],
                                "calibration_proxy_policy": policy}, temporary)
                    temporary.replace(best_path)
                    backup(best_path)
                record["best_epoch"] = best_epoch
                history.append(record)
                atomic_json(directory / "history.json", history)
                # Async copies read immutable files, including rolling history/summary snapshots.
                snapshot = directory / f"history_epoch_{epoch}.json"
                atomic_json(snapshot, history)
                backup(snapshot)
                backup(snapshot, Path(name) / "history.json")
                print(json.dumps(record), flush=True)
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            selected = checkpoint["calibration_proxy_policy"]
            del checkpoint
            prediction, accepted, val_nll = predict(model, val_loader, device, amp)
            cal = torch.from_numpy(calibration_mask)
            result = {
                "parameters": parameters, "trainable_parameters": parameters,
                "model_config": model_spec(name, info, args.dropout), "feature_file": feature_file,
                "best_epoch": best_epoch, "best_checkpoint": best_path.name, "selection_score": best_score,
                "full_validation": metrics(prediction, accepted), "full_validation_nll": val_nll,
                "calibration": metrics(prediction[cal], accepted[cal]),
                "assessment": metrics(prediction[~cal], accepted[~cal]),
                "calibration_proxy_policy": selected,
                "assessment_proxy_policy": proxy_metrics(prediction[~cal], accepted[~cal], selected["alpha"]),
                "selection_caveat": config["selection_caveat"],
                "collection_scope": config["collection_scope"],
                "elapsed_s": time.monotonic() - started,
            }
            np.save(directory / "validation_survival.npy", prediction.numpy())
            atomic_json(directory / "result.json", result)
            results[name] = result
            atomic_json(args.output_dir / "summary.json", results)
            snapshot = args.output_dir / f"summary_after_{name}.json"
            atomic_json(snapshot, results)
            backup(snapshot)
            backup(snapshot, Path("summary.json"))
            backup(directory / "validation_survival.npy")
            backup(directory / "result.json")
            print(json.dumps({"completed": name, "result": result}), flush=True)
            del model, optimizer, train_loader, val_loader, cal_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
