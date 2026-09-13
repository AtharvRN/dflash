"""Matched last-MLP / one-query / per-position-query acceptance ablation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.context_attention import ContextAcceptancePredictor, acceptance_nll, acceptance_survival, choose_budget
from scripts.train_dflashv2_horizon_predictor import HorizonPredictor
from scripts.prepare_context_attention_cache import validate_context_masks, validate_feature_kind


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class ContextDataset(Dataset):
    def __init__(self, path, indices=None):
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.mask = np.load(path / "mask.npy", mmap_mode="r")
        self.accepted = np.load(path / "accepted_len.npy", mmap_mode="r")
        self.indices = np.arange(len(self.accepted)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return (torch.from_numpy(self.features[i].copy()), torch.from_numpy(self.mask[i].copy()),
                torch.tensor(int(self.accepted[i]), dtype=torch.long))


def make_model(name, info, args):
    if name == "last_mlp":
        return HorizonPredictor(input_dim=info["input_dim"], proj_dim=512, hidden_size=256,
                                num_slots=info["num_slots"], architecture="last_mlp", num_layers=1,
                                dropout=args.dropout, context_window=info["context_window"])
    if name not in ("one_query", "position_queries"):
        raise ValueError(f"Unknown architecture {name}")
    return ContextAcceptancePredictor(input_dim=info["input_dim"], num_slots=info["num_slots"],
             context_window=info["context_window"], num_queries=1 if name == "one_query" else info["num_slots"],
             num_layers=args.layers, num_heads=args.heads, ff_width=args.ff_width, dropout=args.dropout)


def metrics(survival, accepted):
    prediction = survival.sum(-1)
    absolute = (prediction-accepted).abs()
    by_length = {}
    for length in range(survival.shape[1]+1):
        keep = accepted == length
        by_length[str(length)] = {"rows": int(keep.sum()),
                                  "mae": float(absolute[keep].mean()) if keep.any() else None}
    return {"rows": len(accepted), "expected_mae": float(absolute.mean()),
            "median_mae": float(((survival >= .5).sum(-1)-accepted).abs().float().mean()),
            "mean_expected": float(prediction.mean()), "mean_accepted": float(accepted.float().mean()),
            "brier": float(((survival-(accepted[:, None] >= torch.arange(1, survival.shape[1]+1)).float())**2).mean()),
            "by_length": by_length}


def proxy_metrics(survival, accepted, alpha):
    budget = choose_budget(survival, alpha)
    retained = torch.minimum(accepted, budget).float()
    denom = float(accepted.sum())
    return {"alpha": float(alpha), "mean_budget": float(budget.float().mean()),
            "mean_accepted": float(retained.mean()),
            "retention": float(retained.sum()/denom) if denom > 0 else None,
            "aggregate_accept_ratio": float(retained.sum()/budget.sum()),
            "mean_cycle_accept_ratio": float((retained/budget).mean())}


def calibrate(survival, accepted, minimum_retention):
    candidates = [proxy_metrics(survival, accepted, a) for a in np.linspace(.5, 1, 101)]
    feasible = [r for r in candidates if r["retention"] is not None and r["retention"] >= minimum_retention]
    if not feasible:
        raise ValueError("No feasible calibration point")
    return min(feasible, key=lambda r: (r["mean_budget"], -r["retention"]))


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval()
    all_survival, all_y, nll = [], [], 0.0
    for features, mask, y in loader:
        features, mask, y = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(features, mask)
        loss = acceptance_nll(logits, y, "sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        nll += float(loss)
        all_survival.append(acceptance_survival(logits).cpu())
        all_y.append(y.cpu())
    return torch.cat(all_survival), torch.cat(all_y), nll/len(loader.dataset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--persistent-dir", type=Path)
    p.add_argument("--models", nargs="+", choices=["last_mlp", "one_query", "position_queries"],
                   default=["last_mlp", "one_query", "position_queries"])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ff-width", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--length-weight", type=float, default=0.0)
    p.add_argument("--retention", type=float, default=.96)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or not 0 < args.retention <= 1:
        raise ValueError("Invalid epoch/batch/retention settings")
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    info = json.loads((args.cache_dir / "manifest.json").read_text())
    if info["format"] != "dflash_context_attention_cache_v1" or info["input_kind"] != "predraft_fused":
        raise ValueError("Not a compatible pre-draft cache")
    if "trace_manifest" in info:
        validate_feature_kind(json.loads(Path(info["trace_manifest"]).read_text()))
    for split in ("train", "val"):
        validate_context_masks(np.load(args.cache_dir / split / "mask.npy"))
    train_index = np.load(args.cache_dir / "train" / "row_index.npy")
    val_index = np.load(args.cache_dir / "val" / "row_index.npy")
    calibration_mask = np.load(args.cache_dir / "val" / "calibration.npy").astype(bool)
    if set(train_index[:, 2]) & set(val_index[:, 2]):
        raise ValueError("Training/validation prompt leakage")
    if len(calibration_mask) != len(val_index) or not calibration_mask.any() or calibration_mask.all():
        raise ValueError("Need disjoint calibration and assessment rows")
    if set(val_index[calibration_mask, 2]) & set(val_index[~calibration_mask, 2]):
        raise ValueError("Calibration/assessment prompt leakage")
    train = ContextDataset(args.cache_dir / "train")
    val = ContextDataset(args.cache_dir / "val")
    calibration = ContextDataset(args.cache_dir / "val", np.flatnonzero(calibration_mask))
    def loader(ds, *, train_mode=False):
        return DataLoader(ds, batch_size=args.batch_size if train_mode else args.eval_batch_size,
                          shuffle=train_mode, generator=torch.Generator().manual_seed(args.seed),
                          num_workers=args.workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)
    val_loader, cal_loader = loader(val), loader(calibration)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({"cache_manifest": info, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "objective": "mean per-example first-rejection NLL", "anchor_used": False,
                   "policy_metrics": "clipped B16 offline proxy, NOT actual short-draft acceptance",
                   "calibration_rows": int(calibration_mask.sum()), "assessment_rows": int((~calibration_mask).sum())})
    atomic_json(args.output_dir / "config.json", config)
    backup_pool = ThreadPoolExecutor(max_workers=1)
    futures = []
    def backup(paths):
        if args.persistent_dir:
            for path in paths:
                destination = args.persistent_dir / path.relative_to(args.output_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(destination.suffix+".tmp")
                shutil.copy2(path, temp)
                temp.replace(destination)
    futures.append(backup_pool.submit(backup, [args.output_dir / "config.json"]))
    results = {}
    try:
        for name in args.models:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            model = make_model(name, info, args).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            train_loader = loader(train, train_mode=True)
            directory = args.output_dir / name
            directory.mkdir()
            best_mae, history, best_path = float("inf"), [], None
            started = time.monotonic()
            parameters = sum(p.numel() for p in model.parameters())
            for epoch in range(1, args.epochs+1):
                model.train()
                total_nll, total_mae, samples = 0.0, 0.0, 0
                for features, mask, y in train_loader:
                    features, mask, y = features.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        logits = model(features, mask)
                    nll = acceptance_nll(logits, y)
                    expected = acceptance_survival(logits).sum(-1)
                    loss = nll + args.length_weight * F.smooth_l1_loss(expected, y.float())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    total_nll += float(nll.detach()) * len(y)
                    total_mae += float((expected.detach()-y).abs().sum())
                    samples += len(y)
                pred, labels, cal_nll = predict(model, cal_loader, device, amp)
                cal_metrics = metrics(pred, labels)
                epoch_record = {"model": name, "epoch": epoch, "train_nll": total_nll/samples,
                                "train_mae": total_mae/samples, "calibration_nll": cal_nll,
                                "calibration_mae": cal_metrics["expected_mae"],
                                "elapsed_s": time.monotonic()-started}
                history.append(epoch_record)
                paths = []
                if cal_metrics["expected_mae"] < best_mae:
                    best_mae = cal_metrics["expected_mae"]
                    best_path = directory / f"best_epoch_{epoch}.pt"
                    torch.save({"model": model.state_dict(), "architecture": name, "epoch": epoch,
                                "model_config": config, "calibration_mae": best_mae}, best_path)
                    paths.append(best_path)
                atomic_json(directory / "history.json", history)
                # Immutable snapshot prevents a racing background copy of an overwritten history file.
                atomic_json(directory / f"epoch_{epoch}.json", epoch_record)
                paths.append(directory / f"epoch_{epoch}.json")
                futures.append(backup_pool.submit(backup, paths))
                print(json.dumps(epoch_record), flush=True)
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            prediction, accepted, val_nll = predict(model, val_loader, device, amp)
            cal = torch.from_numpy(calibration_mask)
            selected = calibrate(prediction[cal], accepted[cal], args.retention)
            result = {"parameters": parameters, "best_epoch": int(best_path.stem.split("_")[-1]),
                      "best_checkpoint": best_path.name, "full_validation": metrics(prediction, accepted),
                      "full_validation_nll": val_nll,
                      "assessment": metrics(prediction[~cal], accepted[~cal]),
                      "calibration_proxy_policy": selected,
                      "assessment_proxy_policy": proxy_metrics(prediction[~cal], accepted[~cal], selected["alpha"]),
                      "elapsed_s": time.monotonic()-started}
            np.save(directory / "validation_survival.npy", prediction.numpy())
            atomic_json(directory / "result.json", result)
            results[name] = result
            atomic_json(args.output_dir / "summary.json", results)
            futures.append(backup_pool.submit(backup, [directory / "result.json", directory / "history.json",
                                                       directory / "validation_survival.npy"]))
            print(json.dumps({"completed": name, "result": result}), flush=True)
            del model, optimizer, train_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        futures.append(backup_pool.submit(backup, [args.output_dir / "summary.json"]))
        for future in futures:
            future.result()
    finally:
        backup_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
