from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_dflashv2_integer_horizon import (
    CompactIntegerHorizonDataset,
    IntegerHorizonHead,
    TraceIntegerHorizonDataset,
    _batch_to_device,
    _loss_and_predictions,
    _make_loader,
)


def _make_dataset(path: Path, *, kind: str, max_rows: int | None) -> Any:
    if kind == "compact":
        return CompactIntegerHorizonDataset(path, max_rows=max_rows)
    if kind == "trace":
        return TraceIntegerHorizonDataset([path], max_rows=max_rows, last_only=True)
    raise ValueError(f"unsupported dataset kind: {kind}")


def _load_model(checkpoint_path: Path, reference: Any, device: torch.device) -> tuple[IntegerHorizonHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = IntegerHorizonHead(
        input_dim=int(reference.hidden_size),
        hidden_size=int(config.get("hidden_size", 512)),
        proj_dim=int(config.get("proj_dim", 512)),
        num_classes=int(reference.num_classes),
        dropout=float(config.get("dropout", 0.1)),
        use_token_tower=bool(config.get("use_token_tower", False)),
        vocab_size=int(config.get("vocab_size", 200000)),
        token_embed_dim=int(config.get("token_embed_dim", 128)),
        token_hidden_size=int(config.get("token_hidden_size", 256)),
        token_encoder=str(config.get("token_encoder", "gru")),
        token_window=int(reference.token_window),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _empty_accumulators(num_classes: int) -> dict[str, torch.Tensor]:
    return {
        "count": torch.zeros(num_classes, dtype=torch.float64),
        "expected_abs": torch.zeros(num_classes, dtype=torch.float64),
        "rounded_abs": torch.zeros(num_classes, dtype=torch.float64),
        "argmax_abs": torch.zeros(num_classes, dtype=torch.float64),
        "expected_sum": torch.zeros(num_classes, dtype=torch.float64),
        "rounded_sum": torch.zeros(num_classes, dtype=torch.float64),
        "argmax_sum": torch.zeros(num_classes, dtype=torch.float64),
        "expected_exact": torch.zeros(num_classes, dtype=torch.float64),
        "rounded_exact": torch.zeros(num_classes, dtype=torch.float64),
        "argmax_exact": torch.zeros(num_classes, dtype=torch.float64),
    }


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1.0)


@torch.inference_mode()
def evaluate_split(
    *,
    name: str,
    dataset: Any,
    model: IntegerHorizonHead,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    loss_args: dict[str, float],
) -> dict[str, Any]:
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = int(dataset.num_classes)
    acc = _empty_accumulators(num_classes)
    totals = {
        "rows": 0.0,
        "expected_abs": 0.0,
        "rounded_abs": 0.0,
        "argmax_abs": 0.0,
        "expected_sq": 0.0,
        "rounded_exact": 0.0,
        "argmax_exact": 0.0,
        "expected_sum": 0.0,
        "rounded_sum": 0.0,
        "argmax_sum": 0.0,
        "target_sum": 0.0,
    }
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        logits = model(batch.features, batch.mask, batch.token_ids, batch.token_mask)
        _loss, preds = _loss_and_predictions(logits, batch.accepted_len, **loss_args)
        target = batch.accepted_len.long().clamp(min=0, max=num_classes - 1).detach().cpu()
        expected = preds["expected_len"].detach().cpu().double()
        rounded = preds["rounded_len"].detach().cpu().double()
        argmax = preds["argmax_len"].detach().cpu().double()
        target_f = target.double()
        expected_abs = (expected - target_f).abs()
        rounded_abs = (rounded - target_f).abs()
        argmax_abs = (argmax - target_f).abs()
        expected_exact = (expected.round().clamp(0, num_classes - 1) == target_f).double()
        rounded_exact = (rounded == target_f).double()
        argmax_exact = (argmax == target_f).double()

        rows = int(target.numel())
        totals["rows"] += rows
        totals["expected_abs"] += float(expected_abs.sum().item())
        totals["rounded_abs"] += float(rounded_abs.sum().item())
        totals["argmax_abs"] += float(argmax_abs.sum().item())
        totals["expected_sq"] += float(((expected - target_f) ** 2).sum().item())
        totals["rounded_exact"] += float(rounded_exact.sum().item())
        totals["argmax_exact"] += float(argmax_exact.sum().item())
        totals["expected_sum"] += float(expected.sum().item())
        totals["rounded_sum"] += float(rounded.sum().item())
        totals["argmax_sum"] += float(argmax.sum().item())
        totals["target_sum"] += float(target_f.sum().item())

        for key, values in (
            ("count", torch.ones_like(target_f)),
            ("expected_abs", expected_abs),
            ("rounded_abs", rounded_abs),
            ("argmax_abs", argmax_abs),
            ("expected_sum", expected),
            ("rounded_sum", rounded),
            ("argmax_sum", argmax),
            ("expected_exact", expected_exact),
            ("rounded_exact", rounded_exact),
            ("argmax_exact", argmax_exact),
        ):
            acc[key] += torch.bincount(target, weights=values, minlength=num_classes).double()

    rows = max(totals["rows"], 1.0)
    count = acc["count"]
    per_horizon = []
    for h in range(num_classes):
        denom = max(float(count[h].item()), 1.0)
        per_horizon.append(
            {
                "horizon": h,
                "count": int(count[h].item()),
                "target_prob": float(count[h].item() / rows),
                "expected_mae": float(acc["expected_abs"][h].item() / denom),
                "rounded_mae": float(acc["rounded_abs"][h].item() / denom),
                "argmax_mae": float(acc["argmax_abs"][h].item() / denom),
                "expected_exact": float(acc["expected_exact"][h].item() / denom),
                "rounded_exact": float(acc["rounded_exact"][h].item() / denom),
                "argmax_exact": float(acc["argmax_exact"][h].item() / denom),
                "mean_expected": float(acc["expected_sum"][h].item() / denom),
                "mean_rounded": float(acc["rounded_sum"][h].item() / denom),
                "mean_argmax": float(acc["argmax_sum"][h].item() / denom),
                "expected_bias": float(acc["expected_sum"][h].item() / denom - h),
                "rounded_bias": float(acc["rounded_sum"][h].item() / denom - h),
            }
        )
    return {
        "name": name,
        "rows": int(totals["rows"]),
        "metrics": {
            "expected_len_mae": totals["expected_abs"] / rows,
            "expected_len_rmse": math.sqrt(totals["expected_sq"] / rows),
            "rounded_expected_len_mae": totals["rounded_abs"] / rows,
            "rounded_expected_len_exact": totals["rounded_exact"] / rows,
            "argmax_len_mae": totals["argmax_abs"] / rows,
            "argmax_len_exact": totals["argmax_exact"] / rows,
            "mean_expected_len": totals["expected_sum"] / rows,
            "mean_rounded_len": totals["rounded_sum"] / rows,
            "mean_argmax_len": totals["argmax_sum"] / rows,
            "mean_target_len": totals["target_sum"] / rows,
        },
        "target_counts": [int(x) for x in count.tolist()],
        "target_probs": [float(x / rows) for x in count.tolist()],
        "per_horizon": per_horizon,
    }


def _js_divergence(p: list[float], q: list[float]) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / max(float(p_arr.sum()), 1e-12)
    q_arr = q_arr / max(float(q_arr.sum()), 1e-12)
    m = 0.5 * (p_arr + q_arr)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))))

    return 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)


def _distribution_compare(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            p = left["target_probs"]
            q = right["target_probs"]
            comparisons.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "l1": float(np.abs(np.asarray(p) - np.asarray(q)).sum()),
                    "js": _js_divergence(p, q),
                    "mean_target_delta": float(left["metrics"]["mean_target_len"] - right["metrics"]["mean_target_len"]),
                    "rounded_mae_delta": float(
                        left["metrics"]["rounded_expected_len_mae"] - right["metrics"]["rounded_expected_len_mae"]
                    ),
                }
            )
    return comparisons


def _write_csvs(output_dir: Path, results: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_metrics.csv").open("w", newline="") as f:
        fieldnames = ["split", "rows", *list(results[0]["metrics"].keys())]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({"split": item["name"], "rows": item["rows"], **item["metrics"]})

    with (output_dir / "per_horizon.csv").open("w", newline="") as f:
        fieldnames = ["split", *list(results[0]["per_horizon"][0].keys())]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            for row in item["per_horizon"]:
                writer.writerow({"split": item["name"], **row})

    with (output_dir / "distribution_shift.csv").open("w", newline="") as f:
        fieldnames = list(comparisons[0].keys()) if comparisons else ["left", "right", "l1", "js"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row)


def _write_plot(output_dir: Path, results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional.
        (output_dir / "plot_skipped.txt").write_text(f"matplotlib import failed: {exc}\n")
        return

    horizons = np.arange(len(results[0]["target_probs"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for item in results:
        axes[0].plot(horizons, item["target_probs"], marker="o", label=item["name"])
        axes[1].plot(
            horizons,
            [row["rounded_mae"] for row in item["per_horizon"]],
            marker="o",
            label=item["name"],
        )
        axes[2].plot(
            horizons,
            [row["mean_expected"] for row in item["per_horizon"]],
            marker="o",
            label=f"{item['name']} pred",
        )
    axes[2].plot(horizons, horizons, color="black", linestyle="--", label="ideal")
    axes[0].set_ylabel("target probability")
    axes[1].set_ylabel("rounded MAE")
    axes[2].set_ylabel("mean expected horizon")
    axes[2].set_xlabel("true horizon")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose integer-horizon policy distribution shift and error by horizon.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--math-trace-dir", type=Path, default=None)
    parser.add_argument(
        "--eval-trace-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional held-out trace split, for example humaneval=/path/to/trace.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-math-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    return parser.parse_args()


def _parse_eval_trace_dirs(args: argparse.Namespace) -> list[tuple[str, Path, int | None]]:
    evals: list[tuple[str, Path, int | None]] = []
    if args.math_trace_dir is not None:
        evals.append(("math500", args.math_trace_dir, args.max_math_rows))
    for item in args.eval_trace_dir:
        if "=" not in item:
            raise ValueError(f"--eval-trace-dir must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--eval-trace-dir has empty name: {item!r}")
        evals.append((name, Path(raw_path), args.max_eval_rows))
    if not evals:
        raise ValueError("provide --math-trace-dir and/or at least one --eval-trace-dir NAME=PATH")
    return evals


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds = _make_dataset(args.train_dir, kind="compact", max_rows=args.max_train_rows)
    val_ds = _make_dataset(args.val_dir, kind="compact", max_rows=args.max_val_rows)
    eval_specs = _parse_eval_trace_dirs(args)
    eval_datasets = [
        (name, path, _make_dataset(path, kind="trace", max_rows=max_rows))
        for name, path, max_rows in eval_specs
    ]
    model, config = _load_model(args.checkpoint, train_ds, device)
    loss_args = dict(config.get("loss_args", {}))
    if not loss_args:
        loss_args = {
            "ce_weight": float(config.get("ce_weight", 1.0)),
            "soft_ce_weight": float(config.get("soft_ce_weight", 0.0)),
            "soft_tau": float(config.get("soft_tau", 1.0)),
            "distance_weight": float(config.get("distance_weight", 0.2)),
            "emd_weight": float(config.get("emd_weight", 0.0)),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, dataset in [("train", train_ds), ("val", val_ds), *[(name, ds) for name, _path, ds in eval_datasets]]:
        print(json.dumps({"event": "diagnose_split_start", "split": name, "rows": len(dataset)}), flush=True)
        results.append(
            evaluate_split(
                name=name,
                dataset=dataset,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                loss_args=loss_args,
            )
        )
        print(json.dumps({"event": "diagnose_split_done", "split": name, **results[-1]["metrics"]}), flush=True)

    comparisons = _distribution_compare(results)
    payload = {
        "checkpoint": str(args.checkpoint),
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "eval_trace_dirs": {name: str(path) for name, path, _ds in eval_datasets},
        "loss_args": loss_args,
        "results": results,
        "distribution_shift": comparisons,
    }
    (args.output_dir / "diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csvs(args.output_dir, results, comparisons)
    _write_plot(args.output_dir, results)
    print(json.dumps({"event": "diagnose_done", "output_dir": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
