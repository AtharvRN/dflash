from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ALPHAS = "0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.95,0.96,0.98"


def _parse_floats(value: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in value.split(",") if x)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a small DFlashv2 horizon-policy loss sweep."
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--math500-trace-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--boundary-weights", type=_parse_floats, default=(1.0, 2.0, 3.0))
    parser.add_argument("--aux-arm-weights", type=_parse_floats, default=(0.0, 0.05, 0.10, 0.20))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--objective", choices=["survival_bce", "hazard"], default="survival_bce")
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--monotonic-weight", type=float, default=0.02)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--compact-cache-dir", type=Path, default=None)
    parser.add_argument("--selection-min-retention", type=float, default=0.95)
    parser.add_argument("--arms", default="4,8,12,16")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    compact_cache_dir = args.compact_cache_dir or (args.output_root / "_compact_last_features")
    rows: list[dict[str, Any]] = []

    for boundary_weight, aux_arm_weight in itertools.product(args.boundary_weights, args.aux_arm_weights):
        name = f"bw{boundary_weight:g}_aux{aux_arm_weight:g}".replace(".", "p")
        run_dir = args.output_root / name
        train_cmd = [
            sys.executable,
            "-u",
            "scripts/train_dflashv2_horizon_predictor.py",
            "--output-dir",
            str(run_dir),
            "--architecture",
            "last_mlp",
            "--proj-dim",
            str(args.proj_dim),
            "--hidden-size",
            str(args.hidden_size),
            "--num-layers",
            "1",
            "--dropout",
            str(args.dropout),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--objective",
            args.objective,
            "--length-loss-weight",
            str(args.length_loss_weight),
            "--monotonic-weight",
            str(args.monotonic_weight),
            "--boundary-weight",
            str(boundary_weight),
            "--boundary-ks",
            "4,8,12,15",
            "--aux-arm-weight",
            str(aux_arm_weight),
            "--calibration-rows",
            str(args.calibration_rows),
            "--compact-cache-dir",
            str(compact_cache_dir),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--monotonicize-eval",
            "--arms",
            args.arms,
            "--alphas",
            args.alphas,
            "--selection-min-retention",
            str(args.selection_min_retention),
            "--checkpoint-selection",
            "selected_accept_ratio",
        ]
        for trace_dir in args.trace_dir:
            train_cmd.extend(["--trace-dir", str(trace_dir)])
        if args.max_total_rows is not None:
            train_cmd.extend(["--max-total-rows", str(args.max_total_rows)])
        _run(train_cmd, dry_run=args.dry_run)

        eval_payload: dict[str, Any] = {}
        if args.math500_trace_dir:
            eval_path = run_dir / "offline_eval_math500_best.json"
            eval_cmd = [
                sys.executable,
                "-u",
                "scripts/evaluate_dflashv2_offline_policy.py",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--output-json",
                str(eval_path),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--monotonicize-eval",
                "--arms",
                args.arms,
                "--alphas",
                args.alphas,
                "--selection-min-retention",
                str(args.selection_min_retention),
            ]
            for trace_dir in args.math500_trace_dir:
                eval_cmd.extend(["--trace-dir", str(trace_dir)])
            _run(eval_cmd, dry_run=args.dry_run)
            if not args.dry_run:
                eval_payload = _load_json(eval_path)

        if args.dry_run:
            continue
        checkpoint = _load_json(run_dir / "config.json")
        best = {}
        best_path = run_dir / "best.pt"
        if best_path.exists():
            import torch

            best = torch.load(best_path, map_location="cpu", weights_only=False).get("val_metrics", {})
        math_metrics = eval_payload.get("metrics", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "objective": args.objective,
                "boundary_weight": boundary_weight,
                "aux_arm_weight": aux_arm_weight,
                "train_rows": checkpoint.get("train_rows"),
                "val_rows": checkpoint.get("val_rows"),
                "val_selected_alpha": best.get("selected_alpha"),
                "val_selected_mean_accepted": best.get("selected_mean_accepted"),
                "val_selected_accept_retention": best.get("selected_accept_retention"),
                "val_selected_accept_ratio": best.get("selected_accept_ratio"),
                "math500_selected_alpha": math_metrics.get("selected_alpha"),
                "math500_selected_mean_block": math_metrics.get("selected_mean_block"),
                "math500_selected_mean_accepted": math_metrics.get("selected_mean_accepted"),
                "math500_selected_accept_retention": math_metrics.get("selected_accept_retention"),
                "math500_selected_accept_ratio": math_metrics.get("selected_accept_ratio"),
            }
        )
        (args.output_root / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1], indent=2), flush=True)

    if not args.dry_run:
        ranked = sorted(
            rows,
            key=lambda row: (
                row.get("math500_selected_accept_retention") or 0.0,
                row.get("math500_selected_accept_ratio") or 0.0,
            ),
            reverse=True,
        )
        (args.output_root / "sweep_summary_ranked.json").write_text(json.dumps(ranked, indent=2) + "\n")


if __name__ == "__main__":
    main()
