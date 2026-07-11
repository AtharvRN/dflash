from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from train_block_policy_head import _draft_budgets, _parse_arms
from train_oracle_arm_classifier import _oracle_idx
from train_predictor_head import FlatSurvivalHead, Normalizer, TemporalSurvivalHead, _load_jsonl


DEFAULT_THRESHOLDS = ",".join(f"{x / 100:.2f}" for x in range(50, 100, 5))


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    threshold: float | None
    aggregation: str
    batches: int
    mean_selected_block: float
    selected_block_hist: dict[str, int]
    accepted_retention: float
    under_rate: float
    sufficient_rate: float
    mean_wasted_drafts: float
    mean_pred_budget: float
    estimated_total_ms: float
    estimated_tokens_per_s: float
    relative_to_fixed_b12: float | None


def _parse_thresholds(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("threshold list cannot be empty")
    for threshold in out:
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(f"threshold must be in [0, 1]: {threshold}")
    return out


def _parse_block_timings(value: str) -> dict[int, float]:
    out: dict[int, float] = {}
    if not value.strip():
        return out
    for item in value.split(","):
        key, raw_ms = item.split(":", 1)
        out[int(key)] = float(raw_ms)
    return out


def _load_profile_timings(path: Path | None) -> dict[int, float]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    timings: dict[int, float] = {}
    for key, value in data.items():
        if not key.startswith("fixed_b"):
            continue
        try:
            block = int(key.removeprefix("fixed_b"))
        except ValueError:
            continue
        section = value.get("bs64") or value.get("all") or {}
        ms = section.get("mean_total_ms") or section.get("mean_total_profiled_ms")
        if ms is not None:
            timings[block] = float(ms)
    return timings


def _complete_timings(
    arms: list[int],
    measured: dict[int, float],
    *,
    missing_policy: str,
) -> tuple[dict[int, float], dict[str, Any]]:
    if not measured:
        measured = {8: 83.201, 12: 104.064, 16: 125.597}

    completed: dict[int, float] = {}
    measured_blocks = sorted(measured)
    for arm in arms:
        if arm in measured:
            completed[arm] = measured[arm]
            continue
        if missing_policy == "nearest":
            nearest = min(measured_blocks, key=lambda block: (abs(block - arm), block < arm))
            completed[arm] = measured[nearest]
        elif missing_policy == "nearest_larger":
            larger = [block for block in measured_blocks if block >= arm]
            completed[arm] = measured[min(larger) if larger else max(measured_blocks)]
        elif missing_policy == "linear":
            xs = np.asarray(measured_blocks, dtype=np.float64)
            ys = np.asarray([measured[int(x)] for x in xs], dtype=np.float64)
            completed[arm] = float(np.interp(arm, xs, ys, left=ys[0], right=ys[-1]))
        else:
            raise ValueError(f"unknown missing timing policy: {missing_policy}")

    return completed, {
        "measured": {str(k): v for k, v in sorted(measured.items())},
        "completed": {str(k): v for k, v in sorted(completed.items())},
        "missing_policy": missing_policy,
    }


def _build_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "gru"))
    seq_shape = config["seq_shape"]
    static_dim = int(config["static_dim"])
    hidden_size = int(config.get("hidden_size", 192))
    dropout = float(config.get("dropout", 0.1))
    num_arms = int(config.get("num_arms", len(config.get("arms", []))))
    if architecture == "gru":
        return TemporalSurvivalHead(
            seq_dim=int(seq_shape[-1]),
            static_dim=static_dim,
            num_slots=num_arms,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if architecture == "mlp":
        return FlatSurvivalHead(
            seq_len=int(seq_shape[0]),
            seq_dim=int(seq_shape[-1]),
            static_dim=static_dim,
            num_slots=num_arms,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    raise ValueError(f"unsupported architecture: {architecture}")


def _predict_from_checkpoint(
    checkpoint: Path,
    val_jsonl: Path,
    *,
    device: torch.device,
    max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    seq, static, _survival, accepted_len = _load_jsonl(
        val_jsonl,
        max_rows=max_rows,
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

    model = _build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        batch_size = 4096
        for start in range(0, seq.shape[0], batch_size):
            seq_batch = torch.from_numpy(seq[start : start + batch_size]).to(device)
            static_batch = torch.from_numpy(static[start : start + batch_size]).to(device)
            logits_chunks.append(model(seq_batch, static_batch).cpu())
    probs = torch.softmax(torch.cat(logits_chunks, dim=0), dim=-1).numpy()
    return probs, accepted_len.astype(np.float32), np.asarray(config["arms"], dtype=np.int64), config


def _threshold_predictions(probs: np.ndarray, threshold: float) -> np.ndarray:
    cumulative = np.cumsum(probs, axis=1)
    ok = cumulative >= threshold
    pred = ok.argmax(axis=1)
    pred[~ok.any(axis=1)] = probs.shape[1] - 1
    return pred.astype(np.int64)


def _aggregate_idx(idx: np.ndarray, mode: str) -> int:
    if mode == "max":
        return int(idx.max())
    if mode == "p90":
        return int(np.ceil(np.quantile(idx, 0.90)))
    if mode == "ceil_mean":
        return int(np.ceil(idx.mean()))
    if mode == "majority":
        return int(np.bincount(idx).argmax())
    raise ValueError(f"unknown aggregation mode: {mode}")


def _evaluate_indices(
    pred_idx: np.ndarray,
    accepted_len: np.ndarray,
    *,
    arms: list[int],
    arm_kind: str,
    block_timings_ms: dict[int, float],
    aggregation: str,
    batch_size: int,
    policy: str,
    threshold: float | None,
    fixed_b12_tps: float | None,
) -> PolicyResult:
    budgets = np.asarray(_draft_budgets(arms, arm_kind), dtype=np.float32)
    arm_arr = np.asarray(arms, dtype=np.int64)
    if aggregation == "none":
        selected_idx = pred_idx
        selected_budget = budgets[selected_idx]
        retained = np.minimum(accepted_len, selected_budget)
        wasted = np.maximum(selected_budget - accepted_len, 0.0)
        total_ms = float(sum(block_timings_ms[int(arm_arr[i])] for i in selected_idx))
        batches = int(len(pred_idx))
    else:
        selected_idx = np.empty_like(pred_idx)
        retained = np.empty_like(accepted_len, dtype=np.float32)
        wasted = np.empty_like(accepted_len, dtype=np.float32)
        total_ms = 0.0
        batches = 0
        for start in range(0, len(pred_idx), batch_size):
            end = min(len(pred_idx), start + batch_size)
            batch_idx = _aggregate_idx(pred_idx[start:end], aggregation)
            selected_idx[start:end] = batch_idx
            budget = budgets[batch_idx]
            retained[start:end] = np.minimum(accepted_len[start:end], budget)
            wasted[start:end] = np.maximum(budget - accepted_len[start:end], 0.0)
            total_ms += float(block_timings_ms[int(arm_arr[batch_idx])])
            batches += 1

    selected_budget = budgets[selected_idx]
    tokens_per_s = float(retained.sum() / max(total_ms, 1e-9) * 1000.0)
    hist = np.bincount(selected_idx, minlength=len(arms))
    return PolicyResult(
        policy=policy,
        threshold=threshold,
        aggregation=aggregation,
        batches=batches,
        mean_selected_block=float(arm_arr[selected_idx].mean()),
        selected_block_hist={str(arm): int(hist[i]) for i, arm in enumerate(arms)},
        accepted_retention=float(retained.mean() / max(float(accepted_len.mean()), 1e-9)),
        under_rate=float((selected_budget < accepted_len).mean()),
        sufficient_rate=float((selected_budget >= accepted_len).mean()),
        mean_wasted_drafts=float(wasted.mean()),
        mean_pred_budget=float(selected_budget.mean()),
        estimated_total_ms=total_ms,
        estimated_tokens_per_s=tokens_per_s,
        relative_to_fixed_b12=None if fixed_b12_tps is None else tokens_per_s / fixed_b12_tps,
    )


def _rows(results: list[PolicyResult]) -> list[dict[str, Any]]:
    return [result.__dict__ for result in results]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path, *, aggregation: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (path.with_suffix(path.suffix + ".skipped.txt")).write_text(
            f"matplotlib import failed: {exc}\n"
        )
        return

    selected = [row for row in rows if row["aggregation"] == aggregation]
    if not selected:
        return
    plt.figure(figsize=(9, 6))
    for prefix, marker in [("risk_q", "o"), ("fixed", "s"), ("argmax", "^"), ("oracle", "*")]:
        group = [row for row in selected if str(row["policy"]).startswith(prefix)]
        if not group:
            continue
        plt.scatter(
            [row["accepted_retention"] for row in group],
            [row["estimated_tokens_per_s"] for row in group],
            label=prefix,
            marker=marker,
            s=55 if prefix != "oracle" else 120,
        )
        if prefix == "risk_q":
            for row in group:
                if row["threshold"] in {0.5, 0.7, 0.85, 0.9, 0.95}:
                    plt.annotate(f"q={row['threshold']:.2f}", (row["accepted_retention"], row["estimated_tokens_per_s"]))
    plt.xlabel("Accepted Retention")
    plt.ylabel("Estimated Tokens / Second")
    plt.title(f"Oracle Arm Policy Proxy ({aggregation})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated DFlash oracle-arm policies offline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-summary-json", type=Path, default=None)
    parser.add_argument("--block-timing-ms", type=_parse_block_timings, default={})
    parser.add_argument("--missing-timing-policy", choices=["nearest", "nearest_larger", "linear"], default="nearest_larger")
    parser.add_argument("--thresholds", type=_parse_thresholds, default=_parse_thresholds(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aggregations", default="none,max,p90,ceil_mean,majority")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    probs, accepted_len, arms_np, config = _predict_from_checkpoint(
        args.checkpoint,
        args.val_jsonl,
        device=device,
        max_rows=args.max_rows,
    )
    arms = [int(x) for x in arms_np.tolist()]
    arm_kind = str(config.get("arm_kind", "block_size"))
    target_idx = _oracle_idx(accepted_len, arms=arms, arm_kind=arm_kind)

    measured = _load_profile_timings(args.profile_summary_json)
    measured.update(args.block_timing_ms)
    block_timings_ms, timing_info = _complete_timings(
        arms,
        measured,
        missing_policy=args.missing_timing_policy,
    )

    aggregations = [x.strip() for x in args.aggregations.split(",") if x.strip()]
    all_results: list[PolicyResult] = []
    fixed_b12_tps_by_aggregation: dict[str, float] = {}
    for aggregation in aggregations:
        fixed_b12_idx = arms.index(12) if 12 in arms else len(arms) - 1
        fixed_b12_pred = np.full_like(target_idx, fixed_b12_idx)
        fixed_b12 = _evaluate_indices(
            fixed_b12_pred,
            accepted_len,
            arms=arms,
            arm_kind=arm_kind,
            block_timings_ms=block_timings_ms,
            aggregation=aggregation,
            batch_size=args.batch_size,
            policy="fixed_b12",
            threshold=None,
            fixed_b12_tps=None,
        )
        fixed_b12_tps_by_aggregation[aggregation] = fixed_b12.estimated_tokens_per_s

        for arm in arms:
            pred = np.full_like(target_idx, arms.index(arm))
            all_results.append(
                _evaluate_indices(
                    pred,
                    accepted_len,
                    arms=arms,
                    arm_kind=arm_kind,
                    block_timings_ms=block_timings_ms,
                    aggregation=aggregation,
                    batch_size=args.batch_size,
                    policy=f"fixed_b{arm}",
                    threshold=None,
                    fixed_b12_tps=fixed_b12_tps_by_aggregation[aggregation],
                )
            )
        all_results.append(
            _evaluate_indices(
                probs.argmax(axis=1).astype(np.int64),
                accepted_len,
                arms=arms,
                arm_kind=arm_kind,
                block_timings_ms=block_timings_ms,
                aggregation=aggregation,
                batch_size=args.batch_size,
                policy="argmax",
                threshold=None,
                fixed_b12_tps=fixed_b12_tps_by_aggregation[aggregation],
            )
        )
        all_results.append(
            _evaluate_indices(
                target_idx,
                accepted_len,
                arms=arms,
                arm_kind=arm_kind,
                block_timings_ms=block_timings_ms,
                aggregation=aggregation,
                batch_size=args.batch_size,
                policy="oracle",
                threshold=None,
                fixed_b12_tps=fixed_b12_tps_by_aggregation[aggregation],
            )
        )
        for threshold in args.thresholds:
            pred = _threshold_predictions(probs, threshold)
            all_results.append(
                _evaluate_indices(
                    pred,
                    accepted_len,
                    arms=arms,
                    arm_kind=arm_kind,
                    block_timings_ms=block_timings_ms,
                    aggregation=aggregation,
                    batch_size=args.batch_size,
                    policy=f"risk_q{threshold:.2f}",
                    threshold=threshold,
                    fixed_b12_tps=fixed_b12_tps_by_aggregation[aggregation],
                )
            )

    rows = _rows(all_results)
    payload = {
        "checkpoint": str(args.checkpoint),
        "val_jsonl": str(args.val_jsonl),
        "rows": int(len(accepted_len)),
        "arms": arms,
        "arm_kind": arm_kind,
        "mean_accepted_len": float(accepted_len.mean()),
        "target_arm_hist": {str(arm): int(x) for arm, x in zip(arms, np.bincount(target_idx, minlength=len(arms)))},
        "timings_ms": timing_info,
        "results": rows,
    }
    (args.output_dir / "policy_proxy_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.output_dir / "policy_proxy_results.csv", rows)
    for aggregation in aggregations:
        _plot(rows, args.output_dir / f"policy_proxy_{aggregation}.png", aggregation=aggregation)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
