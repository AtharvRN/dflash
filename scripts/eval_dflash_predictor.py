#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from torch import nn
import torch

class AcceptPredictorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LoadedPredictor:
    model: AcceptPredictorMLP
    checkpoint_path: str
    input_dim: int
    hidden_dim: int
    dropout: float
    output_mode: str


def _get_nested(mapping: dict, *keys: str):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_dflash_accept_predictor(
    *, checkpoint_path: str, device: torch.device | str
) -> LoadedPredictor:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict checkpoint payload, got {type(payload)!r}.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint missing model_state_dict.")
    first_weight = state.get("net.0.weight")
    if first_weight is None or not hasattr(first_weight, "shape"):
        raise ValueError("Checkpoint missing net.0.weight.")

    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    input_dim = int(metrics.get("input_dim") or int(first_weight.shape[1]))
    hidden_dim = int(_get_nested(metrics, "args", "hidden_dim") or int(first_weight.shape[0]))
    dropout = float(_get_nested(metrics, "args", "dropout") or 0.0)
    output_mode = str(
        payload.get("output_mode")
        or metrics.get("output_mode")
        or (
            "hazard"
            if str(_get_nested(metrics, "args", "objective") or "") == "hazard"
            else "prefix_survival"
        )
    )

    model = AcceptPredictorMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return LoadedPredictor(
        model=model,
        checkpoint_path=str(checkpoint_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_mode=output_mode,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained DFLASH accept predictor on task-specific feature "
            "shards and report both token-classification quality and SpecDec++-style "
            "verify-length decision quality."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80",
        help=(
            "Comma-separated direct prefix-survival thresholds used for verify-length "
            "sweeps. A verify prefix stops at the first drafted position where the "
            "predicted probability P(tau >= i) falls below the threshold."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--token-threshold", type=float, default=0.5)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        val = float(text)
        if not (0.0 < val < 1.0):
            raise ValueError(f"Thresholds must lie in (0,1). Got {val}.")
        values.append(val)
    if not values:
        raise ValueError("No valid thresholds were provided.")
    return sorted(set(values))


def _find_index_path(feature_dir: Path) -> Path:
    matches = sorted(feature_dir.glob("*_index.json"))
    if not matches:
        raise FileNotFoundError(f"No *_index.json found under {feature_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one index file under {feature_dir}, found {len(matches)}."
        )
    return matches[0]


def _iterate_batches(x: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, int(x.shape[0]), int(batch_size)):
        yield x[start : start + int(batch_size)]


@dataclass
class TokenMetrics:
    rows: int
    positive_rate: float
    predicted_positive_rate: float
    mean_prob_accepted: float
    mean_prob_rejected: float
    accuracy_at_token_threshold: float
    precision_at_token_threshold: float
    recall_at_token_threshold: float
    f1_at_token_threshold: float
    bce_loss: float
    brier_score: float


@dataclass
class ThresholdMetrics:
    threshold: float
    cycles: int
    mean_predicted_verify_tokens: float
    mean_runtime_block_size: float
    mean_true_boundary_tokens: float
    mean_true_accept_tokens: float
    mean_retained_accept_tokens: float
    tau_retention: float
    verify_fraction_of_full: float
    verify_reduction_vs_full: float
    accept_ratio_proxy: float
    boundary_exact_rate: float
    boundary_within_1_rate: float
    boundary_mae: float
    boundary_signed_error: float
    boundary_under_rate: float
    boundary_over_rate: float
    boundary_under_mae: float
    boundary_over_mae: float


def _build_features(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    return payload["draft_hidden"].to(torch.float32)


def _convert_hazard_rows_to_survival(rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    request_ids = rows["request_id"].to(torch.int64)
    cycle_indices = rows["cycle_idx"].to(torch.int64)
    draft_positions = rows["draft_pos"].to(torch.int64)
    hazards = rows["probs"].to(torch.float32)
    survival = torch.empty_like(hazards)

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (req_id, cycle_idx) in enumerate(
        zip(request_ids.tolist(), cycle_indices.tolist(), strict=True)
    ):
        grouped[(int(req_id), int(cycle_idx))].append(int(idx))

    for key, idxs in grouped.items():
        idx_tensor = torch.tensor(idxs, dtype=torch.int64)
        pos = draft_positions.index_select(0, idx_tensor)
        hazard_vals = hazards.index_select(0, idx_tensor)
        order = torch.argsort(pos)
        sorted_pos = pos.index_select(0, order)
        expected = torch.arange(1, 1 + int(sorted_pos.numel()), dtype=torch.int64)
        if not torch.equal(sorted_pos.cpu(), expected):
            raise RuntimeError(
                "Hazard evaluation expects contiguous draft positions within a cycle. "
                f"cycle={key} positions={sorted_pos.tolist()}"
            )
        sorted_hazard = hazard_vals.index_select(0, order).clamp(1e-6, 1.0 - 1e-6)
        sorted_survival = torch.cumprod(1.0 - sorted_hazard, dim=0)
        target_idxs = idx_tensor.index_select(0, order)
        survival.index_copy_(0, target_idxs, sorted_survival)

    out = dict(rows)
    out["probs"] = survival
    return out


def _score_tokens(
    *,
    feature_dir: Path,
    shard_names: list[str],
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor, str]:
    loaded = load_dflash_accept_predictor(
        checkpoint_path=str(checkpoint),
        device=device,
    )

    if loaded.input_dim <= 0:
        raise RuntimeError("Predictor checkpoint reports an invalid input_dim.")

    merged: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for shard_name in shard_names:
            payload = torch.load(feature_dir / shard_name, map_location="cpu")
            features = _build_features(payload)
            if int(features.shape[1]) != int(loaded.input_dim):
                raise RuntimeError(
                    "Predictor feature dimension mismatch during evaluation. "
                    f"expected={loaded.input_dim} got={features.shape[1]} shard={shard_name}"
                )

            shard_probs: list[torch.Tensor] = []
            for batch in _iterate_batches(features, batch_size=batch_size):
                logits = loaded.model(batch.to(device=device, dtype=torch.float32))
                shard_probs.append(torch.sigmoid(logits).squeeze(1).to(torch.float32).cpu())

            merged["probs"].append(torch.cat(shard_probs, dim=0))
            for key in (
                "request_id",
                "cycle_idx",
                "draft_pos",
                "runtime_block_size",
                "verify_token_num",
                "accepted_draft_tokens",
                "token_accepted",
            ):
                merged[key].append(payload[key].cpu())

    concatenated = {key: torch.cat(parts, dim=0) for key, parts in merged.items()}
    if str(loaded.output_mode) == "hazard":
        concatenated = _convert_hazard_rows_to_survival(concatenated)
    return (
        concatenated,
        torch.tensor([loaded.input_dim, loaded.hidden_dim], dtype=torch.int64),
        str(loaded.output_mode),
    )


def _compute_token_metrics(
    *,
    probs: torch.Tensor,
    labels: torch.Tensor,
    token_threshold: float,
) -> TokenMetrics:
    probs = probs.to(torch.float32)
    labels = labels.to(torch.float32)
    preds = (probs >= float(token_threshold)).to(torch.float32)

    rows = int(labels.numel())
    pos = labels.sum()
    neg = float(rows) - float(pos.item())
    tp = float(((preds == 1.0) & (labels == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (labels == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (labels == 1.0)).sum().item())

    accuracy = float((preds == labels).to(torch.float32).mean().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)

    probs_clamped = probs.clamp(1e-6, 1.0 - 1e-6)
    bce = -(
        labels * torch.log(probs_clamped) + (1.0 - labels) * torch.log(1.0 - probs_clamped)
    ).mean()
    brier = torch.square(probs - labels).mean()

    accepted_mask = labels == 1.0
    rejected_mask = labels == 0.0
    mean_prob_accepted = (
        float(probs[accepted_mask].mean().item()) if bool(torch.any(accepted_mask)) else math.nan
    )
    mean_prob_rejected = (
        float(probs[rejected_mask].mean().item()) if bool(torch.any(rejected_mask)) else math.nan
    )

    return TokenMetrics(
        rows=rows,
        positive_rate=float(pos.item()) / max(rows, 1),
        predicted_positive_rate=float(preds.mean().item()),
        mean_prob_accepted=mean_prob_accepted,
        mean_prob_rejected=mean_prob_rejected,
        accuracy_at_token_threshold=accuracy,
        precision_at_token_threshold=precision,
        recall_at_token_threshold=recall,
        f1_at_token_threshold=f1,
        bce_loss=float(bce.item()),
        brier_score=float(brier.item()),
    )


def _build_cycle_records(rows: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, object]] = {}

    request_ids = rows["request_id"].to(torch.int64).tolist()
    cycle_indices = rows["cycle_idx"].to(torch.int64).tolist()
    draft_positions = rows["draft_pos"].to(torch.int64).tolist()
    runtime_block_sizes = rows["runtime_block_size"].to(torch.int64).tolist()
    accepted_tokens = rows["accepted_draft_tokens"].to(torch.int64).tolist()
    probs = rows["probs"].to(torch.float32).tolist()

    for req_id, cycle_idx, draft_pos, runtime_bs, accepted, prob in zip(
        request_ids,
        cycle_indices,
        draft_positions,
        runtime_block_sizes,
        accepted_tokens,
        probs,
        strict=True,
    ):
        key = (int(req_id), int(cycle_idx))
        rec = grouped.get(key)
        if rec is None:
            rec = {
                "request_id": int(req_id),
                "cycle_idx": int(cycle_idx),
                "runtime_block_size": int(runtime_bs),
                "accepted_draft_tokens": int(accepted),
                "positions": [],
                "probs": [],
            }
            grouped[key] = rec
        rec["positions"].append(int(draft_pos))
        rec["probs"].append(float(prob))

    cycles: list[dict[str, object]] = []
    for rec in grouped.values():
        positions = list(rec["positions"])
        probs_list = list(rec["probs"])
        order = sorted(range(len(positions)), key=lambda i: positions[i])
        sorted_positions = [positions[i] for i in order]
        sorted_probs = [probs_list[i] for i in order]
        expected_positions = list(range(1, len(sorted_positions) + 1))
        if sorted_positions != expected_positions:
            raise RuntimeError(
                "Predictor feature shard has non-contiguous draft positions within a cycle. "
                f"request_id={rec['request_id']} cycle_idx={rec['cycle_idx']} "
                f"positions={sorted_positions}"
            )
        rec["positions"] = sorted_positions
        rec["probs"] = sorted_probs
        cycles.append(rec)
    cycles.sort(key=lambda rec: (int(rec["request_id"]), int(rec["cycle_idx"])))
    return cycles


def _compute_threshold_metrics(
    *,
    cycles: list[dict[str, object]],
    thresholds: list[float],
) -> list[ThresholdMetrics]:
    out: list[ThresholdMetrics] = []
    if not cycles:
        return out

    mean_runtime_block_size = sum(int(rec["runtime_block_size"]) for rec in cycles) / len(cycles)
    mean_true_accept = sum(int(rec["accepted_draft_tokens"]) for rec in cycles) / len(cycles)
    mean_true_boundary = (
        sum(
            int(rec["runtime_block_size"])
            if int(rec["accepted_draft_tokens"]) >= len(list(rec["probs"]))
            else int(rec["accepted_draft_tokens"]) + 1
            for rec in cycles
        )
        / len(cycles)
    )

    for thr in thresholds:
        total_pred_verify = 0.0
        total_retained_accept = 0.0
        total_boundary_abs_err = 0.0
        total_boundary_signed_err = 0.0
        total_under_amount = 0.0
        total_over_amount = 0.0
        exact = 0
        within_1 = 0
        under = 0
        over = 0

        for rec in cycles:
            runtime_bs = int(rec["runtime_block_size"])
            accepted = int(rec["accepted_draft_tokens"])
            probs = list(rec["probs"])

            pred_verify = int(runtime_bs)
            for idx, prob in enumerate(probs, start=1):
                if float(prob) < float(thr):
                    pred_verify = int(idx)
                    break

            true_boundary = int(runtime_bs) if accepted >= len(probs) else int(accepted + 1)
            retained_accept = int(min(accepted, pred_verify))
            boundary_err = float(pred_verify) - float(true_boundary)

            total_pred_verify += float(pred_verify)
            total_retained_accept += float(retained_accept)
            total_boundary_abs_err += abs(boundary_err)
            total_boundary_signed_err += boundary_err
            total_under_amount += max(float(true_boundary) - float(pred_verify), 0.0)
            total_over_amount += max(float(pred_verify) - float(true_boundary), 0.0)
            exact += int(pred_verify == true_boundary)
            within_1 += int(abs(boundary_err) <= 1.0)
            under += int(pred_verify < true_boundary)
            over += int(pred_verify > true_boundary)

        cycles_n = len(cycles)
        mean_pred_verify = total_pred_verify / cycles_n
        mean_retained_accept = total_retained_accept / cycles_n
        tau_retention = mean_retained_accept / max(mean_true_accept, 1e-12)
        verify_fraction = mean_pred_verify / max(mean_runtime_block_size, 1e-12)
        accept_ratio_proxy = mean_retained_accept / max(mean_pred_verify, 1e-12)
        out.append(
            ThresholdMetrics(
                threshold=float(thr),
                cycles=cycles_n,
                mean_predicted_verify_tokens=mean_pred_verify,
                mean_runtime_block_size=mean_runtime_block_size,
                mean_true_boundary_tokens=mean_true_boundary,
                mean_true_accept_tokens=mean_true_accept,
                mean_retained_accept_tokens=mean_retained_accept,
                tau_retention=tau_retention,
                verify_fraction_of_full=verify_fraction,
                verify_reduction_vs_full=1.0 - verify_fraction,
                accept_ratio_proxy=accept_ratio_proxy,
                boundary_exact_rate=float(exact) / cycles_n,
                boundary_within_1_rate=float(within_1) / cycles_n,
                boundary_mae=total_boundary_abs_err / cycles_n,
                boundary_signed_error=total_boundary_signed_err / cycles_n,
                boundary_under_rate=float(under) / cycles_n,
                boundary_over_rate=float(over) / cycles_n,
                boundary_under_mae=total_under_amount / cycles_n,
                boundary_over_mae=total_over_amount / cycles_n,
            )
        )
    return out


def _pick_best_threshold(
    metrics: list[ThresholdMetrics],
    *,
    tau_floor: float,
) -> dict[str, object] | None:
    eligible = [m for m in metrics if float(m.tau_retention) >= float(tau_floor)]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda m: (
            float(m.accept_ratio_proxy),
            float(m.verify_reduction_vs_full),
            -float(m.boundary_mae),
        ),
    )
    return asdict(best)


def _write_markdown(
    *,
    output_path: Path,
    feature_dir: Path,
    checkpoint: Path,
    input_dim: int,
    hidden_dim: int,
    token_threshold: float,
    token_metrics: TokenMetrics,
    cycle_count: int,
    request_count: int,
    threshold_metrics: list[ThresholdMetrics],
) -> None:
    best_tau99 = _pick_best_threshold(threshold_metrics, tau_floor=0.99)
    best_tau95 = _pick_best_threshold(threshold_metrics, tau_floor=0.95)

    lines: list[str] = []
    lines.append("# DFLASH Predictor Evaluation")
    lines.append("")
    lines.append(f"- feature_dir: `{feature_dir}`")
    lines.append(f"- checkpoint: `{checkpoint}`")
    lines.append(f"- input_dim: `{input_dim}`")
    lines.append(f"- hidden_dim: `{hidden_dim}`")
    lines.append(f"- requests: `{request_count}`")
    lines.append(f"- cycles: `{cycle_count}`")
    lines.append(f"- token_threshold_for_classification: `{token_threshold:.2f}`")
    lines.append("")
    lines.append("## How To Read This")
    lines.append("")
    lines.append(
        "Token accuracy alone is not enough for deployment. The real question is whether "
        "a thresholded predictor preserves accepted draft tokens (`tau`) while shrinking "
        "the verify prefix. Here the threshold sweep treats the existing head output at "
        "draft position `i` as a direct estimate of prefix survival `P(tau >= i)` and "
        "stops at the first drafted position whose predicted prefix survival falls below "
        "the threshold."
    )
    lines.append("")
    lines.append("## Token-Level Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for key, value in asdict(token_metrics).items():
        if isinstance(value, int):
            rendered = f"{value}"
        else:
            rendered = f"{float(value):.6f}"
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    lines.append("## Threshold Sweep")
    lines.append("")
    lines.append(
        "| threshold | mean verify k | mean true boundary | verify/full | verify reduction | retained tau | tau retention | accept-ratio proxy | exact | within-1 | MAE | signed err | under rate | over rate | under amt | over amt |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in threshold_metrics:
        lines.append(
            "| "
            f"{row.threshold:.2f} | "
            f"{row.mean_predicted_verify_tokens:.3f} | "
            f"{row.mean_true_boundary_tokens:.3f} | "
            f"{row.verify_fraction_of_full:.3f} | "
            f"{row.verify_reduction_vs_full:.3f} | "
            f"{row.mean_retained_accept_tokens:.3f} | "
            f"{row.tau_retention:.3f} | "
            f"{row.accept_ratio_proxy:.3f} | "
            f"{row.boundary_exact_rate:.3f} | "
            f"{row.boundary_within_1_rate:.3f} | "
            f"{row.boundary_mae:.3f} | "
            f"{row.boundary_signed_error:.3f} | "
            f"{row.boundary_under_rate:.3f} | "
            f"{row.boundary_over_rate:.3f} | "
            f"{row.boundary_under_mae:.3f} | "
            f"{row.boundary_over_mae:.3f} |"
        )
    lines.append("")
    lines.append("## Recommended Operating Points")
    lines.append("")
    if best_tau99 is None:
        lines.append("- No tested threshold preserved at least `99%` of baseline tau.")
    else:
        lines.append(
            "- Best threshold with `tau_retention >= 0.99`: "
            f"`{best_tau99['threshold']:.2f}` "
            f"(verify reduction `{best_tau99['verify_reduction_vs_full']:.3f}`, "
            f"accept-ratio proxy `{best_tau99['accept_ratio_proxy']:.3f}`)."
        )
    if best_tau95 is None:
        lines.append("- No tested threshold preserved at least `95%` of baseline tau.")
    else:
        lines.append(
            "- Best threshold with `tau_retention >= 0.95`: "
            f"`{best_tau95['threshold']:.2f}` "
            f"(verify reduction `{best_tau95['verify_reduction_vs_full']:.3f}`, "
            f"accept-ratio proxy `{best_tau95['accept_ratio_proxy']:.3f}`)."
        )
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    feature_dir = Path(args.feature_dir)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = _find_index_path(feature_dir)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = list(index_payload.get("shards", []))
    if not shard_names:
        raise ValueError(f"No shard list found in {index_path}")
    if int(args.max_shards) > 0:
        shard_names = shard_names[: int(args.max_shards)]

    thresholds = _parse_thresholds(args.thresholds)
    device = _resolve_device(str(args.device))

    rows, dims, output_mode = _score_tokens(
        feature_dir=feature_dir,
        shard_names=shard_names,
        checkpoint=checkpoint,
        batch_size=int(args.batch_size),
        device=device,
    )
    token_metrics = _compute_token_metrics(
        probs=rows["probs"],
        labels=rows["token_accepted"],
        token_threshold=float(args.token_threshold),
    )
    cycles = _build_cycle_records(rows)
    threshold_metrics = _compute_threshold_metrics(cycles=cycles, thresholds=thresholds)

    payload = {
        "feature_dir": str(feature_dir),
        "checkpoint": str(checkpoint),
        "index_path": str(index_path),
        "num_shards_used": len(shard_names),
        "input_dim": int(dims[0].item()),
        "hidden_dim": int(dims[1].item()),
        "output_mode": str(output_mode),
        "device": str(device),
        "request_count": len(index_payload.get("request_id_to_rid", [])),
        "cycle_count": len(cycles),
        "token_threshold": float(args.token_threshold),
        "token_metrics": asdict(token_metrics),
        "threshold_metrics": [asdict(row) for row in threshold_metrics],
        "best_threshold_tau99": _pick_best_threshold(threshold_metrics, tau_floor=0.99),
        "best_threshold_tau95": _pick_best_threshold(threshold_metrics, tau_floor=0.95),
    }

    (output_dir / "predictor_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        output_path=output_dir / "predictor_eval.md",
        feature_dir=feature_dir,
        checkpoint=checkpoint,
        input_dim=int(dims[0].item()),
        hidden_dim=int(dims[1].item()),
        token_threshold=float(args.token_threshold),
        token_metrics=token_metrics,
        cycle_count=len(cycles),
        request_count=len(index_payload.get("request_id_to_rid", [])),
        threshold_metrics=threshold_metrics,
    )
    print(f"Wrote {(output_dir / 'predictor_eval.json')}")
    print(f"Wrote {(output_dir / 'predictor_eval.md')}")


if __name__ == "__main__":
    main()
