#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


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


@dataclass
class PositionProfile:
    draft_pos: int
    rows: int
    mean_predicted_accept_prob: float
    empirical_accept_rate: float
    abs_gap: float
    mean_cumulative_accept_prob: float
    cumulative_abs_gap: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a trained DFLASH predictor by drafted token index. This writes "
            "direct and cumulative prediction curves so we can inspect how the head "
            "behaves across the verify prefix."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0)
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    )


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


def _score_rows(
    *,
    feature_dir: Path,
    shard_names: list[str],
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], LoadedPredictor]:
    loaded = load_dflash_accept_predictor(checkpoint_path=str(checkpoint), device=device)
    merged: dict[str, list[torch.Tensor]] = defaultdict(list)

    with torch.inference_mode():
        for shard_name in shard_names:
            payload = torch.load(feature_dir / shard_name, map_location="cpu")
            features = payload["draft_hidden"].to(torch.float32)
            if int(features.shape[1]) != int(loaded.input_dim):
                raise RuntimeError(
                    "Predictor feature dimension mismatch during profiling. "
                    f"expected={loaded.input_dim} got={features.shape[1]} shard={shard_name}"
                )

            shard_probs: list[torch.Tensor] = []
            for batch in _iterate_batches(features, batch_size=batch_size):
                logits = loaded.model(batch.to(device=device, dtype=torch.float32))
                shard_probs.append(torch.sigmoid(logits).squeeze(1).cpu())

            merged["probs"].append(torch.cat(shard_probs, dim=0))
            for key in (
                "request_id",
                "cycle_idx",
                "draft_pos",
                "runtime_block_size",
                "accepted_draft_tokens",
                "token_accepted",
            ):
                merged[key].append(payload[key].cpu())

    rows = {key: torch.cat(parts, dim=0) for key, parts in merged.items()}
    return rows, loaded


def _build_cycle_records(rows: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, object]] = {}

    for req_id, cycle_idx, draft_pos, runtime_bs, accepted, prob in zip(
        rows["request_id"].to(torch.int64).tolist(),
        rows["cycle_idx"].to(torch.int64).tolist(),
        rows["draft_pos"].to(torch.int64).tolist(),
        rows["runtime_block_size"].to(torch.int64).tolist(),
        rows["accepted_draft_tokens"].to(torch.int64).tolist(),
        rows["probs"].to(torch.float32).tolist(),
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
        probs = list(rec["probs"])
        order = sorted(range(len(positions)), key=lambda i: positions[i])
        sorted_positions = [positions[i] for i in order]
        sorted_probs = [probs[i] for i in order]
        expected_positions = list(range(1, len(sorted_positions) + 1))
        if sorted_positions != expected_positions:
            raise RuntimeError(
                "Non-contiguous draft positions within a cycle. "
                f"request_id={rec['request_id']} cycle_idx={rec['cycle_idx']} "
                f"positions={sorted_positions}"
            )
        rec["positions"] = sorted_positions
        rec["probs"] = sorted_probs
        cycles.append(rec)
    cycles.sort(key=lambda rec: (int(rec["request_id"]), int(rec["cycle_idx"])))
    return cycles


def _compute_position_profiles(
    rows: dict[str, torch.Tensor], cycles: list[dict[str, object]]
) -> list[PositionProfile]:
    max_pos = int(rows["draft_pos"].max().item()) if int(rows["draft_pos"].numel()) > 0 else 0
    out: list[PositionProfile] = []

    for draft_pos in range(1, max_pos + 1):
        mask = rows["draft_pos"] == int(draft_pos)
        if not bool(torch.any(mask)):
            continue

        direct_probs = rows["probs"][mask].to(torch.float32)
        direct_labels = rows["token_accepted"][mask].to(torch.float32)

        cumulative_probs: list[float] = []
        for rec in cycles:
            probs = list(rec["probs"])
            if len(probs) < draft_pos:
                continue
            prefix_prob = 1.0
            for prob in probs[:draft_pos]:
                prefix_prob *= float(prob)
            cumulative_probs.append(prefix_prob)

        mean_pred = float(direct_probs.mean().item())
        empirical = float(direct_labels.mean().item())
        mean_cumulative = (
            sum(cumulative_probs) / len(cumulative_probs) if cumulative_probs else float("nan")
        )

        out.append(
            PositionProfile(
                draft_pos=int(draft_pos),
                rows=int(mask.sum().item()),
                mean_predicted_accept_prob=mean_pred,
                empirical_accept_rate=empirical,
                abs_gap=abs(mean_pred - empirical),
                mean_cumulative_accept_prob=mean_cumulative,
                cumulative_abs_gap=abs(mean_cumulative - empirical),
            )
        )

    return out


def _weighted_gap(profiles: list[PositionProfile], field: str) -> float:
    num = 0.0
    den = 0.0
    for profile in profiles:
        weight = float(profile.rows)
        num += weight * float(getattr(profile, field))
        den += weight
    return num / max(den, 1e-12)


def _plot_profiles(output_path: Path, profiles: list[PositionProfile]) -> None:
    xs = [profile.draft_pos for profile in profiles]
    counts = [profile.rows for profile in profiles]
    empirical = [profile.empirical_accept_rate for profile in profiles]
    direct = [profile.mean_predicted_accept_prob for profile in profiles]
    cumulative = [profile.mean_cumulative_accept_prob for profile in profiles]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 3.0, 1.3]},
    )

    axes[0].plot(xs, empirical, label="Empirical accept rate", linewidth=2.2, color="#1b5e20")
    axes[0].plot(
        xs,
        direct,
        label="Mean predicted accept prob",
        linewidth=2.0,
        color="#1565c0",
    )
    axes[0].set_ylabel("Direct prob")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[0].set_title("DFLASH predictor profile by drafted token index")

    axes[1].plot(
        xs,
        empirical,
        label="Empirical accept rate",
        linewidth=2.2,
        color="#1b5e20",
    )
    axes[1].plot(
        xs,
        cumulative,
        label="Mean cumulative product",
        linewidth=2.0,
        color="#c62828",
    )
    axes[1].set_ylabel("Cumulative prob")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].bar(xs, counts, color="#616161", width=0.75)
    axes[2].set_xlabel("Draft position index")
    axes[2].set_ylabel("Rows")
    axes[2].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    *,
    output_path: Path,
    feature_dir: Path,
    checkpoint: Path,
    predictor: LoadedPredictor,
    request_count: int,
    cycle_count: int,
    profiles: list[PositionProfile],
    plot_path: Path,
) -> None:
    weighted_direct_gap = _weighted_gap(profiles, "abs_gap")
    weighted_cumulative_gap = _weighted_gap(profiles, "cumulative_abs_gap")
    worst_direct = max(profiles, key=lambda row: row.abs_gap)
    worst_cumulative = max(profiles, key=lambda row: row.cumulative_abs_gap)

    lines: list[str] = []
    lines.append("# DFLASH Predictor Index Profile")
    lines.append("")
    lines.append(f"- feature_dir: `{feature_dir}`")
    lines.append(f"- checkpoint: `{checkpoint}`")
    lines.append(f"- plot: `{plot_path}`")
    lines.append(f"- input_dim: `{predictor.input_dim}`")
    lines.append(f"- hidden_dim: `{predictor.hidden_dim}`")
    lines.append(f"- requests: `{request_count}`")
    lines.append(f"- cycles: `{cycle_count}`")
    lines.append(f"- max draft index observed: `{profiles[-1].draft_pos if profiles else 0}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This profile compares two interpretations of the existing head at each drafted "
        "position: the direct per-index probability output by the head, and the "
        "cumulative-product probability implied by multiplying predictions across the prefix."
    )
    lines.append("")
    lines.append(f"- Weighted direct abs gap vs empirical rate: `{weighted_direct_gap:.4f}`")
    lines.append(
        f"- Weighted cumulative abs gap vs empirical rate: `{weighted_cumulative_gap:.4f}`"
    )
    lines.append(
        f"- Worst direct gap: index `{worst_direct.draft_pos}` with abs gap `{worst_direct.abs_gap:.4f}`"
    )
    lines.append(
        f"- Worst cumulative gap: index `{worst_cumulative.draft_pos}` with abs gap "
        f"`{worst_cumulative.cumulative_abs_gap:.4f}`"
    )
    lines.append("")
    lines.append("## By Index")
    lines.append("")
    lines.append(
        "| idx | rows | empirical accept | mean predicted | direct gap | mean cumulative | cumulative gap |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in profiles:
        lines.append(
            f"| {profile.draft_pos} | {profile.rows} | "
            f"{profile.empirical_accept_rate:.4f} | "
            f"{profile.mean_predicted_accept_prob:.4f} | "
            f"{profile.abs_gap:.4f} | "
            f"{profile.mean_cumulative_accept_prob:.4f} | "
            f"{profile.cumulative_abs_gap:.4f} |"
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

    rows, predictor = _score_rows(
        feature_dir=feature_dir,
        shard_names=shard_names,
        checkpoint=checkpoint,
        batch_size=int(args.batch_size),
        device=_resolve_device(str(args.device)),
    )
    cycles = _build_cycle_records(rows)
    profiles = _compute_position_profiles(rows, cycles)

    payload = {
        "feature_dir": str(feature_dir),
        "checkpoint": str(checkpoint),
        "index_path": str(index_path),
        "num_shards_used": len(shard_names),
        "request_count": len(index_payload.get("request_id_to_rid", [])),
        "cycle_count": len(cycles),
        "input_dim": int(predictor.input_dim),
        "hidden_dim": int(predictor.hidden_dim),
        "weighted_direct_abs_gap": _weighted_gap(profiles, "abs_gap"),
        "weighted_cumulative_abs_gap": _weighted_gap(profiles, "cumulative_abs_gap"),
        "profiles": [asdict(profile) for profile in profiles],
    }

    json_path = output_dir / "predictor_index_profile.json"
    png_path = output_dir / "predictor_index_profile.png"
    md_path = output_dir / "predictor_index_profile.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot_profiles(png_path, profiles)
    _write_markdown(
        output_path=md_path,
        feature_dir=feature_dir,
        checkpoint=checkpoint,
        predictor=predictor,
        request_count=len(index_payload.get("request_id_to_rid", [])),
        cycle_count=len(cycles),
        profiles=profiles,
        plot_path=png_path,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
