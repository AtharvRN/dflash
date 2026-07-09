from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SUMMARY_KEYS = ("count", "mean", "std", "min", "max", "last", "slope")
LATEST_TARGET_KEYS = ("entropy", "pmax", "margin", "token_logprob", "token_prob")
TOP_LEVEL_KEYS = (
    "cycle_id",
    "prompt_tokens",
    "prefix_len_before_cycle",
    "generated_tokens_before_cycle",
    "remaining_budget",
    "temperature",
)


def _coerce_len(value: Any) -> int:
    value_f = _float_or_none(value)
    if value_f is None:
        return 0
    return max(0, min(15, int(round(value_f))))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _append_optional(features: list[float], value: Any) -> None:
    value_f = _float_or_none(value)
    if value_f is None:
        features.extend((0.0, 0.0))
    else:
        features.extend((value_f, 1.0))


def _append_summary(features: list[float], summary: dict[str, Any] | None) -> None:
    summary = summary or {}
    for key in SUMMARY_KEYS:
        _append_optional(features, summary.get(key))


def _sequence_features(inputs: dict[str, Any], *, feature_set: str = "all") -> np.ndarray:
    if feature_set == "internal_window":
        window = inputs.get("dflash_context_window")
        if not window:
            raise ValueError("feature-set=internal_window requires traces collected with --log-internal-features")
        mask = inputs.get("dflash_context_window_mask", [1] * len(window))
        rows = []
        for i, vector in enumerate(window):
            m = float(mask[i]) if i < len(mask) else 1.0
            rows.append([float(x) for x in vector] + [m])
        return np.asarray(rows, dtype=np.float32)

    if feature_set == "internal_only":
        return np.zeros((1, 1), dtype=np.float32)
    target_entropy = inputs.get("last_target_entropy", [])
    target_logprob = inputs.get("last_target_logprob", [])
    draft_entropy = inputs.get("last_draft_entropy", [])
    draft_mask = inputs.get("last_draft_entropy_mask", [])
    window = max(len(target_entropy), len(target_logprob), len(draft_entropy), len(draft_mask))
    rows: list[list[float]] = []
    for i in range(window):
        te = _float_or_none(target_entropy[i] if i < len(target_entropy) else None)
        if feature_set == "target_entropy_only":
            rows.append([0.0 if te is None else te, 0.0 if te is None else 1.0])
            continue
        tl = _float_or_none(target_logprob[i] if i < len(target_logprob) else None)
        de = _float_or_none(draft_entropy[i] if i < len(draft_entropy) else None)
        dm = _float_or_none(draft_mask[i] if i < len(draft_mask) else None)
        pos = 0.0 if window <= 1 else i / (window - 1)
        rows.append(
            [
                0.0 if te is None else te,
                0.0 if te is None else 1.0,
                0.0 if tl is None else tl,
                0.0 if tl is None else 1.0,
                0.0 if de is None else de,
                0.0 if de is None else (1.0 if dm is None else dm),
                pos,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _target_entropy_sequence_from_history(values: list[float], window: int) -> np.ndarray:
    tail = values[-window:]
    padded: list[float | None] = [None] * (window - len(tail)) + tail
    return np.asarray(
        [[0.0 if value is None else float(value), 0.0 if value is None else 1.0] for value in padded],
        dtype=np.float32,
    )


def _append_current_target_entropies(row: dict[str, Any], history: list[float]) -> None:
    accepted_len = _coerce_len(row.get("labels", {}).get("accepted_draft_len"))
    debug = row.get("current_cycle_debug", {})
    for value in debug.get("target_entropy_for_draft_by_pos", [])[:accepted_len]:
        value_f = _float_or_none(value)
        if value_f is not None:
            history.append(value_f)
    correction_entropy = _float_or_none(debug.get("correction_target", {}).get("entropy"))
    if correction_entropy is not None:
        history.append(correction_entropy)


def _history_features(history: dict[str, list[int]] | None, history_cycles: int) -> list[float]:
    features: list[float] = []
    history = history or {}
    accepted = history.get("accepted", [])[-history_cycles:]
    first_reject = history.get("first_reject", [])[-history_cycles:]
    full_accept = history.get("full_accept", [])[-history_cycles:]
    pad = history_cycles - len(accepted)
    accepted_padded = [0] * pad + accepted
    first_reject_padded = [0] * pad + first_reject
    full_accept_padded = [0] * pad + full_accept

    for value in accepted_padded:
        features.append(float(value))
    for value in first_reject_padded:
        features.append(float(value))
    for value in full_accept_padded:
        features.append(float(value))
    for value in accepted_padded:
        features.append(1.0 if value > 0 else 0.0)
    _append_summary(features, {"count": len(accepted), "mean": np.mean(accepted) if accepted else None,
                               "std": np.std(accepted) if accepted else None,
                               "min": min(accepted) if accepted else None,
                               "max": max(accepted) if accepted else None,
                               "last": accepted[-1] if accepted else None,
                               "slope": accepted[-1] - accepted[0] if len(accepted) > 1 else 0.0 if accepted else None})
    return features


def _static_features(
    row: dict[str, Any],
    *,
    history: dict[str, list[int]] | None = None,
    history_cycles: int = 8,
    feature_set: str = "all",
) -> np.ndarray:
    if feature_set == "internal_window":
        return np.zeros((0,), dtype=np.float32)

    if feature_set == "internal_only":
        vector = row["inputs"].get("dflash_context")
        if not vector:
            raise ValueError("feature-set=internal_only requires traces collected with --log-internal-features")
        return np.asarray([float(x) for x in vector], dtype=np.float32)

    if feature_set == "target_entropy_only":
        return np.zeros((0,), dtype=np.float32)

    inputs = row["inputs"]
    features: list[float] = []

    for key in TOP_LEVEL_KEYS:
        _append_optional(features, row.get(key))

    latest_target = inputs.get("latest_target", {})
    for key in LATEST_TARGET_KEYS:
        _append_optional(features, latest_target.get(key))

    prev_cycle = inputs.get("prev_cycle", {})
    features.append(1.0 if prev_cycle.get("exists") else 0.0)
    _append_optional(features, prev_cycle.get("accepted_draft_len"))
    _append_optional(features, prev_cycle.get("committed_len"))
    _append_optional(features, prev_cycle.get("first_reject_pos"))
    _append_summary(features, prev_cycle.get("accepted_draft_entropy"))
    _append_summary(features, prev_cycle.get("accepted_target_entropy"))
    _append_summary(features, prev_cycle.get("accepted_target_logprob"))

    _append_summary(features, inputs.get("rolling_accept_len"))
    _append_summary(features, inputs.get("last_target_entropy_summary"))
    _append_summary(features, inputs.get("last_target_logprob_summary"))
    _append_summary(features, inputs.get("last_draft_entropy_summary"))
    features.extend(_history_features(history, history_cycles))
    if feature_set == "internal_plus_all":
        vector = inputs.get("dflash_context")
        if not vector:
            raise ValueError("feature-set=internal_plus_all requires traces collected with --log-internal-features")
        features.extend(float(x) for x in vector)

    return np.asarray(features, dtype=np.float32)


def _load_jsonl(
    path: Path,
    *,
    max_rows: int | None,
    seed: int,
    sample: bool,
    include_cycle_history: bool,
    history_cycles: int,
    feature_set: str,
    reconstruct_target_entropy_history: bool,
    sequence_window: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    if max_rows is None or not sample:
        with path.open() as f:
            for line_idx, line in enumerate(f):
                if max_rows is not None and line_idx >= max_rows:
                    break
                rows.append(json.loads(line))
    else:
        rng = random.Random(seed)
        with path.open() as f:
            for line_idx, line in enumerate(f):
                if line_idx < max_rows:
                    rows.append(json.loads(line))
                    continue
                j = rng.randint(0, line_idx)
                if j < max_rows:
                    rows[j] = json.loads(line)

    if not rows:
        raise ValueError(f"no rows loaded from {path}")

    seq_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    histories: dict[str, dict[str, list[int]]] = {}
    target_entropy_histories: dict[str, list[float]] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        history = histories.setdefault(prompt_id, {"accepted": [], "first_reject": [], "full_accept": []})
        if reconstruct_target_entropy_history:
            target_history = target_entropy_histories.setdefault(prompt_id, [])
            if not target_history:
                for value in row["inputs"].get("last_target_entropy", []):
                    value_f = _float_or_none(value)
                    if value_f is not None:
                        target_history.append(value_f)
            window = sequence_window or len(row["inputs"].get("last_target_entropy", [])) or 50
            seq_rows.append(_target_entropy_sequence_from_history(target_history, window))
        else:
            seq_rows.append(_sequence_features(row["inputs"], feature_set=feature_set))
        static_rows.append(
            _static_features(
                row,
                history=history if include_cycle_history else None,
                history_cycles=history_cycles,
                feature_set=feature_set,
            )
        )
        accepted = _coerce_len(row["labels"]["accepted_draft_len"])
        first_reject = row["labels"].get("first_reject_pos")
        history["accepted"].append(accepted)
        history["first_reject"].append(_coerce_len(first_reject))
        history["full_accept"].append(1 if first_reject is None else 0)
        if reconstruct_target_entropy_history:
            _append_current_target_entropies(row, target_entropy_histories[prompt_id])

    seq = np.stack(seq_rows)
    static = np.stack(static_rows)
    survival = np.asarray([row["labels"]["draft_survival"] for row in rows], dtype=np.float32)
    accepted_len = np.asarray([row["labels"]["accepted_draft_len"] for row in rows], dtype=np.float32)
    return seq, static, survival, accepted_len


def _take_rows(x: np.ndarray, *, max_rows: int | None, seed: int, sample: bool) -> np.ndarray:
    if max_rows is None or max_rows >= x.shape[0]:
        return np.asarray(x, dtype=np.float32)
    if not sample:
        return np.asarray(x[:max_rows], dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))
    return np.asarray(x[indices], dtype=np.float32)


def _load_cache(
    path: Path,
    *,
    max_rows: int | None,
    seed: int,
    sample: bool,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if feature_set not in {"internal_only", "internal_window"}:
        raise ValueError("--train-cache-dir/--val-cache-dir only support internal_only and internal_window feature sets")

    survival = _take_rows(
        np.load(path / "survival.npy", mmap_mode="r"),
        max_rows=max_rows,
        seed=seed,
        sample=sample,
    )
    accepted_len = _take_rows(
        np.load(path / "accepted_len.npy", mmap_mode="r"),
        max_rows=max_rows,
        seed=seed,
        sample=sample,
    )
    if feature_set == "internal_only":
        static = _take_rows(
            np.load(path / "dflash_context.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=seed,
            sample=sample,
        )
        seq = np.zeros((static.shape[0], 1, 1), dtype=np.float32)
    else:
        seq = _take_rows(
            np.load(path / "dflash_context_window_seq.npy", mmap_mode="r"),
            max_rows=max_rows,
            seed=seed,
            sample=sample,
        )
        static = np.zeros((seq.shape[0], 0), dtype=np.float32)
    return seq, static, survival, accepted_len


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        if x.shape[-1] == 0:
            return cls(mean=[], std=[])
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32).tolist(), std=std.astype(np.float32).tolist())

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] == 0:
            return x.astype(np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((x - mean) / std).astype(np.float32)


class TraceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        seq: np.ndarray,
        static: np.ndarray,
        survival: np.ndarray,
        accepted_len: np.ndarray,
    ) -> None:
        self.seq = torch.from_numpy(seq)
        self.static = torch.from_numpy(static)
        self.survival = torch.from_numpy(survival)
        self.accepted_len = torch.from_numpy(accepted_len)

    def __len__(self) -> int:
        return int(self.survival.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.seq[idx], self.static[idx], self.survival[idx], self.accepted_len[idx]


class TemporalSurvivalHead(nn.Module):
    def __init__(
        self,
        *,
        seq_dim: int,
        static_dim: int,
        num_slots: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_slots),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(seq)
        return self.head(torch.cat([h[-1], static], dim=-1))


class FlatSurvivalHead(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        seq_dim: int,
        static_dim: int,
        num_slots: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(seq_len * seq_dim + static_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_slots),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([seq.flatten(1), static], dim=-1))


def _metrics(logits: torch.Tensor, survival: torch.Tensor, accepted_len: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, survival).item()
    expected_len = probs.sum(dim=-1)
    pred_len = (probs >= 0.5).sum(dim=-1).float()
    return {
        "bce": bce,
        "expected_len_mae": (expected_len - accepted_len).abs().mean().item(),
        "threshold_len_mae": (pred_len - accepted_len).abs().mean().item(),
        "threshold_len_exact": (pred_len == accepted_len).float().mean().item(),
        "mean_pred_len": pred_len.mean().item(),
        "mean_expected_len": expected_len.mean().item(),
        "mean_target_len": accepted_len.mean().item(),
    }


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    length_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    count = 0
    for seq, static, survival, accepted_len in loader:
        seq = seq.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)
        accepted_len = accepted_len.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(seq, static)
            bce = F.binary_cross_entropy_with_logits(logits, survival)
            expected_len = torch.sigmoid(logits).sum(dim=-1)
            length_loss = F.smooth_l1_loss(expected_len, accepted_len)
            loss = bce + length_loss_weight * length_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = seq.shape[0]
        metrics = _metrics(logits.detach(), survival, accepted_len)
        metrics["loss"] = loss.detach().item()
        metrics["length_loss"] = length_loss.detach().item()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size

    return {key: value / count for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFlash pre-draft survival predictor head.")
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--train-cache-dir", type=Path, default=None)
    parser.add_argument("--val-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--length-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--sample-train-rows", action="store_true")
    parser.add_argument("--include-cycle-history", action="store_true")
    parser.add_argument("--history-cycles", type=int, default=8)
    parser.add_argument(
        "--feature-set",
        choices=["all", "target_entropy_only", "internal_only", "internal_window", "internal_plus_all"],
        default="all",
    )
    parser.add_argument("--reconstruct-target-entropy-history", action="store_true")
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.train_jsonl is None) == (args.train_cache_dir is None):
        parser.error("provide exactly one of --train-jsonl or --train-cache-dir")
    if (args.val_jsonl is None) == (args.val_cache_dir is None):
        parser.error("provide exactly one of --val-jsonl or --val-cache-dir")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_cache_dir is not None:
        print(f"loading train cache from {args.train_cache_dir}", flush=True)
        train_seq, train_static, train_survival, train_len = _load_cache(
            args.train_cache_dir,
            max_rows=args.max_train_rows,
            seed=args.seed,
            sample=args.sample_train_rows,
            feature_set=args.feature_set,
        )
    else:
        print(f"loading train rows from {args.train_jsonl}", flush=True)
        train_seq, train_static, train_survival, train_len = _load_jsonl(
            args.train_jsonl,
            max_rows=args.max_train_rows,
            seed=args.seed,
            sample=args.sample_train_rows,
            include_cycle_history=args.include_cycle_history,
            history_cycles=args.history_cycles,
            feature_set=args.feature_set,
            reconstruct_target_entropy_history=args.reconstruct_target_entropy_history,
            sequence_window=args.sequence_window,
        )
    if args.val_cache_dir is not None:
        print(f"loading validation cache from {args.val_cache_dir}", flush=True)
        val_seq, val_static, val_survival, val_len = _load_cache(
            args.val_cache_dir,
            max_rows=args.max_val_rows,
            seed=args.seed,
            sample=False,
            feature_set=args.feature_set,
        )
    else:
        print(f"loading validation rows from {args.val_jsonl}", flush=True)
        val_seq, val_static, val_survival, val_len = _load_jsonl(
            args.val_jsonl,
            max_rows=args.max_val_rows,
            seed=args.seed,
            sample=False,
            include_cycle_history=args.include_cycle_history,
            history_cycles=args.history_cycles,
            feature_set=args.feature_set,
            reconstruct_target_entropy_history=args.reconstruct_target_entropy_history,
            sequence_window=args.sequence_window,
        )

    seq_norm = Normalizer.fit(train_seq)
    static_norm = Normalizer.fit(train_static)
    train_seq = seq_norm.apply(train_seq)
    val_seq = seq_norm.apply(val_seq)
    train_static = static_norm.apply(train_static)
    val_static = static_norm.apply(val_static)

    train_ds = TraceDataset(train_seq, train_static, train_survival, train_len)
    val_ds = TraceDataset(val_seq, val_static, val_survival, val_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_slots = int(train_survival.shape[1])
    if args.architecture == "gru":
        model: nn.Module = TemporalSurvivalHead(
            seq_dim=int(train_seq.shape[-1]),
            static_dim=int(train_static.shape[-1]),
            num_slots=num_slots,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    else:
        model = FlatSurvivalHead(
            seq_len=int(train_seq.shape[1]),
            seq_dim=int(train_seq.shape[-1]),
            static_dim=int(train_static.shape[-1]),
            num_slots=num_slots,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(
        {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "seq_shape": list(train_seq.shape[1:]),
            "static_dim": int(train_static.shape[-1]),
            "num_slots": num_slots,
            "seq_normalizer": asdict(seq_norm),
            "static_normalizer": asdict(static_norm),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            length_loss_weight=args.length_loss_weight,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            length_loss_weight=args.length_loss_weight,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with (args.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        if val_metrics["expected_len_mae"] < best_val:
            best_val = val_metrics["expected_len_mae"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "val_metrics": history[-1]["val"],
        },
        args.output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
