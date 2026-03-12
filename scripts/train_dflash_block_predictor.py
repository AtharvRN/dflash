#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import IterableDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a SpecDec++-style offline DFLASH accept predictor from the "
            "existing predictor feature shards. The predictor uses hidden states "
            "only and is trained for direct prefix-survival stopping."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-rows-per-shard", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument(
        "--objective",
        choices=["prefix_bce", "hazard"],
        default="prefix_bce",
        help=(
            "Training objective. 'prefix_bce' matches the current direct prefix-survival "
            "baseline. 'hazard' trains a discrete-time first-rejection hazard model and "
            "derives prefix survival as a cumulative product."
        ),
    )
    parser.add_argument(
        "--loss-weight-mode",
        choices=["uniform", "boundary"],
        default="boundary",
        help=(
            "Loss weighting scheme. 'boundary' upweights draft positions near the "
            "true verify boundary, which is the control target that matters online."
        ),
    )
    parser.add_argument(
        "--boundary-weight-alpha",
        type=float,
        default=4.0,
        help="Additional weight scale for samples near the true verify boundary.",
    )
    parser.add_argument(
        "--boundary-weight-tau",
        type=float,
        default=1.5,
        help="Distance scale for boundary-aware weighting.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_index_path(feature_dir: Path) -> Path:
    matches = sorted(feature_dir.glob("*_index.json"))
    if not matches:
        raise FileNotFoundError(f"No *_index.json found under {feature_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected one index file under {feature_dir}, found {len(matches)}."
        )
    return matches[0]


def _split_name_for_request_id(rid: str, *, train_frac: float) -> str:
    digest = hashlib.md5(rid.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "train" if score < train_frac else "val"


@dataclass
class ShardSpec:
    path: Path
    split: str


def _build_request_splits(index_payload: dict, *, train_frac: float) -> list[str]:
    request_id_to_rid = index_payload.get("request_id_to_rid")
    if not isinstance(request_id_to_rid, list) or not request_id_to_rid:
        raise ValueError("Index file missing request_id_to_rid list.")
    return [
        _split_name_for_request_id(str(rid), train_frac=train_frac)
        for rid in request_id_to_rid
    ]


class PredictorShardDataset(IterableDataset):
    def __init__(
        self,
        *,
        feature_dir: Path,
        shard_names: list[str],
        request_splits: list[str],
        split: str,
        max_rows_per_shard: int = 0,
        loss_weight_mode: str = "uniform",
        boundary_weight_alpha: float = 4.0,
        boundary_weight_tau: float = 1.5,
    ) -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.shard_names = list(shard_names)
        self.request_splits = list(request_splits)
        self.split = str(split)
        self.max_rows_per_shard = int(max_rows_per_shard)
        self.loss_weight_mode = str(loss_weight_mode)
        self.boundary_weight_alpha = float(boundary_weight_alpha)
        self.boundary_weight_tau = float(max(boundary_weight_tau, 1e-6))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for shard_name in self.shard_names:
            payload = torch.load(self.feature_dir / shard_name, map_location="cpu")
            request_ids = payload["request_id"].to(torch.long)
            split_mask = torch.tensor(
                [self.request_splits[int(i)] == self.split for i in request_ids.tolist()],
                dtype=torch.bool,
            )
            row_indices = torch.nonzero(split_mask, as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            if self.max_rows_per_shard > 0 and row_indices.numel() > self.max_rows_per_shard:
                row_indices = row_indices[: self.max_rows_per_shard]

            draft_hidden = payload["draft_hidden"].index_select(0, row_indices).to(torch.float32)
            draft_pos = payload["draft_pos"].index_select(0, row_indices).to(torch.float32)
            accepted_draft_tokens = payload["accepted_draft_tokens"].index_select(
                0, row_indices
            ).to(torch.float32)
            runtime_block_size = payload["runtime_block_size"].index_select(
                0, row_indices
            ).to(torch.float32)

            true_boundary = torch.clamp(accepted_draft_tokens + 1.0, max=runtime_block_size)
            # With the current shard indexing (draft positions start at 1), the stopping label
            # "keep going past this position" is equivalent to token acceptance.
            labels = (true_boundary > draft_pos).to(torch.float32).unsqueeze(1)

            if self.loss_weight_mode == "boundary":
                boundary_dist = torch.abs(true_boundary - draft_pos)
                weights = 1.0 + self.boundary_weight_alpha * torch.exp(
                    -boundary_dist / self.boundary_weight_tau
                )
            else:
                weights = torch.ones_like(draft_pos, dtype=torch.float32)

            # The predictor still conditions only on the hidden state for the current draft token.
            features = draft_hidden
            weights = weights.unsqueeze(1)

            permutation = torch.randperm(features.shape[0])
            features = features.index_select(0, permutation)
            labels = labels.index_select(0, permutation)
            weights = weights.index_select(0, permutation)
            for feat, label, weight in zip(features, labels, weights, strict=True):
                yield feat, label, weight


class PredictorShardCycleDataset(IterableDataset):
    def __init__(
        self,
        *,
        feature_dir: Path,
        shard_names: list[str],
        request_splits: list[str],
        split: str,
        max_rows_per_shard: int = 0,
    ) -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.shard_names = list(shard_names)
        self.request_splits = list(request_splits)
        self.split = str(split)
        self.max_rows_per_shard = int(max_rows_per_shard)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, int]]:
        for shard_name in self.shard_names:
            payload = torch.load(self.feature_dir / shard_name, map_location="cpu")
            request_ids = payload["request_id"].to(torch.long)
            split_mask = torch.tensor(
                [self.request_splits[int(i)] == self.split for i in request_ids.tolist()],
                dtype=torch.bool,
            )
            row_indices = torch.nonzero(split_mask, as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            if self.max_rows_per_shard > 0 and row_indices.numel() > self.max_rows_per_shard:
                row_indices = row_indices[: self.max_rows_per_shard]

            request_ids = payload["request_id"].index_select(0, row_indices).to(torch.int64)
            cycle_indices = payload["cycle_idx"].index_select(0, row_indices).to(torch.int64)
            draft_pos = payload["draft_pos"].index_select(0, row_indices).to(torch.int64)
            draft_hidden = payload["draft_hidden"].index_select(0, row_indices).to(torch.float32)
            accepted_draft_tokens = payload["accepted_draft_tokens"].index_select(
                0, row_indices
            ).to(torch.int64)

            start = 0
            total = int(row_indices.numel())
            while start < total:
                req_id = int(request_ids[start].item())
                cycle_idx = int(cycle_indices[start].item())
                end = start + 1
                while end < total:
                    if int(request_ids[end].item()) != req_id:
                        break
                    if int(cycle_indices[end].item()) != cycle_idx:
                        break
                    end += 1

                positions = draft_pos[start:end]
                expected_positions = torch.arange(
                    1, 1 + int(positions.numel()), dtype=torch.int64
                )
                if not torch.equal(positions.cpu(), expected_positions):
                    raise RuntimeError(
                        "Hazard training expects contiguous draft positions within each cycle. "
                        f"request_id={req_id} cycle_idx={cycle_idx} "
                        f"positions={positions.tolist()}"
                    )

                features = draft_hidden[start:end]
                accepted = int(accepted_draft_tokens[start].item())
                proposed = int(features.shape[0])
                event_index = accepted if accepted < proposed else -1
                yield features, proposed, event_index
                start = end


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
class EpochMetrics:
    loss: float
    accuracy: float
    positive_rate: float
    predicted_positive_rate: float
    rows: int
    examples: int


def _iterate_batches(
    dataset: IterableDataset,
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for feat, label, weight in dataset:
        feats.append(feat)
        labels.append(label)
        weights.append(weight)
        if len(feats) >= batch_size:
            yield (
                torch.stack(feats, dim=0),
                torch.stack(labels, dim=0),
                torch.stack(weights, dim=0),
            )
            feats.clear()
            labels.clear()
            weights.clear()
    if feats:
        yield (
            torch.stack(feats, dim=0),
            torch.stack(labels, dim=0),
            torch.stack(weights, dim=0),
        )


def _iterate_sequence_batches(
    dataset: IterableDataset,
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    feats: list[torch.Tensor] = []
    lengths: list[int] = []
    event_indices: list[int] = []
    for feat, length, event_index in dataset:
        feats.append(feat)
        lengths.append(int(length))
        event_indices.append(int(event_index))
        if len(feats) >= batch_size:
            max_len = max(lengths)
            input_dim = int(feats[0].shape[1])
            padded = torch.zeros((len(feats), max_len, input_dim), dtype=torch.float32)
            for idx, item in enumerate(feats):
                padded[idx, : int(item.shape[0]), :] = item
            yield (
                padded,
                torch.tensor(lengths, dtype=torch.int64),
                torch.tensor(event_indices, dtype=torch.int64),
            )
            feats.clear()
            lengths.clear()
            event_indices.clear()
    if feats:
        max_len = max(lengths)
        input_dim = int(feats[0].shape[1])
        padded = torch.zeros((len(feats), max_len, input_dim), dtype=torch.float32)
        for idx, item in enumerate(feats):
            padded[idx, : int(item.shape[0]), :] = item
        yield (
            padded,
            torch.tensor(lengths, dtype=torch.int64),
            torch.tensor(event_indices, dtype=torch.int64),
        )


def _run_prefix_bce_epoch(
    *,
    model: nn.Module,
    dataset: IterableDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    print_every: int,
    epoch_idx: int,
    split: str,
) -> EpochMetrics:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    total_positive = 0.0
    total_pred_positive = 0.0

    batch_idx = 0
    for batch_idx, (features, labels, weights) in enumerate(
        _iterate_batches(dataset, batch_size=batch_size), start=1
    ):
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)

        logits = model(features)
        per_row_loss = criterion(logits, labels)
        loss = (per_row_loss * weights).sum() / torch.clamp(weights.sum(), min=1e-6)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.float32)
            rows = int(labels.shape[0])
            total_rows += rows
            total_loss += float(loss.item()) * rows
            total_correct += int((preds == labels).sum().item())
            total_positive += float(labels.sum().item())
            total_pred_positive += float(preds.sum().item())

        if print_every > 0 and batch_idx % print_every == 0:
            print(
                f"[{split}] epoch={epoch_idx} batch={batch_idx} "
                f"rows={total_rows} loss={total_loss / max(total_rows, 1):.6f}"
            )

    if total_rows <= 0:
        raise RuntimeError(f"No rows were observed for split={split}.")

    return EpochMetrics(
        loss=total_loss / total_rows,
        accuracy=total_correct / total_rows,
        positive_rate=total_positive / total_rows,
        predicted_positive_rate=total_pred_positive / total_rows,
        rows=total_rows,
        examples=total_rows,
    )


def _hazard_survival_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hazards = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return hazards, survival


def _run_hazard_epoch(
    *,
    model: nn.Module,
    dataset: IterableDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    print_every: int,
    epoch_idx: int,
    split: str,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    total_rows = 0
    total_correct = 0
    total_positive = 0.0
    total_pred_positive = 0.0

    batch_idx = 0
    for batch_idx, (features, lengths, event_indices) in enumerate(
        _iterate_sequence_batches(dataset, batch_size=batch_size), start=1
    ):
        features = features.to(device=device, dtype=torch.float32)
        lengths = lengths.to(device=device, dtype=torch.int64)
        event_indices = event_indices.to(device=device, dtype=torch.int64)

        batch_n, max_len, hidden_dim = features.shape
        logits = model(features.reshape(batch_n * max_len, hidden_dim)).reshape(batch_n, max_len)
        hazards, survival = _hazard_survival_from_logits(logits)

        positions = torch.arange(max_len, device=device, dtype=torch.int64).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        has_event = event_indices >= 0
        pre_event_mask = valid_mask & (
            (~has_event).unsqueeze(1) | (positions < event_indices.unsqueeze(1))
        )
        event_mask = has_event.unsqueeze(1) & (positions == event_indices.unsqueeze(1))

        loss_matrix = torch.zeros_like(hazards, dtype=torch.float32)
        loss_matrix = loss_matrix + (-torch.log1p(-hazards)) * pre_event_mask.to(torch.float32)
        loss_matrix = loss_matrix + (-torch.log(hazards)) * event_mask.to(torch.float32)
        per_example_loss = loss_matrix.sum(dim=1)
        loss = per_example_loss.mean()

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            labels = pre_event_mask.to(torch.float32)
            preds = (survival >= 0.5).to(torch.float32)
            valid_rows = int(valid_mask.sum().item())
            total_examples += int(batch_n)
            total_rows += valid_rows
            total_loss += float(loss.item()) * int(batch_n)
            total_correct += int(((preds == labels) & valid_mask).sum().item())
            total_positive += float((labels * valid_mask.to(torch.float32)).sum().item())
            total_pred_positive += float((preds * valid_mask.to(torch.float32)).sum().item())

        if print_every > 0 and batch_idx % print_every == 0:
            print(
                f"[{split}] epoch={epoch_idx} batch={batch_idx} "
                f"examples={total_examples} rows={total_rows} "
                f"loss={total_loss / max(total_examples, 1):.6f}"
            )

    if total_examples <= 0 or total_rows <= 0:
        raise RuntimeError(f"No examples were observed for split={split}.")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_rows,
        positive_rate=total_positive / total_rows,
        predicted_positive_rate=total_pred_positive / total_rows,
        rows=total_rows,
        examples=total_examples,
    )


def _run_epoch(
    *,
    objective: str,
    model: nn.Module,
    dataset: IterableDataset,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    print_every: int,
    epoch_idx: int,
    split: str,
) -> EpochMetrics:
    if objective == "hazard":
        return _run_hazard_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            print_every=print_every,
            epoch_idx=epoch_idx,
            split=split,
        )
    return _run_prefix_bce_epoch(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        device=device,
        batch_size=batch_size,
        print_every=print_every,
        epoch_idx=epoch_idx,
        split=split,
    )


def main() -> None:
    args = _parse_args()
    _set_seed(int(args.seed))

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = _find_index_path(feature_dir)
    index_payload = json.loads(index_path.read_text())
    shard_names = list(index_payload.get("shards", []))
    if not shard_names:
        raise ValueError(f"Index file has no shards: {index_path}")
    if int(args.max_shards) > 0:
        shard_names = shard_names[: int(args.max_shards)]

    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    if val_frac < 0.0:
        raise ValueError("val_frac must be >= 0.0")
    if not math.isclose(train_frac + val_frac, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_frac + val_frac must equal 1.0")

    request_splits = _build_request_splits(index_payload, train_frac=train_frac)
    device = _resolve_device(str(args.device))

    first_payload = torch.load(feature_dir / shard_names[0], map_location="cpu")
    input_dim = int(first_payload["draft_hidden"].shape[1])
    model = AcceptPredictorMLP(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_metrics: list[dict] = []
    val_metrics: list[dict] = []

    for epoch_idx in range(1, int(args.epochs) + 1):
        if str(args.objective) == "hazard":
            train_dataset = PredictorShardCycleDataset(
                feature_dir=feature_dir,
                shard_names=shard_names,
                request_splits=request_splits,
                split="train",
                max_rows_per_shard=int(args.max_rows_per_shard),
            )
        else:
            train_dataset = PredictorShardDataset(
                feature_dir=feature_dir,
                shard_names=shard_names,
                request_splits=request_splits,
                split="train",
                max_rows_per_shard=int(args.max_rows_per_shard),
                loss_weight_mode=str(args.loss_weight_mode),
                boundary_weight_alpha=float(args.boundary_weight_alpha),
                boundary_weight_tau=float(args.boundary_weight_tau),
            )
        train_epoch = _run_epoch(
            objective=str(args.objective),
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            device=device,
            batch_size=int(args.batch_size),
            print_every=int(args.print_every),
            epoch_idx=epoch_idx,
            split="train",
        )
        train_metrics.append(train_epoch.__dict__)

        if val_frac > 0.0:
            with torch.no_grad():
                if str(args.objective) == "hazard":
                    val_dataset = PredictorShardCycleDataset(
                        feature_dir=feature_dir,
                        shard_names=shard_names,
                        request_splits=request_splits,
                        split="val",
                        max_rows_per_shard=int(args.max_rows_per_shard),
                    )
                else:
                    val_dataset = PredictorShardDataset(
                        feature_dir=feature_dir,
                        shard_names=shard_names,
                        request_splits=request_splits,
                        split="val",
                        max_rows_per_shard=int(args.max_rows_per_shard),
                        loss_weight_mode=str(args.loss_weight_mode),
                        boundary_weight_alpha=float(args.boundary_weight_alpha),
                        boundary_weight_tau=float(args.boundary_weight_tau),
                    )
                val_epoch = _run_epoch(
                    objective=str(args.objective),
                    model=model,
                    dataset=val_dataset,
                    optimizer=None,
                    device=device,
                    batch_size=int(args.batch_size),
                    print_every=0,
                    epoch_idx=epoch_idx,
                    split="val",
                )
            val_metrics.append(val_epoch.__dict__)
            val_payload = val_epoch.__dict__
        else:
            val_payload = None

        print(
            json.dumps(
                {
                    "epoch": epoch_idx,
                    "train": train_epoch.__dict__,
                    "val": val_payload,
                }
            )
        )

    metrics = {
        "args": vars(args),
        "output_mode": "hazard" if str(args.objective) == "hazard" else "prefix_survival",
        "index_path": str(index_path),
        "num_shards_used": len(shard_names),
        "request_count": len(request_splits),
        "input_dim": int(input_dim),
        "device": str(device),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")
    print(f"Wrote model to {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
