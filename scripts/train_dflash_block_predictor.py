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
            "existing predictor feature shards. This baseline predicts token-level "
            "acceptance from hidden states only and derives verify length later."
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
    ) -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.shard_names = list(shard_names)
        self.request_splits = list(request_splits)
        self.split = str(split)
        self.max_rows_per_shard = int(max_rows_per_shard)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
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
            token_accepted = payload["token_accepted"].index_select(0, row_indices).to(
                torch.float32
            )

            # SpecDec++ conditions only on the hidden state for the current draft token.
            features = draft_hidden
            labels = token_accepted.unsqueeze(1)

            permutation = torch.randperm(features.shape[0])
            features = features.index_select(0, permutation)
            labels = labels.index_select(0, permutation)
            for feat, label in zip(features, labels, strict=True):
                yield feat, label


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


def _iterate_batches(
    dataset: IterableDataset,
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for feat, label in dataset:
        feats.append(feat)
        labels.append(label)
        if len(feats) >= batch_size:
            yield torch.stack(feats, dim=0), torch.stack(labels, dim=0)
            feats.clear()
            labels.clear()
    if feats:
        yield torch.stack(feats, dim=0), torch.stack(labels, dim=0)


def _run_epoch(
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
    criterion = nn.BCEWithLogitsLoss()
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    total_positive = 0.0
    total_pred_positive = 0.0

    batch_idx = 0
    for batch_idx, (features, labels) in enumerate(
        _iterate_batches(dataset, batch_size=batch_size), start=1
    ):
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)

        logits = model(features)
        loss = criterion(logits, labels)

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
    if not math.isclose(train_frac + val_frac, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_frac + val_frac must equal 1.0")

    request_splits = _build_request_splits(index_payload, train_frac=train_frac)
    device = _resolve_device(str(args.device))

    first_payload = torch.load(feature_dir / shard_names[0], map_location="cpu")
    input_dim = int(first_payload["draft_hidden"].shape[1]) + 3
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
        train_dataset = PredictorShardDataset(
            feature_dir=feature_dir,
            shard_names=shard_names,
            request_splits=request_splits,
            split="train",
            max_rows_per_shard=int(args.max_rows_per_shard),
        )
        train_epoch = _run_epoch(
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

        with torch.no_grad():
            val_dataset = PredictorShardDataset(
                feature_dir=feature_dir,
                shard_names=shard_names,
                request_splits=request_splits,
                split="val",
                max_rows_per_shard=int(args.max_rows_per_shard),
            )
            val_epoch = _run_epoch(
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

        print(
            json.dumps(
                {
                    "epoch": epoch_idx,
                    "train": train_epoch.__dict__,
                    "val": val_epoch.__dict__,
                }
            )
        )

    metrics = {
        "args": vars(args),
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
