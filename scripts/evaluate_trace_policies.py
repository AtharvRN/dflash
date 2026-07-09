from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable


PolicyFn = Callable[[dict, dict[str, deque[int]]], int]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clip_len(value: float | int | None) -> int:
    if value is None:
        return 0
    return max(0, min(15, int(round(float(value)))))


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _label_len(row: dict) -> int:
    return _clip_len(row["labels"]["accepted_draft_len"])


def _entropy(row: dict) -> float:
    value = _safe_float(row["inputs"]["latest_target"].get("entropy"))
    return 0.0 if value is None else value


def _evaluate_rows(rows: list[dict], policy_name: str, policy: PolicyFn) -> dict[str, float | int | str]:
    histories: dict[str, dict[str, deque[int]]] = defaultdict(lambda: {"accepted": deque(maxlen=16)})
    total_abs = 0.0
    exact = 0
    pred_sum = 0.0
    target_sum = 0.0
    committed_sum = 0.0
    drafted_sum = 0.0
    wasted_sum = 0.0
    under = 0
    over = 0
    for row in rows:
        prompt_id = str(row["prompt_id"])
        history = histories[prompt_id]
        target = _label_len(row)
        pred = _clip_len(policy(row, history))
        total_abs += abs(pred - target)
        exact += int(pred == target)
        pred_sum += pred
        target_sum += target
        committed_sum += min(pred, target) + 1
        drafted_sum += pred
        wasted_sum += max(0, pred - target)
        under += int(pred < target)
        over += int(pred > target)
        history["accepted"].append(target)

    n = len(rows)
    return {
        "name": policy_name,
        "rows": n,
        "len_mae": total_abs / n,
        "len_exact": exact / n,
        "mean_pred_len": pred_sum / n,
        "mean_target_len": target_sum / n,
        "mean_committed": committed_sum / n,
        "mean_drafted": drafted_sum / n,
        "mean_wasted_drafts": wasted_sum / n,
        "committed_per_drafted": committed_sum / max(1e-9, drafted_sum),
        "underdraft_rate": under / n,
        "overdraft_rate": over / n,
    }


def _fixed_policy(k: int) -> PolicyFn:
    return lambda row, history: k


def _previous_policy(default: int) -> PolicyFn:
    def policy(row: dict, history: dict[str, deque[int]]) -> int:
        accepted = history["accepted"]
        return accepted[-1] if accepted else default

    return policy


def _rolling_mean_policy(default: int, window: int, mode: str) -> PolicyFn:
    def policy(row: dict, history: dict[str, deque[int]]) -> int:
        values = list(history["accepted"])[-window:]
        if not values:
            return default
        avg = sum(values) / len(values)
        if mode == "floor":
            return math.floor(avg)
        if mode == "ceil":
            return math.ceil(avg)
        return round(avg)

    return policy


def _entropy_gate_policy(threshold: float, low_len: int, high_len: int) -> PolicyFn:
    def policy(row: dict, history: dict[str, deque[int]]) -> int:
        return high_len if _entropy(row) <= threshold else low_len

    return policy


def _fit_fixed(train_rows: list[dict]) -> tuple[int, dict[str, float | int | str]]:
    candidates = [(_evaluate_rows(train_rows, f"fixed_{k}", _fixed_policy(k)), k) for k in range(16)]
    best, k = min(candidates, key=lambda x: x[0]["len_mae"])
    return k, best


def _fit_entropy_gate(train_rows: list[dict]) -> tuple[tuple[float, int, int], dict[str, float | int | str]]:
    entropies = sorted(_entropy(row) for row in train_rows)
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [entropies[min(len(entropies) - 1, int(q * (len(entropies) - 1)))] for q in quantiles]
    best_result: dict[str, float | int | str] | None = None
    best_params = (thresholds[0], 0, 15)
    for threshold in thresholds:
        for low_len in range(0, 16):
            for high_len in range(low_len, 16):
                name = f"entropy_gate_t{threshold:.3f}_lo{low_len}_hi{high_len}"
                result = _evaluate_rows(train_rows, name, _entropy_gate_policy(threshold, low_len, high_len))
                if best_result is None or result["len_mae"] < best_result["len_mae"]:
                    best_result = result
                    best_params = (threshold, low_len, high_len)
    assert best_result is not None
    return best_params, best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple pre-draft length policies on DFlash traces.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-fit-rows", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = _load_rows(args.train_jsonl)
    eval_rows = _load_rows(args.eval_jsonl)
    fit_rows = train_rows[: args.max_fit_rows] if args.max_fit_rows else train_rows

    best_fixed_k, best_fixed_train = _fit_fixed(fit_rows)
    best_entropy_params, best_entropy_train = _fit_entropy_gate(fit_rows)
    threshold, low_len, high_len = best_entropy_params
    default_len = best_fixed_k

    policies: list[tuple[str, PolicyFn]] = []
    for k in range(16):
        policies.append((f"fixed_{k}", _fixed_policy(k)))
    policies.extend(
        [
            (f"best_fixed_train_{best_fixed_k}", _fixed_policy(best_fixed_k)),
            ("previous_len", _previous_policy(default_len)),
            ("rolling_mean_4_round", _rolling_mean_policy(default_len, 4, "round")),
            ("rolling_mean_8_round", _rolling_mean_policy(default_len, 8, "round")),
            ("rolling_mean_8_floor", _rolling_mean_policy(default_len, 8, "floor")),
            ("rolling_mean_8_ceil", _rolling_mean_policy(default_len, 8, "ceil")),
            (
                f"best_entropy_gate_t{threshold:.3f}_lo{low_len}_hi{high_len}",
                _entropy_gate_policy(threshold, low_len, high_len),
            ),
        ]
    )

    results = [_evaluate_rows(eval_rows, name, policy) for name, policy in policies]
    payload = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "fit": {
            "fit_rows": len(fit_rows),
            "best_fixed": best_fixed_train,
            "best_entropy_gate": best_entropy_train,
            "best_entropy_gate_params": {
                "threshold": threshold,
                "low_len": low_len,
                "high_len": high_len,
            },
        },
        "results": sorted(results, key=lambda x: x["len_mae"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
