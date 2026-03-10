#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Optional, TextIO

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    rid: Optional[str],
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
    }
    if rid is not None:
        payload["rid"] = rid
    resp = requests.post(
        base_url + "/generate",
        json=payload,
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    rids: list[str],
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> list[dict]:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "rid": rids,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            f"Expected batched /generate response to be a list, got {type(out).__name__}."
        )
    return out


def _load_prompt_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            turns = row.get("turns")
            if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
                raise ValueError(
                    f"Invalid turns payload at {path}:{line_no}. Expected a non-empty list[str]."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompt rows found in {path}.")
    return rows


def _build_chat_prompts(tokenizer, rows: list[dict]) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["turns"][0]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


def _write_request_manifest(
    path: Path,
    *,
    rows: list[dict],
    prompts: list[str],
    rids: list[str],
    include_prompt_text: bool,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, (row, prompt, rid) in enumerate(zip(rows, prompts, rids, strict=True)):
            out = {
                "request_local_idx": int(i),
                "rid": rid,
                "id": row.get("id"),
                "source_dataset": row.get("source_dataset"),
                "source_config": row.get("source_config"),
                "source_split": row.get("source_split"),
                "source_name": row.get("source_name"),
                "task_family": row.get("task_family"),
                "source_index": row.get("source_index"),
                "sample_rank_within_source": row.get("sample_rank_within_source"),
                "turns": row.get("turns"),
            }
            if include_prompt_text:
                out["prompt"] = prompt
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def _extract_float(meta: dict, key: str) -> Optional[float]:
    if key not in meta:
        return None
    try:
        return float(meta[key])
    except Exception:
        return None


def _write_call_trace_row(
    fp: TextIO,
    *,
    request_local_idx: int,
    rid: str,
    prompt_row: dict,
    out: dict,
) -> None:
    meta = out.get("meta_info") or {}
    row = {
        "request_local_idx": int(request_local_idx),
        "rid": rid,
        "source_name": prompt_row.get("source_name"),
        "task_family": prompt_row.get("task_family"),
        "source_index": prompt_row.get("source_index"),
        "sample_rank_within_source": prompt_row.get("sample_rank_within_source"),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "e2e_latency_s": _extract_float(meta, "e2e_latency"),
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
        "spec_accept_token_num": int(meta.get("spec_accept_token_num", 0)),
        "spec_draft_token_num": int(meta.get("spec_draft_token_num", 0)),
        "spec_accept_length": _extract_float(meta, "spec_accept_length"),
        "spec_accept_rate": _extract_float(meta, "spec_accept_rate"),
        "draft_time_s": _extract_float(meta, "spec_draft_time_s"),
        "verify_time_s": _extract_float(meta, "spec_verify_time_s"),
        "draft_time_per_cycle_s": _extract_float(meta, "spec_draft_time_per_cycle_s"),
        "verify_time_per_cycle_s": _extract_float(meta, "spec_verify_time_per_cycle_s"),
        "spec_cycle_trace": meta.get("spec_cycle_trace"),
    }
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run DFLASH on a local JSONL prompt mix and write binary predictor-training "
            "feature shards from the server."
        )
    )
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--speculative-dflash-block-size", type=int, default=16)
    parser.add_argument("--predictor-shard-rows", type=int, default=100000)
    parser.add_argument("--save-call-trace-path", type=str, default=None)
    parser.add_argument("--include-request-prompt", action="store_true")
    parser.add_argument(
        "--server-extra-args",
        type=str,
        default="",
        help="Raw extra args appended to launch_server (parsed with shlex.split).",
    )
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "feature_shards"
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_prompt_rows(dataset_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = _build_chat_prompts(tokenizer, rows)
    rids = [f"pred-{i:07d}" for i in range(len(rows))]

    request_manifest_path = output_dir / "request_manifest.jsonl"
    _write_request_manifest(
        request_manifest_path,
        rows=rows,
        prompts=prompts,
        rids=rids,
        include_prompt_text=bool(args.include_request_prompt),
    )

    call_trace_path = (
        Path(args.save_call_trace_path)
        if args.save_call_trace_path
        else (output_dir / "collector_calls.jsonl")
    )
    run_manifest_path = output_dir / "run_manifest.json"

    port = find_available_port(20000)
    base_url = f"http://127.0.0.1:{port}"
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        str(args.attention_backend),
        "--tp-size",
        str(int(args.tp_size)),
        "--dtype",
        str(args.dtype),
        "--mem-fraction-static",
        str(float(args.mem_fraction_static)),
        "--max-running-requests",
        str(int(args.max_running_requests)),
        "--cuda-graph-bs",
        *[str(i) for i in range(1, 33)],
        "--cuda-graph-max-bs",
        "32",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        str(args.draft_model),
        "--speculative-dflash-block-size",
        str(int(args.speculative_dflash_block_size)),
        "--speculative-dflash-predictor-dataset-output-dir",
        str(feature_dir),
        "--speculative-dflash-predictor-dataset-shard-rows",
        str(int(args.predictor_shard_rows)),
        "--speculative-dflash-cycle-trace",
    ]
    if args.server_extra_args.strip():
        other_args.extend(shlex.split(args.server_extra_args))

    proc = popen_launch_server(
        args.target_model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )

    t0 = time.time()
    try:
        total_requests = len(prompts)
        with call_trace_path.open("w", encoding="utf-8") as trace_fp:
            for start in range(0, total_requests, int(args.concurrency)):
                chunk_prompts = prompts[start : start + int(args.concurrency)]
                chunk_rows = rows[start : start + int(args.concurrency)]
                chunk_rids = rids[start : start + int(args.concurrency)]
                outs = _send_generate_batch(
                    base_url,
                    chunk_prompts,
                    rids=chunk_rids,
                    max_new_tokens=int(args.max_new_tokens),
                    stop=[],
                    timeout_s=int(args.timeout_s),
                )
                if len(outs) != len(chunk_prompts):
                    raise RuntimeError(
                        f"Expected {len(chunk_prompts)} outputs, got {len(outs)}."
                    )
                for local_idx, (rid, row, out) in enumerate(
                    zip(chunk_rids, chunk_rows, outs, strict=True)
                ):
                    _write_call_trace_row(
                        trace_fp,
                        request_local_idx=start + local_idx,
                        rid=rid,
                        prompt_row=row,
                        out=out,
                    )

        run_manifest = {
            "dataset_jsonl": str(dataset_jsonl),
            "request_manifest_jsonl": str(request_manifest_path),
            "call_trace_jsonl": str(call_trace_path),
            "feature_shard_dir": str(feature_dir),
            "target_model": str(args.target_model),
            "draft_model": str(args.draft_model),
            "tp_size": int(args.tp_size),
            "attention_backend": str(args.attention_backend),
            "dtype": str(args.dtype),
            "concurrency": int(args.concurrency),
            "max_new_tokens": int(args.max_new_tokens),
            "speculative_dflash_block_size": int(args.speculative_dflash_block_size),
            "predictor_shard_rows": int(args.predictor_shard_rows),
            "num_requests": len(rows),
            "server_other_args": other_args,
            "wall_time_s": float(time.time() - t0),
        }
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote request manifest: {request_manifest_path}")
        print(f"Wrote call trace: {call_trace_path}")
        print(f"Wrote run manifest: {run_manifest_path}")
        print(f"Feature shards directory: {feature_dir}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass


if __name__ == "__main__":
    main()
