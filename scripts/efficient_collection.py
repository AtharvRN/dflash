"""Measured prompt concurrency and append-only recovery for actual-block data.

Every worker retains the unchanged single-request numerical path. Concurrency
is across independent prompts, not padded blocks or a different draft size.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256, validate_states
from scripts.collect_policy_granularity import atomic_json, CHECKPOINT_SHA
from scripts.worker_diagnostics import (DiagnosticSpawnContext, controller_event,
    process_resources, worker_event)


DIAGNOSTICS_SOURCE = Path(__file__).with_name("worker_diagnostics.py")


def inventory(root):
    """Read-only, fail-closed audit; a interrupted cache needs no final manifest."""
    config = json.loads((root/"config.json").read_text())
    if config.get("collection_kind") != "training" or config["blocks"] != list(range(2, 17)):
        raise ValueError("Expected the training-only B2--B16 collection")
    planned = config["prompt_ids"]
    if len(planned) != len(set(planned)):
        raise ValueError("Duplicate planned prompt")
    receipts, rows, progress, files = {}, [], [], {"config.json": sha256(root/"config.json")}
    for path in root.glob("receipt_*.json"):
        r = json.loads(path.read_text())
        pid = r["prompt_id"]
        if path.name != f"receipt_{pid}.json" or pid in receipts or pid not in planned:
            raise ValueError("Unexpected or duplicate receipt")
        if r["group"] != "train" or config["prompt_groups"][str(pid)] != "train":
            raise ValueError("Nontraining receipt")
        expected = {f"prompt_{pid}.json"} | ({f"prompt_{pid}_fused.npy"} if r["states"] else set())
        if set(r["files"]) != expected:
            raise ValueError("Unexpected receipt file set")
        for name, digest in r["files"].items():
            if Path(name).name != name or sha256(root/name) != digest:
                raise ValueError("Receipt file hash mismatch")
            files[name] = digest
        files[path.name] = sha256(path)
        receipts[pid] = r
    if set(receipts) != set(planned[:len(receipts)]):
        raise ValueError("Recovery requires a contiguous saved prompt prefix")
    ordered = [receipts[pid] for pid in planned[:len(receipts)]]
    for r in ordered:
        pid = r["prompt_id"]
        shard = json.loads((root/f"prompt_{pid}.json").read_text())
        states = shard["states"]
        validate_states(states, config["blocks"])
        if len(states) != r["states"] or any(int(s["prompt_id"]) != pid or s["group"] != "train" or s["source"] != r["source"] for s in states):
            raise ValueError("Receipt/state alignment mismatch")
        if any(s["canonical_disagreements"] for s in states):
            raise ValueError("Canonical disagreement requires investigation")
        if states:
            x = np.load(root/f"prompt_{pid}_fused.npy", allow_pickle=False)
            if x.shape != (len(states), 2560) or x.dtype != np.float16 or not np.isfinite(x).all():
                raise ValueError("Invalid fused feature alignment")
        rows.extend(states)
        progress.append(shard["progress"])
    # Unreceipted evidence is preserved at the source, never imported as valid.
    ignored = sorted(p.name for p in root.iterdir() if p.is_file() and p.name not in files)
    audit = {"passed": True, "prompt_receipts": len(ordered), "rows": len(rows),
        "eligible_rows": sum(r["eligible"] for r in rows), "files_verified": len(files),
        "source_config_sha256": files["config.json"], "ignored_source_files": ignored,
        "canonical_disagreements": 0}
    return config, ordered, rows, progress, files, audit


def selected_messages(config):
    wanted = set(config["prompt_ids"])
    rows = {}
    for path, digest in config["input_hashes"].items():
        if sha256(path) != digest:
            raise ValueError("An original collection input changed")
    if sha256(config["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("Frozen checkpoint changed")
    split = Path(config["split_dir"])
    train = set(map(int, json.loads((split/"train_prompt_ids.json").read_text())["train_prompt_ids"]))
    val = set(map(int, json.loads((split/"val_prompt_ids.json").read_text())["val_prompt_ids"]))
    if train & val or not wanted <= train or wanted & val:
        raise ValueError("Planned recovery prompts violate the canonical training split")
    with Path(config["manifest"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = int(row["manifest_index"])
            if pid not in wanted:
                continue
            digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if pid in rows or digest != config["prompt_content_hashes"][str(pid)]:
                raise ValueError("Original prompt identity/content changed")
            rows[pid] = {**row, "group": "train"}
    if set(rows) != wanted:
        raise ValueError("Missing planned prompts")
    sampler = Path("scripts/diagnose_dflash_paired_lengths.py")
    original = [v for k, v in config["source_hashes"].items() if Path(k).name == sampler.name]
    if len(original) != 1 or sha256(sampler) != original[0]:
        raise ValueError("The original single-request sampler changed")
    return [rows[pid] for pid in config["prompt_ids"]]


def gpu_state(gpu):
    result = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    used, total, util = map(int, result.strip().split(","))
    return {"memory_used_mib": used, "memory_total_mib": total, "utilization": util}


def require_idle(gpu):
    if gpu_state(gpu)["memory_used_mib"] > 1024:
        raise RuntimeError("Requested GPU is occupied; no workers started")


_WORKER = None


def initialize_worker(config, gpu, warmup_row, ready, barrier):
    global _WORKER
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    worker_event("model_loading", gpu=gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel
    from scripts.diagnose_dflash_paired_lengths import run_prompt
    if transformers.__version__ != config["transformers"] or torch.__version__ != config["torch"]:
        raise RuntimeError("Pinned model runtime changed")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = config["models"]
    pilot = json.loads(Path(config["pilot_manifest"]).read_text())
    for key in ("target", "draft"):
        expected = pilot["collection_config"]["hashes"][f"/tmp/dflash-prefusion-{key}/config.json"]
        if sha256(Path(models[key]["path"])/"config.json") != expected:
            raise ValueError("Target/draft model identity changed")
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(models["target"]["path"], local_files_only=True)
    if draft.target_layer_ids != pilot["target_layer_ids"]:
        raise ValueError("Target feature layers changed")
    class NoPolicies:
        def predict(self, fused):
            return {}
    options = SimpleNamespace(**config)
    _WORKER = (options, target, draft, tokenizer, NoPolicies())
    warm = copy.copy(options)
    warm.max_new_tokens, warm.states_per_prompt = 32, 1
    warm.reverse_check_states = warm.canonical_check_states = 0
    run_prompt(warm, warmup_row, target, draft, tokenizer, _WORKER[-1], 0)
    torch.cuda.synchronize()
    worker_event("models_ready", resources=process_resources(),
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    ready.put({"pid": os.getpid(), "model_and_warmup_peak_mib": torch.cuda.max_memory_allocated()/2**20})
    barrier.wait(timeout=180)


def worker_prompt(row, count_before):
    from scripts.diagnose_dflash_paired_lengths import run_prompt
    import torch
    options, target, draft, tokenizer, policies = _WORKER
    started = time.monotonic()
    worker_event("prompt_started", task={"prompt_id": int(row["manifest_index"]),
                                         "count_before": count_before}, resources=process_resources())
    try:
        batch, progress = run_prompt(options, row, target, draft, tokenizer, policies, count_before)
        for state, _ in batch:
            state["group"] = "train"
        torch.cuda.synchronize()
    except BaseException as exc:
        worker_event("prompt_exception", error=repr(exc), traceback=traceback.format_exc(),
                     resources=process_resources())
        raise
    execution = {"worker_pid": os.getpid(), "elapsed_s": time.monotonic()-started,
                 "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20}
    worker_event("prompt_finished", **execution, states=len(batch), resources=process_resources())
    return batch, progress, execution


def ready_ping():
    return os.getpid()


def start_pool(workers, gpu, config, warmup_row, diagnostics):
    require_idle(gpu)
    ctx = DiagnosticSpawnContext(diagnostics)
    controller_event(diagnostics, "pool_starting", workers=workers, gpu=gpu)
    ready, barrier = ctx.Queue(), ctx.Barrier(workers)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=initialize_worker,
        initargs=(config, gpu, warmup_row, ready, barrier))
    try:
        checks = [pool.submit(ready_ping) for _ in range(workers)]
        metadata = [ready.get(timeout=180) for _ in range(workers)]
        for future in checks:
            future.result(timeout=180)
    except BaseException as exc:
        controller_event(diagnostics, "pool_startup_failed", tuple((pool._processes or {}).values()),
                         error=repr(exc), traceback=traceback.format_exc())
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    controller_event(diagnostics, "pool_ready", tuple((pool._processes or {}).values()))
    return pool, metadata


def compare_replay(root, pid, batch, progress):
    saved = json.loads((root/f"prompt_{pid}.json").read_text())
    states = [r for r, _ in batch]
    if saved["states"] != states or saved["progress"] != progress:
        raise ValueError(f"Serial/concurrent trajectory or outcome mismatch for prompt {pid}")
    if batch and not np.array_equal(np.stack([x for _, x in batch]), np.load(root/f"prompt_{pid}_fused.npy", allow_pickle=False)):
        raise ValueError(f"Serial/concurrent fused feature mismatch for prompt {pid}")
    return len(states)


def choose_workers(results):
    passed = [r for r in results if r["exact_replay_passed"] and r["peak_used_mib"] <= .85*r["gpu_total_mib"]]
    baseline = next(r for r in passed if r["workers"] == 1)
    passed = [r for r in passed if r["workers"] == 1 or r["eligible_rows_per_s"] >= 1.1*baseline["eligible_rows_per_s"]]
    best = max(r["eligible_rows_per_s"] for r in passed)
    return min(r["workers"] for r in passed if r["eligible_rows_per_s"] >= .95*best)


def benchmark(args):
    config, receipts, _, _, _, audit = inventory(args.source)
    messages = selected_messages(config)
    by_id = {int(r["manifest_index"]): r for r in messages}
    cohort, count_before = [], 0
    for receipt in receipts:
        if receipt["states"] and len(cohort) < args.benchmark_prompts:
            cohort.append((receipt["prompt_id"], count_before))
        count_before += receipt["states"]
    if len(cohort) != args.benchmark_prompts or args.output.exists():
        raise ValueError("Insufficient benchmark cohort or existing destination")
    args.output.mkdir(parents=True)
    results = []
    for workers in (1, 2, 4):
        diagnostics = (args.diagnostics or args.output/"diagnostics")/f"workers_{workers}"
        pool, worker_metadata = start_pool(workers, args.gpu, config, by_id[cohort[0][0]], diagnostics)
        processes = tuple(pool._processes.values())
        samples, stop = [], threading.Event()
        def sample_gpu():
            while not stop.is_set():
                try:
                    samples.append(gpu_state(args.gpu))
                except Exception:
                    pass
                stop.wait(1)
        monitor = threading.Thread(target=sample_gpu, daemon=True)
        monitor.start()
        started = time.monotonic()
        rows = eligible = 0
        error = None
        try:
            futures = [(pid, pool.submit(worker_prompt, by_id[pid], offset)) for pid, offset in cohort]
            for pid, future in futures:
                batch, progress, _ = future.result()
                rows += compare_replay(args.source, pid, batch, progress)
                eligible += sum(r["eligible"] for r, _ in batch)
                print("BENCH_PROMPT", workers, pid, rows, flush=True)
        except Exception as exc:
            error = repr(exc)
            controller_event(diagnostics, "benchmark_failed", processes, error=error,
                             traceback=traceback.format_exc())
        finally:
            elapsed = time.monotonic()-started
            stop.set()
            monitor.join(timeout=3)
            pool.shutdown(wait=True, cancel_futures=True)
            controller_event(diagnostics, "pool_closed", processes, error=error)
        if not samples:
            raise RuntimeError("No GPU memory measurements")
        result = {"workers": workers, "prompts": len(cohort), "rows": rows, "eligible": eligible,
            "elapsed_s": elapsed, "eligible_rows_per_s": eligible/elapsed, "exact_replay_passed": error is None,
            "error": error, "peak_used_mib": max(s["memory_used_mib"] for s in samples),
            "gpu_total_mib": samples[0]["memory_total_mib"],
            "mean_utilization": float(np.mean([s["utilization"] for s in samples])), "workers_ready": worker_metadata}
        results.append(result)
        atomic_json(args.output/"progress.json", {"results": results})
        print("BENCH_RESULT", json.dumps(result), flush=True)
        if error and workers == 1:
            raise RuntimeError("Serial replay failed; do not resume with changed evidence")
    selected = choose_workers(results)
    summary = {"source": str(args.source), "source_config_sha256": sha256(args.source/"config.json"),
        "source_audit": audit, "cohort_prompt_ids": [p for p, _ in cohort], "results": results,
        "selected_workers": selected, "sampler_sha256": sha256("scripts/diagnose_dflash_paired_lengths.py"),
        "driver_sha256": sha256(__file__), "diagnostics_sha256": sha256(DIAGNOSTICS_SOURCE),
        "scope": "same GPU and matched prompts; collection throughput only, not serving speedup",
        "selection": "exact replay, <85% memory, >=10% gain; fewest workers within 5% of fastest eligible setting"}
    atomic_json(args.output/"summary.json", summary)
    print("BENCH_COMPLETE", json.dumps({"selected_workers": selected, "results": results}), flush=True)


def verified_copy(source, destination, digest=None):
    if destination.exists():
        raise ValueError("Refusing to overwrite recovered evidence")
    temporary = destination.with_suffix(destination.suffix+".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != (digest or sha256(source)):
        raise ValueError("Recovery copy mismatch")
    temporary.replace(destination)


def recovery_elapsed(source):
    """Repeated recovery consumes one original time allowance, not a new one."""
    elapsed = []
    for name in ("progress.json", "collection_summary.json"):
        if (source/name).exists():
            value = float(json.loads((source/name).read_text())["elapsed_s"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid prior collection elapsed time")
            elapsed.append(value)
    if not elapsed:
        raise ValueError("Missing prior collection elapsed time")
    return max(elapsed)


def collect(args):
    original, receipts, rows, progress, files, audit = inventory(args.source)
    messages = selected_messages(original)
    bench = json.loads(args.benchmark.read_text())
    if (bench["source_config_sha256"] != sha256(args.source/"config.json") or
            bench["sampler_sha256"] != sha256("scripts/diagnose_dflash_paired_lengths.py") or
            bench["driver_sha256"] != sha256(__file__) or
            bench["diagnostics_sha256"] != sha256(DIAGNOSTICS_SOURCE) or
            bench["selected_workers"] != choose_workers(bench["results"])):
        raise ValueError("Benchmark does not bind to this recovery/source implementation")
    workers = bench["selected_workers"]
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Use distinct new temporary and durable destinations")
    require_idle(args.gpu)
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    def backup(paths):
        for path in paths:
            target = args.backup/path.name
            temporary = target.with_suffix(target.suffix+".tmp")
            shutil.copyfile(path, temporary)
            if sha256(temporary) != sha256(path):
                raise ValueError("Durable backup mismatch")
            temporary.replace(target)
    for name, digest in files.items():
        if name != "config.json":
            verified_copy(args.source/name, args.output/name, digest)
            verified_copy(args.source/name, args.backup/name, digest)
    verified_copy(args.source/"config.json", args.output/"recovery_source_config.json")
    backup([args.output/"recovery_source_config.json"])
    config = copy.deepcopy(original)
    diagnostics = args.diagnostics or args.backup.parent/"diagnostics"/"collection"
    config.update({"output": str(args.output), "backup": str(args.backup), "workers": workers,
        "execution": "independent single-request prompt workers on one GPU, same kernels/shapes per request",
        "recovery": {"source": str(args.source), "audit": audit, "source_files": files,
            "benchmark": str(args.benchmark), "benchmark_sha256": sha256(args.benchmark)},
        "original_commit": original.get("original_commit", original["commit"]),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "efficient_driver_sha256": sha256(__file__), "diagnostics_sha256": sha256(DIAGNOSTICS_SOURCE),
        "diagnostics_directory": str(diagnostics)})
    atomic_json(args.output/"config.json", config)
    backup([args.output/"config.json"])
    atomic_json(args.output/"recovery_audit.json", audit)
    backup([args.output/"recovery_audit.json"])
    prior_elapsed = recovery_elapsed(args.source)
    seconds_left = max(0, original["max_seconds"]-prior_elapsed)
    interrupted = []
    def handle_signal(number, frame):
        interrupted.append({"signal": number, "unix_time": time.time()})
        controller_event(diagnostics, "controller_signal", number=number)
        print("STOP_REQUESTED: finish pending prompts and preserve evidence", number, flush=True)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, handle_signal)
    initial_rows, initial_eligible = len(rows), sum(r["eligible"] for r in rows)
    eligible = initial_eligible
    next_prompt = len(receipts)
    if eligible >= original["training_rows"] or not seconds_left:
        raise ValueError("No missing rows or remaining collection time")
    started = time.monotonic()
    pool, metadata = start_pool(workers, args.gpu, config, messages[next_prompt], diagnostics)
    processes = tuple(pool._processes.values())
    print("RECOVERED_MODELS_READY", json.dumps({"workers": workers, "preserved_eligible": eligible,
        "preserved_prompts": len(receipts), "remaining_time_bound_s": seconds_left, "workers_ready": metadata}), flush=True)
    pending = deque()
    fatal = None
    try:
        while pending or (next_prompt < len(messages) and eligible < original["training_rows"] and
                          not interrupted and time.monotonic()-started < seconds_left):
            while len(pending) < 2*workers and next_prompt < len(messages) and eligible < original["training_rows"] and not interrupted and time.monotonic()-started < seconds_left:
                row = messages[next_prompt]
                # Initial reverse/canonical checks were already completed. Preserve
                # the argument below their state cap, without changing sampler code.
                offset = initial_rows
                pending.append((row, pool.submit(worker_prompt, row, offset)))
                next_prompt += 1
            if not pending:
                break
            row, future = pending.popleft()
            batch, status, execution = future.result()
            pid = int(row["manifest_index"])
            states = [r for r, _ in batch]
            validate_states(states, config["blocks"])
            if any(r["canonical_disagreements"] for r in states):
                raise ValueError("Canonical disagreement")
            if states and (states[0]["cycle"] != 0 or states[0]["generated_before_anchor"] != 0):
                raise ValueError("Missing initial state")
            shard = args.output/f"prompt_{pid}.json"
            if shard.exists():
                raise ValueError("Attempted duplicate prompt write")
            atomic_json(shard, {"progress": status, "states": states})
            paths = [shard]
            if states:
                x = np.stack([f for _, f in batch])
                if x.shape != (len(states), 2560) or x.dtype != np.float16 or not np.isfinite(x).all():
                    raise ValueError("Invalid new fused features")
                feature_path = args.output/f"prompt_{pid}_fused.npy"
                np.save(feature_path, x, allow_pickle=False)
                paths.append(feature_path)
            receipt = {"prompt_id": pid, "source": row["source"], "group": "train", "states": len(states),
                       "files": {p.name: sha256(p) for p in paths}}
            receipt_path = args.output/f"receipt_{pid}.json"
            atomic_json(receipt_path, receipt)
            backup(paths+[receipt_path])
            receipts.append(receipt)
            rows.extend(states)
            progress.append(status)
            eligible += sum(r["eligible"] for r in states)
            elapsed = time.monotonic()-started
            latest = {"prompts_processed": len(receipts), "prompts_planned": len(messages), "states": len(rows),
                "eligible": eligible, "elapsed_s": prior_elapsed+elapsed, "resumed_elapsed_s": elapsed,
                "preserved_eligible": initial_eligible, "new_eligible": eligible-initial_eligible,
                "new_eligible_per_s": (eligible-initial_eligible)/elapsed, "workers": workers,
                "gpu": gpu_state(args.gpu), "latest": status, "execution": execution}
            atomic_json(args.output/"progress.json", latest)
            backup([args.output/"progress.json"])
            controller_event(diagnostics, "prompt_committed", prompt_id=pid, eligible=eligible)
            print(json.dumps(latest), flush=True)
    except BaseException as exc:
        fatal = repr(exc)
        controller_event(diagnostics, "collection_failed", processes, error=fatal,
                         traceback=traceback.format_exc(), eligible=eligible,
                         uncommitted_prompt_ids=[int(r["manifest_index"]) for r, _ in pending],
                         current_prompt_id=int(row["manifest_index"]) if "row" in locals() else None)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        controller_event(diagnostics, "pool_closed", processes, error=fatal, eligible=eligible)
        summary = {"sample_complete": eligible >= original["training_rows"] and fatal is None and not interrupted,
            "prompts": len(receipts), "states": len(rows), "target_training_rows": original["training_rows"],
            "eligible_states": eligible, "elapsed_s": prior_elapsed+time.monotonic()-started,
            "group_states": dict(Counter(r["group"] for r in rows if r["eligible"])),
            "canonical_disagreements": sum(r["canonical_disagreements"] for r in rows),
            "canonical_comparisons": sum(r["canonical_checked"] for r in rows),
            "reverse_checked_states": sum(r["reverse_order_checked"] for r in rows),
            "prompt_progress": progress, "signals": interrupted, "fatal_error": fatal,
            "preserved_eligible": initial_eligible, "workers": workers,
            "diagnostics_directory": str(diagnostics)}
        atomic_json(args.output/"receipts.json", receipts)
        atomic_json(args.output/"collection_summary.json", summary)
        backup([args.output/"receipts.json", args.output/"collection_summary.json"])
        atomic_json(args.output/"COMPLETE.json", {**{k: summary[k] for k in ("sample_complete", "states", "eligible_states")},
            "binding": {f: sha256(args.output/f) for f in ("config.json", "receipts.json", "collection_summary.json")}})
        backup([args.output/"COMPLETE.json"])
    if not summary["sample_complete"]:
        raise RuntimeError("Collection interrupted or time-limited; training is blocked")
    print("COLLECTION_COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "prompt_progress"}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("benchmark", "collect"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backup", type=Path)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--benchmark-prompts", type=int, default=12)
    p.add_argument("--diagnostics", type=Path, help="Durable task-scoped worker logs and resource snapshots")
    args = p.parse_args()
    if args.mode == "collect" and (args.backup is None or args.benchmark is None):
        p.error("Recovery requires --backup and --benchmark")
    if args.mode == "benchmark":
        benchmark(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
