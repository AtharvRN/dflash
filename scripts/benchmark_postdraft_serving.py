"""Actual HTTP closed-loop SGLang throughput, never replay-derived tokens/s.

Restores the verified runtime and applies the post-draft extension. Runs complete
requests through the scheduler, with disjoint warmup and fixed validation prompts.
Only subprocess groups created here are terminated. Results persist per phase.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import requests
from prepare_postdraft_serving import prepare

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_cycle_profile(source, destination):
    """Keep per-cycle control/trace I/O local; persist snapshots after each phase."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            shutil.copy2(path, destination / path.name)


def workload(args, model):
    if args.workload_file:
        saved = json.loads(args.workload_file.read_text())
        if len(saved["warmup"]) < args.warmup or len(saved["measurement"]) < args.requests:
            raise ValueError("Frozen workload is smaller than requested experiment")
        if len(saved["warmup"]) == args.warmup and len(saved["measurement"]) == args.requests:
            shutil.copy2(args.workload_file, args.output / "workload.json")
        else:
            saved = saved | {"parent_workload_sha256": sha(args.workload_file),
                            "warmup": saved["warmup"][:args.warmup],
                            "measurement": saved["measurement"][:args.requests]}
            save(args.output / "workload.json", saved)
        return saved["warmup"], saved["measurement"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = set(json.loads(args.split.read_text())["val_prompt_ids"])
    selected = []
    with args.manifest.open() as f:
        for line in f:
            item = json.loads(line)
            if str(item["manifest_index"]) in ids:
                selected.append(item)
    random.Random(934).shuffle(selected)
    rows = []
    content_seen = set()
    for item in selected:
        messages = item["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("Unexpected original conversation schema")
        tokens = tokenizer.apply_chat_template(messages[:1], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise TypeError("Expected unbatched token IDs from chat template")
        if not 1 <= len(tokens) <= 2048:
            continue
        content_key = tuple(tokens)
        if content_key in content_seen:
            continue
        content_seen.add(content_key)
        rows.append({"prompt_id": str(item["manifest_index"]), "source": item["source"],
                     "input_ids": tokens})
        if len(rows) == args.requests + args.warmup:
            break
    if len(rows) != args.requests + args.warmup:
        raise ValueError("Insufficient eligible unique held-out prompts")
    save(args.output / "workload.json", {"selection": "seed934 shuffled canonical validation IDs; unique original user prompts, <=2048 tokens; thinking off",
         "manifest_sha256": sha(args.manifest), "split_sha256": sha(args.split),
         "warmup": rows[:args.warmup], "measurement": rows[args.warmup:]})
    return rows[:args.warmup], rows[args.warmup:]


def request_one(base, row, max_tokens, phase=None):
    started = time.perf_counter()
    payload = {"input_ids": row["input_ids"],
        "sampling_params": {"temperature": 0, "top_k": 1, "top_p": 1,
                            "max_new_tokens": max_tokens}}
    if phase is not None:
        payload["rid"] = f"pdv:{phase}:{row['prompt_id']}"
    response = requests.post(base + "/generate", json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or int(result.get("meta_info", {}).get("completion_tokens", 0)) <= 0:
        raise RuntimeError(f"Invalid generation response: {result}")
    return {"prompt_id": row["prompt_id"], "latency_s": time.perf_counter() - started,
            "response": result}


def run_requests(base, rows, concurrency, max_tokens, phase=None):
    started_unix = time.time()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda row: request_one(base, row, max_tokens, phase), rows))
    elapsed = time.perf_counter() - started
    tokens = sum(r["response"]["meta_info"]["completion_tokens"] for r in results)
    times = sorted(r["latency_s"] for r in results)
    return {"started_unix_s": started_unix, "finished_unix_s": time.time(),
            "wall_time_s": elapsed, "output_tokens": tokens, "throughput_tok_s": tokens / elapsed,
            "requests": len(results), "concurrency": concurrency, "max_new_tokens": max_tokens,
            "request_latency_mean_s": statistics.mean(times), "request_latency_p95_s": times[int(.95*(len(times)-1))],
            "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "configs/rejected_trace_nrp_models.json")
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/dflashv2_data/manifests/qwen3_4b_instruct_100k_messages.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("/workspace/dflashv2_data/splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719/val_prompt_ids.json"))
    parser.add_argument("--cases", nargs="+", choices=["fixed16", "fixed12", "fixed8", "fixed4", "raw", "raw_no_trim", "target_ar", "history8", "history16", "history32", "history64"], default=["fixed16", "raw", "fixed12", "fixed8"])
    parser.add_argument('--history-artifacts', type=Path, help='Frozen_cC.json and cost_profile.json directory')
    parser.add_argument('--history-table', type=Path)
    parser.add_argument('--history-acceptance-check', action='store_true', help='Separate diagnostic only; not clean throughput')
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=-0.8720796704292296)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--port", type=int, default=22508)
    parser.add_argument("--verify-length-audit", action="store_true")
    parser.add_argument("--cycle-cost-profile", action="store_true",
                        help="Separate instrumented fixed-B worker-cycle cost run; NOT clean throughput")
    parser.add_argument('--runtime-site-packages', nargs='+', default=[],
                        help='Explicit ordered dependency paths for staged container environments')
    parser.add_argument("--workload-file", type=Path)
    parser.add_argument("--mem-fraction-static", type=float, default=.50)
    parser.add_argument("--max-total-tokens", type=int, default=262144)
    args = parser.parse_args()
    if any(not Path(path).is_dir() for path in args.runtime_site_packages):
        raise ValueError('Missing runtime dependency directory')
    if args.cycle_cost_profile and (args.verify_length_audit or any(not c.startswith('fixed') for c in args.cases)):
        raise ValueError('Cycle costs require fixed-B cases without another instrumentation mode')
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    if any(c.startswith('history') for c in args.cases):
        if args.history_artifacts is None or args.history_table is None:
            raise ValueError('History serving needs frozen artifacts and table')
        if any(int(c[7:]) not in args.concurrencies for c in args.cases if c.startswith('history')):
            raise ValueError('Each history case needs its calibrated concurrency in --concurrencies')
        from prepare_history_serving import prepare as prepare_history
        extension = prepare_history(args.scratch / "source")
        table_identity = json.loads(args.history_table.read_text())['provenance']['model_identity']
        for role in ('target', 'draft'):
            if any(models[role][k] != table_identity[role][k] for k in ('repo', 'revision')):
                raise ValueError('Serving models differ from frozen history table')
        artifact_dir = args.output/'frozen_history'
        artifact_dir.mkdir()
        shutil.copy2(args.history_table, artifact_dir/'table.json')
        shutil.copy2(args.history_artifacts/'cost_profile.json', artifact_dir/'cost_profile.json')
        for c in args.cases:
            if c.startswith('history'):
                shutil.copy2(args.history_artifacts/f'frozen_c{c[7:]}.json', artifact_dir/f'frozen_c{c[7:]}.json')
    else:
        extension = prepare(args.scratch / "source")
    warmup, measured = workload(args, models["target"]["path"])
    save(args.output / "config.json", {"args": vars(args) | {k: str(v) for k,v in vars(args).items() if isinstance(v, Path)},
         "models": models, "extension": extension, "workload_sha256": sha(args.output / "workload.json"),
         "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "scope": "Actual HTTP serving: prompt prefill + all autoregressive speculative cycles + scheduler + allocator + detokenization + client wall time. Input tokenization performed before timer. Greedy, natural EOS, no cache reuse, max512 unless overridden. Not decode-only throughput.",
         "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv"], text=True),
         "environment": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()})
    runtime = args.scratch / "cache"
    runtime.mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith(("DFLASH_", "SGLANG_DFLASH_"))}
    runtime_path = ':'.join([str(args.scratch / 'source/python'), *args.runtime_site_packages])
    env.update(PYTHONPATH=runtime_path, PYTHONDONTWRITEBYTECODE="1",
               SGLANG_ENABLE_SPEC_V2="1", SGLANG_ENABLE_DFLASH_SPEC_V2="1", SGLANG_DFLASH_TIMING="0",
               NVIDIA_TF32_OVERRIDE="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", XDG_CACHE_HOME=str(runtime))
    base = f"http://127.0.0.1:{args.port}"
    process = None
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    summaries = []
    for case in args.cases:
        case_dir = args.output / case
        case_dir.mkdir()
        for key in ('SGLANG_DFLASH_HISTORY_TABLE', 'SGLANG_DFLASH_HISTORY_PROFILE', 'SGLANG_DFLASH_HISTORY_FROZEN'):
            env.pop(key, None)
        if case.startswith('history'):
            env.update(SGLANG_DFLASH_HISTORY_TABLE=str(artifact_dir/'table.json'),
                       SGLANG_DFLASH_HISTORY_PROFILE=str(artifact_dir/'cost_profile.json'),
                       SGLANG_DFLASH_HISTORY_FROZEN=str(artifact_dir/f'frozen_c{case[7:]}.json'))
            if args.history_acceptance_check:
                env['SGLANG_DFLASH_HISTORY_CHECK'] = '1'
        else:
            env.pop('SGLANG_DFLASH_HISTORY_CHECK', None)
        if args.cycle_cost_profile:
            profile_dir = args.scratch / f'{case}_cycle_profile'
            profile_dir.mkdir()
            env.update(DFLASH_V2_PROFILE='1', DFLASH_V2_PROFILE_DIR=str(profile_dir),
                       PYTHONPATH=f'{ROOT}/scripts/sglang_v2_profile_hook:{runtime_path}')
        if args.verify_length_audit:
            env["SGLANG_DFLASH_VERIFY_AUDIT_PATH"] = str(case_dir / "verify_lengths.jsonl")
        block = int(case[5:]) if case.startswith("fixed") else 16
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", models["target"]["path"],
                   "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", "1", "--dtype", "bfloat16",
                   "--random-seed", "934", "--attention-backend", "flashinfer", "--mem-fraction-static", str(args.mem_fraction_static),
                   "--max-running-requests", str(max(args.concurrencies)), "--max-total-tokens", str(args.max_total_tokens),
                   "--context-length", "4096", "--disable-radix-cache", "--disable-piecewise-cuda-graph"]
        if case != "target_ar":
            command += ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", models["draft"]["path"],
                        "--speculative-dflash-block-size", str(block), "--speculative-num-draft-tokens", str(block),
                        "--speculative-draft-attention-backend", "flashinfer"]
        if case.startswith("raw"):
            command += ["--speculative-dflash-postdraft-logprob-threshold", str(args.threshold if case == "raw" else -1000.)]
        if args.eager:
            command += ["--disable-cuda-graph"]
        else:
            sizes = [n for n in [1, 2, 4, 8, 16, 32, 64, 128] if n <= max(args.concurrencies)]
            command += ["--cuda-graph-bs", *map(str, sizes)]
        save(case_dir / "launch.json", command)
        # Server logging must not put network-PVC writes on the timed hot path.
        local_server_log = args.scratch / f'{case}_server.log'
        try:
            with local_server_log.open("x") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{case} server exited {process.returncode}")
                    try:
                        ready = requests.get(base + "/health_generate", timeout=3)
                        if ready.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() - started > 900:
                        raise TimeoutError("Server startup exceeded 900s")
                    time.sleep(2)
                save(case_dir / "server_info.json", requests.get(base + "/get_server_info", timeout=30).json())
                shutil.copy2(local_server_log, case_dir/'server.log')
                case_concurrencies = [int(case[7:])] if case.startswith('history') else args.concurrencies
                for concurrency in case_concurrencies:
                    print(f"WARMUP {case} C{concurrency}", flush=True)
                    if args.cycle_cost_profile:
                        save(profile_dir / 'control.json', {'label': f'warmup_b{block}_c{concurrency}'})
                    run_requests(base, warmup, concurrency, min(64, args.max_new_tokens),
                                 f"{case}_c{concurrency}_warmup" if args.verify_length_audit else None)
                    for repeat in range(args.repeats):
                        if args.cycle_cost_profile:
                            save(profile_dir / 'control.json', {'label': f'measured_b{block}_c{concurrency}_r{repeat}'})
                        result = run_requests(base, measured, concurrency, args.max_new_tokens,
                                             f"{case}_c{concurrency}_r{repeat}" if args.verify_length_audit else None)
                        result["measurement_kind"] = "instrumented_length_audit" if args.verify_length_audit else "throughput"
                        if args.cycle_cost_profile:
                            result['measurement_kind'] = 'instrumented_worker_cycle_cost; not clean throughput'
                        if args.history_acceptance_check:
                            result['measurement_kind'] = 'acceptance correctness diagnostic; not clean throughput'
                        result.update(case=case, repeat=repeat, workload_sha256=sha(args.output / "workload.json"))
                        save(case_dir / f"c{concurrency}_r{repeat}.json", result)
                        shutil.copy2(local_server_log, case_dir/'server.log')
                        if args.cycle_cost_profile:
                            backup_cycle_profile(profile_dir, case_dir / 'cycle_profile')
                        brief = {k:v for k,v in result.items() if k != "results"}
                        summaries.append(brief)
                        save(args.output / "progress.json", summaries)
                        print(json.dumps(brief), flush=True)
        except BaseException as error:
            save(args.output / "FAILED.json", {"case": case, "error": repr(error)})
            raise
        finally:
            if process:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process = None
            if local_server_log.exists():
                shutil.copy2(local_server_log, case_dir/'server.log')
            if args.cycle_cost_profile:
                backup_cycle_profile(profile_dir, case_dir / 'cycle_profile')
    save(args.output / "COMPLETE.json", summaries)


if __name__ == "__main__":
    main()
