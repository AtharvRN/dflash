"""Bounded, isolated serving-cost study. Run on the designated workstation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import struct
import subprocess
import time

import requests

IMAGE = "sha256:a7317182c71d35712ee4edc86a5d1c313dc969efdf0026d339673299c186ea75"
ROOT = Path("/data/scratch/zekaili/atharv/dflash")
COMPONENTS = {
    "draft_setup_allocation": ("draft_and_verify_prepare", True),
    "draft_forward": ("draft_forward", False),
    "draft_projection_argmax": ("draft_projection_argmax", False),
    "verify_preparation": ("target_verify_prepare", False),
    "target_forward_including_logits": ("target_forward", False),
    "acceptance_target_kv_commit": ("acceptance_and_target_kv_commit", False),
    "draft_kv_upkeep": ("draft_kv_upkeep", False),
    "other_worker": ("decode_cycle", True),
}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[min(len(values)-1, round(.95*(len(values)-1)))],
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": values[0], "max": values[-1]}


def decompose(record, field="stream_elapsed_ms"):
    spans = record["spans"]
    children = defaultdict(float)
    names = defaultdict(list)
    for i, item in enumerate(spans):
        if item["parent"] is not None:
            children[item["parent"]] += item[field]
        names[item["name"]].append(i)
    values = {}
    for label, (name, exclusive) in COMPONENTS.items():
        if not names[name]:
            raise ValueError("Missing component " + name)
        values[label] = sum(spans[i][field] - (children[i] if exclusive else 0.0) for i in names[name])
    if len(names["decode_cycle"]) != 1:
        raise ValueError("Expected one decode-cycle root")
    total = spans[names["decode_cycle"][0]][field]
    if abs(sum(values.values()) - total) > 1e-5:
        raise ValueError("Non-disjoint component accounting")
    return values, total


def summarize_profiles(paths):
    groups = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["phase"] == "decode" and row["label"].startswith("measured_"):
                groups[row["label"]].append(row)
    out = {}
    for label, records in groups.items():
        requested = int(label.split("_c")[1].split("_")[0])
        strata = {"all_decode": records,
                  "full_batch_decode": [r for r in records if r["batch_size"] == requested]}
        result = {}
        for stratum, rows in strata.items():
            if not rows:
                continue
            stream, host, totals, host_totals = defaultdict(list), defaultdict(list), [], []
            for row in rows:
                values, total = decompose(row)
                hosts, host_total = decompose(row, "host_call_ms")
                totals.append(total)
                host_totals.append(host_total)
                for k, v in values.items():
                    stream[k].append(v)
                    host[k].append(hosts[k])
            result[stratum] = {
                "cycles": len(rows), "batch_size": distribution([r["batch_size"] for r in rows]),
                "prefix_length": distribution([x for r in rows for x in (r["prefix_lens"] or [])]),
                "stream_cycle_ms": distribution(totals), "host_cycle_ms": distribution(host_totals),
                "stream_components_ms": {k: distribution(v) for k, v in stream.items()},
                "host_components_ms": {k: distribution(v) for k, v in host.items()},
                "accepted_drafts": sum(r["accepted_drafts"] for r in rows),
                "request_cycles": sum(r["batch_size"] for r in rows),
                "target_graph_fraction": statistics.mean(r["target_graph"] for r in rows),
                "draft_graph_observed_fraction": statistics.mean(r.get("draft_forward_graph", False) for r in rows),
            }
        out[label] = result
    return out


def load_workload(cache):
    config = json.loads((cache / "config.json").read_text())
    prompts = []
    for pid in config["prompt_ids"]:
        data = json.loads((cache / f"prompt_{pid}.json").read_text())
        rows = [s for s in data["states"] if s["cycle"] == 0]
        if not rows:
            continue
        state = rows[0]
        if state["group"] != "assessment":
            continue
        ids = state["prefix_token_ids"]
        if len(ids) != state["prefix_length"] + 1:
            raise ValueError("Prefix length mismatch")
        digest = hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
        if digest != state["prefix_sha256"]:
            raise ValueError("Saved prefix/anchor hash mismatch")
        # Saved state includes the known anchor. Prefill only the committed
        # prompt and let the serving target generate that anchor itself.
        prompts.append({"prompt_id": pid, "input_ids": ids[:-1], "saved_anchor": ids[-1],
                        "prefix_with_anchor_sha256": state["prefix_sha256"]})
    random.Random(930).shuffle(prompts)
    if len(prompts) < 64:
        raise ValueError("Need at least 64 audited development prompts")
    return prompts


def send_one(base, item, max_tokens):
    start = time.perf_counter()
    response = requests.post(base + "/generate", json={
        "input_ids": item["input_ids"],
        "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": max_tokens},
        "return_logprob": False,
    }, timeout=180)
    response.raise_for_status()
    data = response.json()
    return {"prompt_id": item["prompt_id"], "latency_s": time.perf_counter()-start,
            "response": data}


def run_requests(base, prompts, concurrency, count, max_tokens):
    items = [prompts[i % len(prompts)] for i in range(count)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda item: send_one(base, item, max_tokens), items))
    elapsed = time.perf_counter()-start
    tokens = sum(row["response"].get("meta_info", {}).get("completion_tokens", 0) for row in rows)
    if tokens <= 0:
        raise RuntimeError("No completion-token accounting returned")
    return {"wall_s": elapsed, "output_tokens": tokens, "output_tokens_per_s": tokens/elapsed,
            "latency_s": distribution([r["latency_s"] for r in rows]),
            "requests": rows, "concurrency": concurrency, "count": count, "max_new_tokens": max_tokens}


def check_gpu(index):
    out = command(["nvidia-smi", f"--id={index}",
                   "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
                   "--format=csv,noheader,nounits"])
    fields = [x.strip() for x in out.strip().split(",")]
    apps = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    if int(fields[3]) >= 1024 or any(fields[1] in line for line in apps.splitlines()):
        raise RuntimeError("Selected GPU is occupied; no process will be interrupted")
    return dict(zip(["index", "uuid", "name", "memory_used_mib", "memory_total_mib", "utilization", "driver"], fields))


def gpu_telemetry(index):
    fields = "timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu"
    return command(["nvidia-smi", f"--id={index}", "--query-gpu="+fields, "--format=csv"])


def wait_ready(base, container, log_path, timeout=720):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(base + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", container],
                               text=True, capture_output=True)
        if state.returncode == 0 and state.stdout.strip() == "false":
            raise RuntimeError("Server exited; inspect " + str(log_path))
        time.sleep(2)
    raise TimeoutError("Server startup timeout; inspect " + str(log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--blocks", nargs="+", type=int, default=[16, 12, 8])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 32, 64])
    parser.add_argument("--modes", nargs="+", choices=["clean", "events", "trace"], default=["clean", "events"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seconds", type=int, default=5400)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--target-only", action="store_true",
                        help="Non-speculative target control; clean runs only")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior evidence: output must be new")
    if args.target_only:
        if args.modes != ["clean"]:
            raise ValueError("Target-only control uses clean throughput mode")
        args.blocks = [1]
    elif any(b < 2 or b > 16 for b in args.blocks):
        raise ValueError("This experiment uses the original B16-trained drafter, B2--16")
    lock = (ROOT / f"gpu_{args.gpu}_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    args.output.mkdir(parents=True)
    shared_cache = args.output / "runtime_cache"
    shared_cache.mkdir()
    repo = Path(__file__).resolve().parents[1]
    prompts = load_workload(ROOT / "runs/policy_granularity_20260927/cache")
    models = json.loads((ROOT / "models.json").read_text())
    atomic_json(args.output / "workload.json", prompts)
    atomic_json(args.output / "config.json", {"args": {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "gpu": gpu, "image": IMAGE, "models": models, "code_commit": command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
        "workload_sha256": hashlib.sha256((args.output/"workload.json").read_bytes()).hexdigest(),
        "limitations": ["fixed-width DFlash v1, not packed variable-B serving", "previously inspected development prompts",
                        "clean HTTP throughput includes prefill and draining tails", "event spans include GPU idle gaps"]})
    started = time.monotonic()
    results = []
    container = None
    try:
        for mode in args.modes:
            for block in args.blocks:
                if time.monotonic()-started > args.max_seconds:
                    raise TimeoutError("Bounded study deadline reached")
                stage = args.output / f"{mode}_b{block}"
                stage.mkdir()
                cache = shared_cache
                log_path = stage / "server.log"
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                base = f"http://127.0.0.1:{port}"
                container = "atharv-dflash-profile-" + hashlib.sha256(str(stage).encode()).hexdigest()[:12]
                launch = ["docker", "run", "--name", container, "--network", "host", "--gpus", "device="+gpu["uuid"],
                          "--workdir", str(stage),
                          # This public image has editable Python packages under
                          # /root; retain its image user. Host code/data are RO,
                          # and only this stage's output/cache directory is RW.
                          "--shm-size", "8g", "--cpus", "12", "--cap-drop", "ALL",
                          "--cap-add", "DAC_OVERRIDE", "--security-opt", "no-new-privileges",
                          "-v", f"{repo}:{repo}:ro", "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{stage}:{stage}:rw",
                          "-v", f"{cache}:{cache}:rw",
                          "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4", "-e", "TOKENIZERS_PARALLELISM=false",
                          "-e", "LOGNAME=" + getpass.getuser(),
                          "-e", "NVIDIA_TF32_OVERRIDE=0",
                          "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                          "-e", f"HF_HOME={ROOT}/hf", "-e", f"XDG_CACHE_HOME={cache}",
                          "-e", f"TRITON_CACHE_DIR={cache}/triton", "-e", f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor",
                          "-e", f"CUDA_CACHE_PATH={cache}/cuda", "-e", f"FLASHINFER_WORKSPACE_BASE={cache}/flashinfer",
                          "-e", f"SGLANG_CACHE_DIR={cache}/sglang", "-e", f"SGLANG_DG_CACHE_DIR={cache}/deep_gemm",
                          "-e", f"TORCH_HOME={cache}/torch", "-e", f"TORCH_EXTENSIONS_DIR={cache}/extensions"]
                if mode != "clean":
                    launch += ["-e", f"PYTHONPATH={repo}/scripts/sglang_profile_hook",
                               "-e", "DFLASH_COMPONENT_PROFILE=" + mode, "-e", f"DFLASH_PROFILE_DIR={stage}"]
                launch += ["--entrypoint", "python", IMAGE, "-m", "sglang.launch_server",
                           "--model-path", models["target"]["path"], "--speculative-algorithm", "DFLASH",
                           "--speculative-draft-model-path", models["draft"]["path"],
                           "--speculative-dflash-block-size", str(block), "--speculative-num-draft-tokens", str(block),
                           "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1", "--dtype", "bfloat16",
                           "--attention-backend", "triton", "--speculative-draft-attention-backend", "triton",
                           "--mem-fraction-static", "0.65", "--max-running-requests", str(max(args.concurrency)),
                           "--max-total-tokens", "131072", "--context-length", "4096",
                           "--cuda-graph-max-bs", str(max(args.concurrency)), "--disable-radix-cache",
                           "--disable-piecewise-cuda-graph"]
                if args.target_only:
                    for flag in ("--speculative-algorithm", "--speculative-draft-model-path",
                                 "--speculative-dflash-block-size", "--speculative-num-draft-tokens",
                                 "--speculative-draft-attention-backend"):
                        i = launch.index(flag)
                        del launch[i:i+2]
                atomic_json(stage / "launch.json", launch)
                atomic_json(stage / "control.json", {"label": "startup"})
                with log_path.open("w") as log:
                    proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        wait_ready(base, container, log_path)
                        try:
                            atomic_json(stage / "server_info.json", requests.get(base+"/get_server_info", timeout=10).json())
                        except Exception as error:
                            atomic_json(stage / "server_info_error.json", {"error": repr(error)})
                        for concurrency in args.concurrency:
                            if time.monotonic()-started > args.max_seconds:
                                raise TimeoutError("Bounded study deadline reached")
                            atomic_json(stage/"control.json", {"label": f"warmup_c{concurrency}"})
                            run_requests(base, prompts, concurrency, concurrency, 32 if args.smoke else 96)
                            reps = 1 if args.smoke or mode != "clean" else args.repeats
                            for repeat in range(reps):
                                label = f"measured_b{block}_c{concurrency}_{mode}_r{repeat}"
                                atomic_json(stage/"control.json", {"label": label})
                                if mode == "trace":
                                    response = requests.post(base+"/start_profile", json={
                                        "output_dir": str(stage / "trace"), "num_steps": 20,
                                        "activities": ["CPU", "GPU"], "with_stack": False,
                                        "record_shapes": False}, timeout=30)
                                    response.raise_for_status()
                                count = max(concurrency, 4) if args.smoke else (8 if concurrency == 1 else 6*concurrency)
                                cap = 32 if args.smoke else args.max_new_tokens
                                before = gpu_telemetry(args.gpu)
                                result = run_requests(base, prompts, concurrency, count, cap)
                                result["gpu_before"], result["gpu_after"] = before, gpu_telemetry(args.gpu)
                                atomic_json(stage / f"{label}.json", result)
                                results.append({"label": label, **{k:v for k,v in result.items() if k != "requests"}})
                                atomic_json(args.output / "progress.json", {"results": results, "elapsed_s": time.monotonic()-started})
                                print(json.dumps({"label": label, "tokens_per_s": result["output_tokens_per_s"],
                                                  "wall_s": result["wall_s"]}), flush=True)
                                if mode == "trace":
                                    requests.post(base+"/stop_profile", timeout=180)
                            atomic_json(stage/"control.json", {"label": "flush"})
                            run_requests(base, prompts, 1, 1, 2)
                        if mode != "clean":
                            atomic_json(stage / "components.json", summarize_profiles(stage.glob("cycles_*.jsonl")))
                    finally:
                        subprocess.run(["docker", "stop", "--time", "20", container], capture_output=True, timeout=40)
                        proc.wait(timeout=30)
                        # Keep the stopped named container for startup-failure inspection.
                        container = None
        atomic_json(args.output / "summary.json", {"results": results,
            "components": summarize_profiles(args.output.glob("*/cycles_*.jsonl")), "elapsed_s": time.monotonic()-started})
        atomic_json(args.output / "COMPLETE.json", {"success": True, "elapsed_s": time.monotonic()-started})
    except BaseException as error:
        atomic_json(args.output / "FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise
    finally:
        if container:
            subprocess.run(["docker", "stop", "--time", "20", container], capture_output=True, timeout=40)


if __name__ == "__main__":
    main()
