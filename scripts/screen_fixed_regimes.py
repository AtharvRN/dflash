"""Bounded fixed-width dense/MoE cost screen, not an adaptive-policy benchmark."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import requests

try:
    from .block_headroom import select_training_prompts
    from .profile_sglang_latency import (IMAGE, ROOT, atomic_json, check_gpu, choose_http_port,
                                         command, distribution, gpu_telemetry, wait_ready)
    from .recover_legacy_ragged import restore
except ImportError:
    from block_headroom import select_training_prompts
    from profile_sglang_latency import (IMAGE, ROOT, atomic_json, check_gpu, choose_http_port,
                                        command, distribution, gpu_telemetry, wait_ready)
    from recover_legacy_ragged import restore

PHASES = ("runtime_block_policy", "draft_block_setup", "draft_model_forward", "draft_token_projection",
          "verify_preparation", "target_verify_forward", "acceptance_bonus", "post_verify_kv_materialize")
SPLIT = "qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def components(row, field="stream_elapsed_ms"):
    values = {p: 0.0 for p in PHASES}
    roots = []
    for item in row["spans"]:
        if item["name"] == "decode_cycle":
            roots.append(item[field])
        elif item["name"] in values:
            values[item["name"]] += item[field]
        else:
            raise ValueError("Unexpected/nested phase in FIXED screen: "+item["name"])
    if len(roots) != 1 or not all(any(s["name"] == p for s in row["spans"]) for p in PHASES):
        raise ValueError("Incomplete fixed-cycle phase accounting")
    values["other_worker"] = roots[0]-sum(values.values())
    if values["other_worker"] < -0.02:
        raise ValueError("Overlapping phase accounting")
    return values, roots[0]


def summarize_events(stage):
    groups = defaultdict(list)
    for path in stage.glob("cycles_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["label"].startswith("measured_"):
                groups[row["label"]].append(row)
    summary = {}
    for label, rows in groups.items():
        concurrency = int(label.split("_c")[1].split("_")[0])
        summary[label] = {}
        for name, subset in (("all_decode", rows), ("full_batch_decode", [r for r in rows if r["batch_size"] == concurrency])):
            if not subset:
                summary[label][name] = {"cycles": 0, "usable": False}
                continue
            by_field = {}
            for field in ("stream_elapsed_ms", "host_call_ms"):
                phase_values = defaultdict(list)
                totals = []
                for row in subset:
                    values, total = components(row, field)
                    totals.append(total)
                    for k, v in values.items():
                        phase_values[k].append(v)
                by_field[field] = {"cycle": distribution(totals),
                                   "components": {k: distribution(v) for k, v in phase_values.items()}}
            accepted = sum(r["native"]["total_accepted_drafts"] for r in subset)
            request_cycles = sum(r["batch_size"] for r in subset)
            budget = sum(r["batch_size"]*(r["block_size"]-1) for r in subset)
            summary[label][name] = {"cycles": len(subset), "usable": len(subset) >= 10,
                "batch_size": distribution([r["batch_size"] for r in subset]),
                "prefix_length": distribution([v for r in subset for v in r["prefix_lens"]]),
                "accepted_drafts": accepted, "request_cycles": request_cycles,
                "aggregate_ratio": accepted/budget, "mean_accepted": accepted/request_cycles,
                "target_graph_fraction": sum(r["target_graph"] for r in subset)/len(subset),
                "timing": by_field}
    return summary


def make_workload(registries):
    """Same 128 long TRAIN messages, model-specific chat wrappers/tokenization.

    Body token truncation produces exactly 512/1024 tokens including wrappers.
    This is a controlled-length stress workload, NOT a natural task accuracy set.
    """
    from transformers import AutoTokenizer
    data = ROOT/"data/dflashv2_data"
    manifest = data/"manifests/qwen3_4b_instruct_100k_messages.jsonl"
    split = data/"splits"/SPLIT
    candidates = select_training_prompts(manifest, split, 1000, 928)
    tokenizers = {name: AutoTokenizer.from_pretrained(m["target"]["path"], local_files_only=True,
                                                     trust_remote_code=False) for name, m in registries.items()}
    wrappers = {}
    sentinel = "<<DFLASH_SCREEN_USER_CONTENT>>"
    for name, tokenizer in tokenizers.items():
        template = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        if template.count(sentinel) != 1:
            raise ValueError("Unexpected chat template")
        prefix, suffix = template.split(sentinel)
        wrappers[name] = [tokenizer.encode(s, add_special_tokens=False) for s in (prefix, suffix)]
    rows = []
    for candidate in candidates:
        if len(candidate["messages"]) != 1 or candidate["messages"][0]["role"] != "user":
            raise ValueError("Unexpected prompt manifest conversation")
        text = candidate["messages"][0]["content"]
        bodies = {name: tok.encode(text, add_special_tokens=False) for name, tok in tokenizers.items()}
        if any(len(body)+sum(map(len, wrappers[name])) < 1024 for name, body in bodies.items()):
            continue
        ids = {}
        for name, body in bodies.items():
            prefix, suffix = wrappers[name]
            ids[name] = {str(n): prefix+body[:n-len(prefix)-len(suffix)]+suffix for n in (512, 1024)}
            if any(len(v) != int(n) for n, v in ids[name].items()):
                raise AssertionError("Controlled token length failure")
        rows.append({"prompt_id": candidate["manifest_index"], "source": candidate["source"],
                     "content_sha256": candidate["content_sha256"], "input_ids": ids})
        if len(rows) == 128:
            break
    if len(rows) != 128:
        raise ValueError("Insufficient long, cross-split-deduplicated training prompts")
    return {"kind": "controlled-length chat stress; truncated user bodies; ignore EOS; not task accuracy",
            "selection_seed": 928, "rows": rows, "source_counts": dict(Counter(r["source"] for r in rows)),
            "manifest_sha256": sha(manifest), "split_sha256": {p.name: sha(p) for p in split.glob("*_prompt_ids.json")},
            "model_revisions": {k: v["target"]["revision"] for k, v in registries.items()}, "wrappers": wrappers}


def wave(base, items, model, length, cap):
    """One API batch avoids client-thread launch skew; server still schedules it."""
    start = time.perf_counter()
    response = requests.post(base+"/generate", json={
        "input_ids": [row["input_ids"][model][str(length)] for row in items],
        "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": cap, "ignore_eos": True},
        "return_logprob": False}, timeout=900)
    response.raise_for_status()
    elapsed = time.perf_counter()-start
    result = response.json()
    if not isinstance(result, list) or len(result) != len(items):
        raise ValueError("Unexpected batched response")
    counts = [r["meta_info"]["completion_tokens"] for r in result]
    if counts != [cap]*len(items):
        raise ValueError("Fixed-cap stress did not generate the requested tokens")
    return {"wall_s": elapsed, "output_tokens": sum(counts), "tokens_per_s": sum(counts)/elapsed,
            "concurrency_requested": len(items), "prompt_tokens": length, "max_new_tokens": cap,
            "prompt_ids": [r["prompt_id"] for r in items], "responses": result}


def serve(args, model, registry, block, mode, workload, gpu, source, cache, smoke=False):
    repo = Path(__file__).resolve().parents[1]
    stage = args.output/f"{model}_b{block}_{'smoke' if smoke else mode}"
    stage.mkdir()
    max_c = 4 if smoke else 128
    port = choose_http_port()
    base = f"http://127.0.0.1:{port}"
    container = "atharv-fixed-screen-"+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
    launch = ["docker", "run", "--name", container, "--network", "host", "--gpus", "device="+gpu["uuid"],
        "--cpus", "12", "--shm-size", "8g", "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
        "--security-opt", "no-new-privileges", "--workdir", str(stage),
        "-v", f"{repo}:{repo}:ro", "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{stage}:{stage}:rw",
        "-v", f"{cache}:{cache}:rw"]
    env = {"PYTHONPATH": f"{source}/python", "PYTHONDONTWRITEBYTECODE": "1", "LOGNAME": getpass.getuser(),
        "NVIDIA_TF32_OVERRIDE": "0", "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT/"hf"),
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "XDG_CACHE_HOME": str(cache), "TRITON_CACHE_DIR": str(cache/"triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache/"inductor"), "CUDA_CACHE_PATH": str(cache/"cuda"),
        "FLASHINFER_WORKSPACE_BASE": str(cache/"flashinfer"), "SGLANG_CACHE_DIR": str(cache/"sglang"),
        "SGLANG_DG_CACHE_DIR": str(cache/"deep_gemm"), "TORCH_HOME": str(cache/"torch"),
        "TORCH_EXTENSIONS_DIR": str(cache/"extensions"), "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_DFLASH_SPEC_V2": "1", "SGLANG_DFLASH_TIMING": "0"}
    if mode == "events" and not smoke:
        env.update(PYTHONPATH=f"{repo}/scripts/sglang_v2_profile_hook:{source}/python",
                   DFLASH_V2_PROFILE="1", DFLASH_V2_PROFILE_DIR=str(stage))
    for key, value in env.items():
        launch += ["-e", f"{key}={value}"]
    launch += ["--entrypoint", "python", IMAGE, "-m", "sglang.launch_server",
        "--model-path", registry["target"]["path"], "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", registry["draft"]["path"],
        "--speculative-dflash-block-size", str(block), "--speculative-num-draft-tokens", str(block),
        "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1", "--dtype", "bfloat16",
        "--random-seed", "928", "--attention-backend", "flashinfer", "--speculative-draft-attention-backend", "flashinfer",
        "--mem-fraction-static", "0.88", "--max-running-requests", str(max_c),
        "--max-total-tokens", str(8192 if smoke else 196608), "--context-length", "1536",
        "--chunked-prefill-size", "4096", "--page-size", "1", "--cuda-graph-max-bs", str(max_c),
        "--cuda-graph-bs", *[str(n) for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= max_c],
        "--disable-radix-cache", "--disable-piecewise-cuda-graph"]
    if model == "moe":
        # SM120 BF16 baseline, not SM100-only TRT-LLM or quantized weights.
        launch += ["--moe-runner-backend", "triton"]
    atomic_json(stage/"launch.json", launch)
    atomic_json(stage/"control.json", {"label": "startup"})
    (stage/"gpu_before.csv").write_text(gpu_telemetry(args.gpu))
    proc = None
    try:
        with (stage/"server.log").open("x") as log:
            proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(base, container, stage/"server.log", timeout=1200)
            atomic_json(stage/"server_info.json", requests.get(base+"/get_server_info", timeout=15).json())
            if smoke:
                for n in (1, 4):
                    result = wave(base, workload["rows"][:n], model, 512, 32)
                    atomic_json(stage/f"smoke_c{n}.json", result)
                atomic_json(stage/"COMPLETE.json", {"load_and_generate_smoke": True,
                    "not_a_certificate": "No AR-reference or stochastic correctness comparison"})
            else:
                for length in (512, 1024):
                    for concurrency in (64, 128):
                        for repeat in range(args.repeats+1):
                            label = f"{'warmup' if repeat == 0 else 'measured'}_c{concurrency}_l{length}_r{repeat}"
                            atomic_json(stage/"control.json", {"label": label})
                            result = wave(base, workload["rows"][:concurrency], model, length, args.output_tokens)
                            atomic_json(stage/f"{label}.json", result)
                            print(json.dumps({"stage": stage.name, "label": label,
                                "wall_s": result["wall_s"], "tokens_per_s": result["tokens_per_s"]}), flush=True)
                # A short final call drains measured CUDA events without attributing
                # its work to measurement; no per-phase synchronization is added.
                atomic_json(stage/"control.json", {"label": "flush"})
                wave(base, workload["rows"][:1], model, 512, 8)
                if mode == "events":
                    events = summarize_events(stage)
                    expected = args.repeats*4
                    if len(events) != expected:
                        raise ValueError(f"Expected {expected} timing groups, got {len(events)}")
                    atomic_json(stage/"component_summary.json", events)
                atomic_json(stage/"COMPLETE.json", {"bounded_screen_complete": True})
            (stage/"gpu_after.csv").write_text(gpu_telemetry(args.gpu))
    except BaseException as error:
        atomic_json(stage/"FAILED.json", {"error": repr(error)})
        raise
    finally:
        subprocess.run(["docker", "stop", "--time", "15", container], capture_output=True, timeout=40)
        if proc is not None:
            proc.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[4], default=4)
    parser.add_argument("--models", nargs="+", choices=["dense", "moe"], default=["dense", "moe"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.repeats <= 3 or not 64 <= args.output_tokens <= 256:
        raise ValueError("Use new output and bounded screening settings")
    if not 300 <= args.deadline_seconds <= 10800:
        raise ValueError("Deadline must be between five minutes and three hours")
    args.output = args.output.resolve()
    if ROOT.resolve() not in args.output.parents:
        raise ValueError("Screen output must be within the task storage root")
    lock = (ROOT/"gpu_4_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    args.output.mkdir(parents=True)

    def stop(signum, frame):
        raise TimeoutError(f"Bounded screen stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(args.deadline_seconds)
    repo = Path(__file__).resolve().parents[1]
    registries = {"dense": json.loads((ROOT/"models.json").read_text()),
                  "moe": json.loads((ROOT/"models_moe_screen_20260928.json").read_text())}
    started = time.monotonic()
    try:
        source = args.output/"source"
        restore(repo/"vendor/sglang_ragged_20260723", source)
        cache = args.output/"runtime_cache"
        cache.mkdir()
        workload = make_workload(registries)
        atomic_json(args.output/"workload.json", workload)
        atomic_json(args.output/"config.json", {"gpu": gpu, "image": IMAGE, "models": registries,
            "code_commit": command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "workload_sha256": sha(args.output/"workload.json"),
            "scope": "Recovered spec-v2 fixed B8/B16, target graphs/draft eager, FlashInfer, TP1 BF16, GPU4 only",
            "limitations": ["not an adaptive-policy benchmark", "not a task accuracy test", "no kernel trace",
                "two timing repeats only", "actual full-batch occupancy must be checked", "MoE Triton BF16 baseline"]})
        for name in args.models:
            # Test actual model/drafter load and generation at C1/C4 before scaling.
            serve(args, name, registries[name], 16, "clean", workload, gpu, source, cache, smoke=True)
            if args.smoke_only:
                continue
            for block, modes in ((16, ("clean", "events")), (8, ("events", "clean"))):
                for mode in modes:
                    serve(args, name, registries[name], block, mode, workload, gpu, source, cache)
        atomic_json(args.output/"COMPLETE.json", {"elapsed_s": time.monotonic()-started,
            "workload_sha256": sha(args.output/"workload.json"), "config_sha256": sha(args.output/"config.json")})
    except BaseException as error:
        atomic_json(args.output/"FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
