"""Bounded recovered-engine integration checks; these are NOT benchmarks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import time

import requests

from profile_sglang_latency import (IMAGE, ROOT, atomic_json, check_gpu, choose_http_port,
                                    command, load_workload, wait_ready)
from recover_legacy_ragged import restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--backend", choices=["flashinfer", "triton"], default="flashinfer")
    parser.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    parser.add_argument("--audit-forwards", type=int, default=8)
    parser.add_argument("--audit-min-bs", type=int, default=2)
    parser.add_argument("--concurrency", type=int, choices=[8, 16, 32, 64], default=8)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--fixed-cap", type=int, default=0,
                        help="Fixed output length with EOS ignored, for sustained-concurrency stress only")
    parser.add_argument("--pattern", choices=["rotating", "fixed16"], default="rotating")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use SGLang's existing batch-invariant mode as a correctness control, not a timing configuration")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Audit slot aliasing, isolate the worst row, and preserve large-discrepancy tensors/KV")
    parser.add_argument("--saved-target-fixture", type=Path,
                        help="Run only the B7 saved-state numerical fixture; no request workload")
    args = parser.parse_args()
    if not 2 <= args.audit_min_bs <= args.concurrency or not 0 <= args.fixed_cap <= 256:
        raise ValueError("Invalid bounded audit/stress settings")
    if args.output.exists():
        raise ValueError("Preserve previous evidence: output must be new")
    lock = (ROOT / f"gpu_{args.gpu}_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    source = args.output / "source"
    restore(repo / "vendor/sglang_ragged_20260723", source)
    models = json.loads((ROOT / "models.json").read_text())
    prompts = load_workload(ROOT / "runs/policy_granularity_20260927/cache")[:args.prompts]
    if len(prompts) < args.prompts:
        raise ValueError("Not enough audited development prompts")
    atomic_json(args.output / "workload.json", prompts)
    atomic_json(args.output / "config.json", {"gpu": gpu, "image": IMAGE, "models": models,
                "code_commit": command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "purpose": "correctness; expensive shadow forwards invalidate all latency/throughput"})
    cache = args.output / "runtime_cache"
    cache.mkdir()
    started = time.monotonic()
    container = None
    try:
        for mode in args.modes:
            stage = args.output / mode
            stage.mkdir()
            port = choose_http_port()
            base = f"http://127.0.0.1:{port}"
            container = "atharv-ragged-model-"+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
            launch = ["docker", "run", "--name", container, "--network", "host", "--gpus", "device="+gpu["uuid"],
                      "--cpus", "12", "--shm-size", "8g", "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
                      "--security-opt", "no-new-privileges", "--workdir", str(stage),
                      "-v", f"{repo}:{repo}:ro", "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{stage}:{stage}:rw",
                      "-v", f"{cache}:{cache}:rw"]
            env = {"PYTHONPATH": f"{repo}/scripts/ragged_audit_hook:{source}/python",
                   "PYTHONDONTWRITEBYTECODE": "1", "NVIDIA_TF32_OVERRIDE": "0", "LOGNAME": getpass.getuser(),
                   "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "hf"), "TOKENIZERS_PARALLELISM": "false",
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "XDG_CACHE_HOME": str(cache),
                   "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                   "CUDA_CACHE_PATH": str(cache / "cuda"), "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
                   "SGLANG_CACHE_DIR": str(cache / "sglang"), "SGLANG_DG_CACHE_DIR": str(cache / "deep_gemm"),
                   "TORCH_HOME": str(cache / "torch"), "TORCH_EXTENSIONS_DIR": str(cache / "extensions"),
                   "SGLANG_ENABLE_SPEC_V2": "1", "SGLANG_ENABLE_DFLASH_SPEC_V2": "1",
                   "SGLANG_DFLASH_FORCE_RAGGED_BLOCK_PATTERN": "16" if args.pattern == "fixed16" else "2,3,4,7,8,11,15,16",
                   "DFLASH_RAGGED_AUDIT_DIR": str(stage), "DFLASH_RAGGED_AUDIT_ROTATE": "0" if args.pattern == "fixed16" else "1",
                   "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1",
                   "DFLASH_RAGGED_AUDIT_FORWARDS": str(args.audit_forwards), "SGLANG_DFLASH_TIMING": "0"}
            env["DFLASH_RAGGED_AUDIT_MIN_BATCH"] = str(args.audit_min_bs)
            env["DFLASH_RAGGED_AUDIT_DIAGNOSTICS"] = "1" if args.diagnostics else "0"
            if args.saved_target_fixture:
                env["DFLASH_SAVED_TARGET_FIXTURE"] = str(args.saved_target_fixture)
            for k, v in env.items():
                launch += ["-e", f"{k}={v}"]
            launch += ["--entrypoint", "python", IMAGE, "-m", "sglang.launch_server",
                       "--model-path", models["target"]["path"], "--speculative-algorithm", "DFLASH",
                       "--speculative-draft-model-path", models["draft"]["path"],
                       "--speculative-dflash-block-size", "16", "--speculative-num-draft-tokens", "16",
                       "--speculative-dflash-dynamic-block-size", "--speculative-dflash-dynamic-block-arms",
                       ",".join(map(str, range(2, 17))), "--speculative-dflash-dynamic-warmup-batches", "1000000",
                       "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1", "--dtype", "bfloat16",
                       "--random-seed", "934",
                       "--attention-backend", args.backend, "--speculative-draft-attention-backend", args.backend,
                       "--mem-fraction-static", "0.60", "--max-running-requests", str(args.concurrency),
                       "--max-total-tokens", str(min(131072, args.concurrency*4096)),
                       "--context-length", "4096", "--cuda-graph-max-bs", str(args.concurrency),
                       "--cuda-graph-bs", *[str(n) for n in (1, 2, 4, 8, 16, 32, 64) if n <= args.concurrency],
                       "--disable-radix-cache", "--disable-piecewise-cuda-graph"]
            if mode == "eager":
                launch += ["--disable-cuda-graph"]
            if args.deterministic:
                launch += ["--enable-deterministic-inference"]
            atomic_json(stage / "launch.json", launch)
            with (stage / "server.log").open("x") as log:
                proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
                try:
                    wait_ready(base, container, stage / "server.log", timeout=900)
                    atomic_json(stage / "server_info.json", requests.get(base + "/get_server_info", timeout=10).json())
                    if args.saved_target_fixture:
                        if not (stage / "FIXTURE.json").is_file():
                            raise AssertionError("Saved-state fixture did not complete")
                        print(json.dumps({"mode": mode, "fixture_completed": True}), flush=True)
                        continue
                    for repeat in range(2):
                        items = prompts if repeat == 0 else list(reversed(prompts))

                        def send(pair):
                            i, item = pair
                            cap = args.fixed_cap or [1, 7, 16, 63, 96, 32, 8, 64][i % 8]
                            response = requests.post(base + "/generate", json={"input_ids": item["input_ids"],
                                "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": cap,
                                                    "ignore_eos": bool(args.fixed_cap)},
                                "return_logprob": False}, timeout=240)
                            response.raise_for_status()
                            data = response.json()
                            count = data.get("meta_info", {}).get("completion_tokens", 0)
                            if not 0 < count <= cap:
                                raise AssertionError(f"Invalid terminal/output count {count}, cap={cap}")
                            return {"prompt_id": item["prompt_id"], "cap": cap, "response": data}

                        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            results = list(pool.map(send, enumerate(items)))
                        atomic_json(stage / f"responses_{repeat}.json", results)
                        print(json.dumps({"mode": mode, "repeat": repeat, "responses": len(results)}), flush=True)
                    audits = [json.loads(line) for path in stage.glob("audit_*.jsonl") for line in path.read_text().splitlines()]
                    for role in ("draft", "target"):
                        if not any(r.get("role") == role for r in audits):
                            raise AssertionError("No actual mixed model audit: " + role)
                    if mode == "graph" and not any(r.get("role") == "target" and r["graph_used"] for r in audits):
                        raise AssertionError("Target graph mode never actually replayed")
                    atomic_json(stage / "CHECKS_RAN.json", {"audit_records": len(audits),
                        "note": "Completion is not a numerical-parity certificate; inspect detailed comparisons"})
                finally:
                    subprocess.run(["docker", "stop", "--time", "15", container], capture_output=True, timeout=35)
                    proc.wait(timeout=25)
                    container = None
        atomic_json(args.output / "COMPLETE.json", {"checks_ran": True, "elapsed_s": time.monotonic()-started,
            "note": "Inspect discrepancies before claiming correctness or running profiles"})
    except BaseException as error:
        atomic_json(args.output / "FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise
    finally:
        if container:
            subprocess.run(["docker", "stop", "--time", "15", container], capture_output=True, timeout=35)


if __name__ == "__main__":
    main()
