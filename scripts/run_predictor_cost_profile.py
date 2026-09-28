"""One bounded, GPU-locked predictor microbenchmark in the pinned image."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import time

from profile_sglang_latency import IMAGE, ROOT, atomic_json, check_gpu, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous measurements")
    lock = (ROOT / f"gpu_{args.gpu}_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    name = "atharv-predictor-profile-" + hashlib.sha256(str(args.output).encode()).hexdigest()[:12]
    launch = ["docker", "run", "--name", name, "--network", "none",
              "--gpus", "device="+gpu["uuid"], "--cpus", "4", "--shm-size", "1g",
              "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE", "--security-opt", "no-new-privileges",
              "--workdir", str(args.output), "-v", f"{repo}:{repo}:ro", "-v", f"{ROOT}:{ROOT}:ro",
              "-v", f"{args.output}:{args.output}:rw", "-e", "NVIDIA_TF32_OVERRIDE=0",
              "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4",
              "--entrypoint", "python", IMAGE, str(repo/"scripts/profile_predictor_cost.py"),
              "--root", str(ROOT), "--output", str(args.output/"predictor_cost.json")]
    atomic_json(args.output/"launch.json", launch)
    atomic_json(args.output/"config.json", {"gpu": gpu, "image": IMAGE,
                "code_commit": command(["git", "rev-parse", "HEAD"], cwd=repo).strip()})
    started = time.monotonic()
    try:
        with (args.output/"container.log").open("x") as log:
            subprocess.run(launch, check=True, timeout=360, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(args.output/"COMPLETE.json", {"success": True, "elapsed_s": time.monotonic()-started})
    except BaseException as error:
        atomic_json(args.output/"FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise
    finally:
        subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, timeout=30)


if __name__ == "__main__":
    main()
