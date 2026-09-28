"""Bounded GPU-locked primitive check in an isolated recovered runtime."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
from pathlib import Path
import subprocess
import time

from profile_sglang_latency import IMAGE, ROOT, atomic_json, check_gpu, command
from recover_legacy_ragged import restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite previous evidence")
    lock = (ROOT / f"gpu_{args.gpu}_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    source = args.output / "source"
    restore(repo / "vendor/sglang_ragged_20260723", source)
    name = "atharv-ragged-primitives-" + hashlib.sha256(str(args.output).encode()).hexdigest()[:12]
    launch = ["docker", "run", "--name", name, "--network", "none", "--gpus", "device="+gpu["uuid"],
              "--cpus", "4", "--shm-size", "1g", "--cap-drop", "ALL", "--cap-add", "DAC_OVERRIDE",
              "--security-opt", "no-new-privileges", "--workdir", str(args.output),
              "-v", f"{repo}:{repo}:ro", "-v", f"{args.output}:{args.output}:rw",
              "-e", f"PYTHONPATH={source}/python", "-e", "PYTHONDONTWRITEBYTECODE=1",
              "-e", "OMP_NUM_THREADS=4", "-e", "NVIDIA_TF32_OVERRIDE=0", "-e", "LOGNAME="+getpass.getuser(),
              "-e", f"TRITON_CACHE_DIR={args.output}/triton", "--entrypoint", "python", IMAGE,
              str(repo / "scripts/check_ragged_primitives.py"), "--source", str(source), "--device", "cuda",
              "--output", str(args.output / "primitives.json")]
    atomic_json(args.output / "config.json", {"gpu": gpu, "image": IMAGE,
                "code_commit": command(["git", "rev-parse", "HEAD"], cwd=repo).strip(), "launch": launch})
    started = time.monotonic()
    try:
        with (args.output / "container.log").open("x") as log:
            subprocess.run(launch, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(args.output / "COMPLETE.json", {"success": True, "elapsed_s": time.monotonic()-started})
    except BaseException as error:
        atomic_json(args.output / "FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise
    finally:
        subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, timeout=30)


if __name__ == "__main__":
    main()
