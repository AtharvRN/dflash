"""Stage immutable public MoE checkpoints without changing the dense registry."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

SPECS = {
    "target": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"),
    "draft": ("z-lab/Qwen3-Coder-30B-A3B-DFlash", "98ca0e3e2e6a372f2789d3a5e146566194084317"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/scratch/zekaili/atharv/dflash"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.parent != root or output.name == "models.json":
        raise ValueError("Use a separate model registry directly under the task root")
    os.environ["HF_HOME"] = str(root / "hf")
    from huggingface_hub import HfApi, snapshot_download

    lock = (root / "moe_screen_download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists():
        raise FileExistsError("Completed registry exists; preserve it")
    api = HfApi(token=False)
    plans = {}
    needed = 0
    for kind, (repo, revision) in SPECS.items():
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Model revision did not resolve exactly")
        files = []
        snapshot = root / "hf/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        for item in info.siblings:
            if "/" in item.rfilename or not item.rfilename.endswith((".json", ".safetensors", ".py", ".txt", ".model", ".jinja")):
                continue
            files.append({"name": item.rfilename, "bytes": item.size,
                          "lfs_sha256": item.lfs.sha256 if item.lfs else None})
            existing = snapshot / item.rfilename
            if not existing.exists() or existing.stat().st_size != item.size:
                needed += item.size
        plans[kind] = {"repo": repo, "revision": revision, "files": files}
    free = shutil.disk_usage(root).free
    if free < needed + 100 * 2**30:
        raise RuntimeError(f"Insufficient headroom: free={free}, remaining_download={needed}, reserve=100 GiB")
    atomic_json(output.with_suffix(".plan.json"), {"models": plans, "free_bytes_before": free,
                "remaining_download_bytes": needed, "started_unix": time.time()})
    paths = {}
    for kind, plan in plans.items():
        print(json.dumps({"stage": "download", "kind": kind, "repo": plan["repo"], "revision": plan["revision"]}), flush=True)
        path = Path(snapshot_download(repo_id=plan["repo"], revision=plan["revision"], token=False,
                    cache_dir=str(root / "hf/hub"), max_workers=4,
                    allow_patterns=[f["name"] for f in plan["files"]]))
        verified = []
        for item in plan["files"]:
            file = path / item["name"]
            if file.stat().st_size != item["bytes"]:
                raise ValueError("Incorrect downloaded size: " + item["name"])
            sha = digest(file)
            if item["lfs_sha256"] and sha != item["lfs_sha256"]:
                raise ValueError("Incorrect downloaded SHA256: " + item["name"])
            verified.append({**item, "sha256": sha})
            print(json.dumps({"stage": "verified", "kind": kind, "file": item["name"]}), flush=True)
        paths[kind] = {"repo": plan["repo"], "revision": plan["revision"], "path": str(path), "files": verified}
    atomic_json(output, paths)
    print(json.dumps({"complete": True, "registry": str(output), "free_bytes_after": shutil.disk_usage(root).free}), flush=True)


if __name__ == "__main__":
    main()
