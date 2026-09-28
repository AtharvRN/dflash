"""Package/reconstruct the historical SGLang runtime without modifying its sources.

The base archive is git-authored Python source, not an installed environment.
All dirty Python files are copied verbatim as a reviewable overlay. The deployed
acceptance kernel is a separately attributed overlay because it was missing from
the local parent worktree. No upstream repository is mutated or pushed to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def package(source, deployed, destination):
    if destination.exists():
        raise ValueError("Refusing to overwrite a recovery package")
    destination.mkdir(parents=True)
    base = destination / "base-python.tar.gz"
    git(source, "archive", "--format=tar.gz", "--output=" + str(base),
        "HEAD", "python", "LICENSE")
    changed = set(git(source, "diff", "--name-only", "HEAD", "--", "python").decode().splitlines())
    changed.update(git(source, "ls-files", "--others", "--exclude-standard", "--", "python").decode().splitlines())
    if any(not (source / p).is_file() for p in changed):
        raise ValueError("Recovery cannot silently ignore deleted/non-file paths")
    entries = {}
    for rel in sorted(changed):
        dst = destination / "overlay" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
        entries[rel] = {"source": str(source / rel), "sha256": digest(dst),
                        "provenance": "historical_worktree_verbatim"}
    rel = "python/sglang/srt/speculative/triton_ops/dflash_accept_bonus.py"
    dst = destination / "overlay" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deployed / "dflash_accept_bonus.py", dst)
    entries[rel] = {"source": str(deployed / "dflash_accept_bonus.py"), "sha256": digest(dst),
                    "provenance": "saved_deployed_kernel_verbatim",
                    "parent_worktree_sha256": digest(source / rel)}
    # These two files agree between the deployment snapshot and parent worktree
    # except for later post-draft additions in the worker. Retain both hashes.
    comparison = {}
    for name in ("dflash_worker_v2.py", "dflash_prepare_block.py"):
        parent = source / "python/sglang/srt/speculative" / ("triton_ops" if name == "dflash_prepare_block.py" else "") / name
        comparison[name] = {"parent_sha256": digest(parent), "deployed_sha256": digest(deployed / name)}
    manifest = {
        "format": 1, "source_root": str(source),
        "base_commit": git(source, "rev-parse", "HEAD").decode().strip(),
        "source_status": git(source, "status", "--porcelain=v1").decode().splitlines(),
        "source_diff_sha256": hashlib.sha256(git(source, "diff", "HEAD", "--", "python")).hexdigest(),
        "base_archive_sha256": digest(base), "overlay": entries,
        "deployed_comparison": comparison,
        "scope": "Python runtime only; no benchmark code, data, models or environment",
        "limitations": ["greedy ragged path only", "draft CUDA graphs disabled in recovered worker",
                        "not a claim of bitwise identity to the old live pod"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"package": str(destination), "base_bytes": base.stat().st_size,
                      "overlay_files": len(entries), "base_commit": manifest["base_commit"]}))


def restore(package_path, destination):
    if destination.exists():
        raise ValueError("Refusing to overwrite a recovered source tree")
    manifest = json.loads((package_path / "manifest.json").read_text())
    base = package_path / "base-python.tar.gz"
    if digest(base) != manifest["base_archive_sha256"]:
        raise ValueError("Base archive checksum mismatch")
    actual = {p.relative_to(package_path / "overlay").as_posix()
              for p in (package_path / "overlay").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != set(manifest["overlay"]):
        raise ValueError("Overlay inventory mismatch")
    for rel, info in manifest["overlay"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("Invalid overlay path")
        if digest(package_path / "overlay" / rel) != info["sha256"]:
            raise ValueError("Overlay checksum mismatch: " + rel)
    destination.mkdir(parents=True)
    with tarfile.open(base, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    for rel in manifest["overlay"]:
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_path / "overlay" / rel, dst)
    shutil.copy2(package_path / "manifest.json", destination / "RECOVERY_MANIFEST.json")
    print(json.dumps({"source": str(destination), "verified_overlay_files": len(actual)}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--deployed", type=Path, required=True)
    pack.add_argument("--destination", type=Path, required=True)
    unpack = sub.add_parser("restore")
    unpack.add_argument("--package", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "package":
        package(args.source.resolve(), args.deployed.resolve(), args.destination.resolve())
    else:
        restore(args.package.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
