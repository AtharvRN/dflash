"""Copy immutable pilot inputs to node-local storage; never modify PVC sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/workspace/dflashv2_data"))
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bindings = {}

    def copy_file(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if source.stat().st_size != destination.stat().st_size:
            raise RuntimeError(f"Incomplete copy: {source}")

    inputs = ["manifests/qwen3_4b_instruct_100k_messages.jsonl",
              "runs/prefusion_pilot_20260915/cache/manifest.json",
              "splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719",
              "runs/policy_granularity_20260927/cache"]
    for relative in inputs:
        source = args.source / relative
        files = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
        for path in files:
            dest = args.destination / path.relative_to(args.source)
            copy_file(path, dest)
            with path.open("rb") as handle:
                source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            with dest.open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == source_hash
            bindings[str(path.relative_to(args.source))] = source_hash
        print("STAGED_INPUT", relative, flush=True)
    models = json.loads(args.models.read_text())
    for name, model in models.items():
        source = Path(model["path"])
        destination = args.destination / "models" / name / model["revision"]
        # Preserve revision basename, materialize snapshot symlinks, and avoid
        # mmap'ing shared-network weights independently in each worker.
        for path in sorted(source.iterdir()):
            if path.is_file():
                copy_file(path, destination / path.name)
                print("STAGED_MODEL_FILE", name, path.name, path.stat().st_size, flush=True)
        model["path"] = str(destination)
    (args.destination / "models.json").write_text(json.dumps(models, indent=2)+"\n")
    proof = {"source": str(args.source), "destination": str(args.destination),
             "input_sha256": bindings, "models": models, "elapsed_seconds": time.monotonic()-started,
             "model_copy_check": "file size; snapshot revisions preserved; no full weight checksum"}
    (args.destination / "staging.json").write_text(json.dumps(proof, indent=2)+"\n")
    print("STAGING_COMPLETE", json.dumps({"elapsed_seconds": proof["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
