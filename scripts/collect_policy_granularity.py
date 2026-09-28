"""Actual B2--B16 outcomes on fixed calibration/assessment prompt groups.

Common B16 reference states, not closed-loop deployment or a throughput test.
The collector never trains or selects a predictor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256, validate_states

CHECKPOINT_SHA = "84a37d00f61b1c8ae5ecd76e1ad49b4cd00850613e783edf72ecdf875663af35"


def select_training_groups(manifest, split_dir, seed, limit):
    """Uniform canonical training prompts; exclude cross-split exact duplicates."""
    train = set(map(int, json.loads((Path(split_dir)/"train_prompt_ids.json").read_text())["train_prompt_ids"]))
    val = set(map(int, json.loads((Path(split_dir)/"val_prompt_ids.json").read_text())["val_prompt_ids"]))
    if train & val or limit < 1:
        raise ValueError("Invalid training selection/split")
    rows, seen, groups = [], set(), defaultdict(set)
    with Path(manifest).open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = int(row["manifest_index"])
            if pid in seen:
                raise ValueError("Duplicate manifest prompt")
            seen.add(pid)
            digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            group = "train" if pid in train else "validation" if pid in val else "other"
            groups[digest].add(group)
            if group == "train":
                rows.append({**row, "group": "train", "content_sha256": digest})
    unique, eligible = set(), []
    for row in rows:
        digest = row["content_sha256"]
        if groups[digest] == {"train"} and digest not in unique:
            eligible.append(row)
            unique.add(digest)
    random.Random(seed).shuffle(eligible)
    if len(eligible) < limit:
        raise ValueError("Insufficient unique canonical training prompts")
    return eligible[:limit]


def select_groups(manifest, pilot_manifest, split_dir, seed, limit=0):
    pilot = json.loads(Path(pilot_manifest).read_text())
    train = set(map(int, json.loads((Path(split_dir)/"train_prompt_ids.json").read_text())["train_prompt_ids"]))
    val = set(map(int, json.loads((Path(split_dir)/"val_prompt_ids.json").read_text())["val_prompt_ids"]))
    if train & val:
        raise ValueError("Canonical prompt overlap")
    wanted = {int(s["prompt_id"]): s["group"] for s in pilot["shards"]
              if s.get("rows", 0) and s["group"] in ("calibration", "assessment")}
    expected = {int(s["prompt_id"]): s for s in pilot["collection_config"]["selected_prompts"]}
    rows, seen = [], set()
    with Path(manifest).open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = int(row["manifest_index"])
            if pid not in wanted:
                continue
            if pid in seen or pid not in val or pid in train:
                raise ValueError("Invalid or duplicate evaluation prompt")
            seen.add(pid)
            digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if digest != expected[pid]["content_sha256"] or wanted[pid] != expected[pid]["group"]:
                raise ValueError("Original prompt content/group changed")
            rows.append({**row, "group": wanted[pid], "content_sha256": digest})
    if seen != set(wanted):
        raise ValueError("Missing evaluation prompts")
    # Interleave both fixed groups, preserving useful coverage in bounded smoke runs.
    rng = random.Random(seed)
    groups = [[r for r in rows if r["group"] == g] for g in ("calibration", "assessment")]
    for group in groups:
        rng.shuffle(group)
    ordered = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    return ordered[:limit] if limit else ordered


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "pilot-manifest", "split-dir", "models", "checkpoint", "output", "backup"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--limit-prompts", type=int, default=0)
    p.add_argument("--states-per-prompt", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=928)
    p.add_argument("--training-rows", type=int, default=0,
                   help="Training-only mode: stop after this many eligible rows; finish the last prompt")
    args = p.parse_args()
    if args.output.exists() or args.backup.exists():
        raise ValueError("Refusing existing experiment destinations")
    if min(args.states_per_prompt, args.max_seconds) < 1 or args.max_new_tokens < 32 or min(args.limit_prompts, args.training_rows) < 0:
        raise ValueError("Invalid limits")
    if args.training_rows and args.limit_prompts < 1:
        raise ValueError("Training collection requires an explicit prompt bound")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Not the agreed frozen 100k MLP checkpoint")
    selected = (select_training_groups(args.manifest, args.split_dir, args.seed, args.limit_prompts)
                if args.training_rows else select_groups(args.manifest, args.pilot_manifest, args.split_dir, args.seed, args.limit_prompts))
    used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError("GPU is occupied; no inference launched")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel
    from scripts.diagnose_dflash_paired_lengths import run_prompt

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Expected Transformers 4.57.1")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.blocks = list(range(2, 17))
    args.reverse_check_states, args.canonical_check_states = 4, 8
    args.max_states = len(selected)*args.states_per_prompt
    models = json.loads(args.models.read_text())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"models": models, "checkpoint_sha256": CHECKPOINT_SHA,
        "collection_kind": "training" if args.training_rows else "evaluation",
        "prompt_sampling": "uniform shuffled canonical training prompts, exact cross-split content duplicates excluded" if args.training_rows else "unchanged fixed pilot validation groups",
        "prompt_ids": [int(r["manifest_index"]) for r in selected],
        "prompt_groups": {str(r["manifest_index"]): r["group"] for r in selected},
        "prompt_content_hashes": {str(r["manifest_index"]): r["content_sha256"] for r in selected},
        "group_prompts": dict(Counter(r["group"] for r in selected)),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(), "tf32": False, "dtype": "bfloat16", "attention": "sdpa",
        "temperature": 0, "thinking": False,
        "scope": "actual B2-B16 on common B16 reference states; " + ("canonical training data" if args.training_rows else "development evaluation") + ", NOT closed-loop or throughput",
        "feature": "latest pre-draft hidden_norm(fc(target_hidden)); separate causal fusion replay, FP16 storage",
        "request_rule": "use cycle-zero feature even if that outcome row is excluded; never use a later state as the request input",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_hashes": {str(f): sha256(f) for f in (args.manifest, args.pilot_manifest,
            args.split_dir/"train_prompt_ids.json", args.split_dir/"val_prompt_ids.json", args.models)},
        "source_hashes": {str(f): sha256(f) for f in (Path(__file__), Path("scripts/diagnose_dflash_paired_lengths.py"))}})
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)

    def backup(paths):
        for source in paths:
            temp = args.backup/(source.name+".tmp")
            shutil.copyfile(source, temp)
            if sha256(temp) != sha256(source):
                raise ValueError("Durable backup hash mismatch")
            temp.replace(args.backup/source.name)

    atomic_json(args.output/"config.json", config)
    backup([args.output/"config.json"])
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(models["target"]["path"], local_files_only=True)
    pilot = json.loads(args.pilot_manifest.read_text())
    if draft.target_layer_ids != pilot["target_layer_ids"]:
        raise ValueError("Target feature layers changed")
    for key in ("target", "draft"):
        expected = pilot["collection_config"]["hashes"][f"/tmp/dflash-prefusion-{key}/config.json"]
        if sha256(Path(models[key]["path"])/"config.json") != expected:
            raise ValueError("Target/draft config identity changed")

    class NoPolicies:
        def predict(self, fused):
            return {}

    started, receipts, rows, progress = time.monotonic(), [], [], []
    print("MODELS_READY", json.dumps(config["group_prompts"]), flush=True)
    for i, row in enumerate(selected):
        if args.training_rows and sum(r["eligible"] for r in rows) >= args.training_rows:
            break
        if time.monotonic()-started > args.max_seconds:
            print("TIME_BOUND: preserving partial evidence", flush=True)
            break
        batch, status = run_prompt(args, row, target, draft, tokenizer, NoPolicies(), len(rows))
        pid = int(row["manifest_index"])
        states = [r for r, _ in batch]
        for state in states:
            state["group"] = row["group"]
        validate_states(states, args.blocks)
        if states and (states[0]["cycle"] != 0 or states[0]["generated_before_anchor"] != 0):
            raise ValueError("Missing cycle-zero request feature")
        shard = args.output/f"prompt_{pid}.json"
        atomic_json(shard, {"progress": status, "states": states})
        paths = [shard]
        if states:
            features = np.stack([f for _, f in batch])
            if features.shape != (len(states), 2560) or not np.isfinite(features).all():
                raise ValueError("Invalid fresh fused features")
            feature_path = args.output/f"prompt_{pid}_fused.npy"
            np.save(feature_path, features, allow_pickle=False)
            paths.append(feature_path)
        receipt = {"prompt_id": pid, "source": row["source"], "group": row["group"],
                   "states": len(states), "files": {f.name: sha256(f) for f in paths}}
        receipt_path = args.output/f"receipt_{pid}.json"
        atomic_json(receipt_path, receipt)
        backup(paths+[receipt_path])
        rows.extend(states)
        receipts.append(receipt)
        progress.append(status)
        latest = {"prompts_processed": i+1, "prompts_planned": len(selected), "states": len(rows),
                  "eligible": sum(r["eligible"] for r in rows), "elapsed_s": time.monotonic()-started, "latest": status}
        atomic_json(args.output/"progress.json", latest)
        backup([args.output/"progress.json"])
        print(json.dumps(latest), flush=True)
    reached = sum(r["eligible"] for r in rows) >= args.training_rows if args.training_rows else len(receipts) == len(selected)
    summary = {"sample_complete": reached, "prompts": len(receipts), "states": len(rows),
        "target_training_rows": args.training_rows,
        "eligible_states": sum(r["eligible"] for r in rows), "elapsed_s": time.monotonic()-started,
        "group_states": dict(Counter(r["group"] for r in rows if r["eligible"])),
        "canonical_disagreements": sum(r["canonical_disagreements"] for r in rows),
        "canonical_comparisons": sum(r["canonical_checked"] for r in rows),
        "reverse_checked_states": sum(r["reverse_order_checked"] for r in rows), "prompt_progress": progress}
    atomic_json(args.output/"receipts.json", receipts)
    atomic_json(args.output/"collection_summary.json", summary)
    backup([args.output/"receipts.json", args.output/"collection_summary.json"])
    atomic_json(args.output/"COMPLETE.json", {**{k: summary[k] for k in ("sample_complete", "states", "eligible_states")},
        "binding": {f: sha256(args.output/f) for f in ("config.json", "receipts.json", "collection_summary.json")}})
    backup([args.output/"COMPLETE.json"])
    print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "prompt_progress"}), flush=True)
    if not summary["sample_complete"]:
        raise RuntimeError("Incomplete collection; do not assess as a complete experiment")


if __name__ == "__main__":
    main()
