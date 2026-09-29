"""Batched frozen-prefix/B16-candidate replay for intermediate target probes.

Source artifacts are read-only. Fresh target labels and hidden states come from
one verification forward. Recomputed draft confidence is explicitly a replay
baseline, not a claim to recover the original cached-forward draft logits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import accepted_lengths
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json


def select_rows(train_cache, eval_cache, split_dir, train_rows, smoke=False):
    from scripts.analyze_policy_granularity import load_pairs
    from scripts.train_actual_block_predictor import load_training_pairs

    ev_config, ev_rows, _, _ = load_pairs(eval_cache)
    _, _, training_audit, index = load_training_pairs(train_cache, split_dir, ev_config)
    if not 1 <= train_rows <= len(index):
        raise ValueError("Training subset exceeds audited source")
    selected, loaded = [], {}
    for item in index[:train_rows]:
        pid = item["prompt_id"]
        if pid not in loaded:
            loaded[pid] = {s["cycle"]: s for s in
                          json.loads((train_cache / f"prompt_{pid}.json").read_text())["states"]}
        row = loaded[pid][item["cycle"]]
        if row["prefix_sha256"] != item["prefix_sha256"] or not row["eligible"]:
            raise ValueError("Training index/prefix mismatch")
        selected.append(row)
    selected.extend(r for r in ev_rows if r["eligible"])
    if smoke:
        selected = [r for group in ("train", "calibration", "assessment")
                    for r in [x for x in selected if x["group"] == group][:8]]
    counts = Counter(r["group"] for r in selected)
    if not smoke and counts != {"train": train_rows, "calibration": 342, "assessment": 1416}:
        raise ValueError(f"Unexpected fixed group membership: {counts}")
    groups = {g: {r["prompt_id"] for r in selected if r["group"] == g} for g in counts}
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError("Prompt leakage")
    rows = []
    for i, row in enumerate(selected):
        prefix = row["prefix_token_ids"]
        if hashlib.sha256(np.asarray(prefix, dtype=np.int64).tobytes()).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Prefix hash mismatch")
        rows.append({"row": i, **{k: row[k] for k in
                    ("prompt_id", "cycle", "source", "group", "prefix_length", "prefix_sha256", "prefix_token_ids")},
                    "draft_ids": row["outcomes"]["16"]["draft_ids"],
                    "old_accepted_len": row["outcomes"]["16"]["accepted"]})
    return rows, ev_config, training_audit


def pack_rows(rows, pad=0):
    """Left-pad committed prefixes; anchor remains the first verification row."""
    lengths = np.asarray([r["prefix_length"] for r in rows], dtype=np.int64)
    width = int(lengths.max())
    ids = np.full((len(rows), width), pad, dtype=np.int64)
    mask = np.zeros_like(ids)
    block = np.zeros((len(rows), 16), dtype=np.int64)
    for i, row in enumerate(rows):
        prefix = row["prefix_token_ids"]
        if len(prefix) != lengths[i] + 1 or lengths[i] < 1 or len(row["draft_ids"]) != 15:
            raise ValueError("Invalid saved prefix/candidate alignment")
        ids[i, -lengths[i]:] = prefix[:-1]
        mask[i, -lengths[i]:] = 1
        block[i] = [prefix[-1], *row["draft_ids"]]
    positions = np.maximum(mask.cumsum(1) - 1, 0)
    return ids, mask, positions, block, lengths


def replay_batch(rows, target, draft, layers):
    import torch
    from transformers import DynamicCache

    to_gpu = lambda x: torch.as_tensor(x, device="cuda")
    ids, mask, positions, block, lengths = map(to_gpu, pack_rows(rows))
    early, hooks = {}, []
    for idx in draft.target_layer_ids:
        hooks.append(target.model.layers[idx].register_forward_hook(
            lambda module, inputs, output, idx=idx: early.__setitem__(idx, output.detach())))
    cache = DynamicCache()
    try:
        pre = target(ids, attention_mask=mask, position_ids=positions,
                     past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    anchor_match = pre.logits[:, -1].argmax(-1) == block[:, 0]
    raw = torch.cat([early[i] for i in draft.target_layer_ids], dim=-1)
    early.clear()
    del pre
    query_positions = lengths[:, None] + torch.arange(16, device="cuda")[None]
    all_positions = torch.cat([positions, query_positions], dim=1)
    draft_mask = torch.cat([mask, torch.ones_like(block)], dim=1)
    # DFlash's SDPA wrapper consumes a 4D additive mask, not a causal mask.
    additive = torch.zeros((len(rows), 1, 1, draft_mask.shape[1]), dtype=torch.bfloat16, device="cuda")
    additive.masked_fill_(~draft_mask[:, None, None].bool(), torch.finfo(torch.bfloat16).min)
    noise = torch.full_like(block, draft.mask_token_id)
    noise[:, 0] = block[:, 0]
    drafted = draft(target_hidden=raw, noise_embedding=target.model.embed_tokens(noise),
                    position_ids=all_positions, attention_mask=additive, use_cache=False, is_causal=False)
    del raw, additive
    logits = target.lm_head(drafted[:, 1:]).float()
    lp = logits.log_softmax(-1)
    chosen = logits.gather(-1, block[:, 1:, None]).squeeze(-1)
    draft_stats = torch.stack([lp.gather(-1, block[:, 1:, None]).squeeze(-1),
                               -(lp.exp() * lp).sum(-1), chosen - logits.amax(-1)], dim=-1)
    draft_argmax = logits.argmax(-1)
    del logits, lp, drafted
    intermediate, hooks = {}, []
    for layer in layers:
        hooks.append(target.model.layers[layer - 1].register_forward_hook(
            lambda module, inputs, output, layer=layer: intermediate.__setitem__(layer, output[:, :15].detach().clone())))
    try:
        verified = target(block, attention_mask=draft_mask, position_ids=query_positions,
                          past_key_values=cache, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()
    posterior = verified.logits[:, :15].argmax(-1)
    matches = posterior == block[:, 1:]
    as_np = lambda x: x.detach().cpu().numpy()
    result = {"matches": as_np(matches).astype(np.uint8), "target_argmax": as_np(posterior),
              "candidate_ids": as_np(block[:, 1:]), "anchor_match": as_np(anchor_match),
              "draft_argmax": as_np(draft_argmax), "draft_stats": as_np(draft_stats),
              "candidate_vectors": as_np(target.lm_head.weight[block[:, 1:]].to(torch.float16))}
    for layer in layers:
        h = intermediate.pop(layer)
        result[f"hidden_L{layer}"] = as_np(h.to(torch.float16))
        # An untuned logit-lens diagnostic, explicitly NOT a trained tuned lens.
        z = target.lm_head(target.model.norm(h)).float()
        picked = z.gather(-1, block[:, 1:, None]).squeeze(-1)
        result[f"lens_rank_L{layer}"] = as_np((z > picked[..., None]).sum(-1) + 1).astype(np.int32)
        result[f"lens_margin_L{layer}"] = as_np(picked - z.amax(-1))
        del h, z, picked
    return result


def backup_file(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(path, temporary)
    if sha256(temporary) != sha256(path):
        raise ValueError("Backup checksum failure")
    temporary.replace(destination)


def seed_mapping(original, expanded):
    """Map immutable original states into a larger nested training selection."""
    key = lambda r: (r["group"], str(r["prompt_id"]), int(r["cycle"]))
    lookup = {key(r): i for i, r in enumerate(expanded)}
    if len(lookup) != len(expanded) or len({key(r) for r in original}) != len(original):
        raise ValueError("Duplicate state identity")
    mapping = []
    for row in original:
        if key(row) not in lookup:
            raise ValueError("Original state missing from expansion")
        idx = lookup[key(row)]
        if {k: v for k, v in row.items() if k != "row"} != {k: v for k, v in expanded[idx].items() if k != "row"}:
            raise ValueError("Original state contents changed")
        mapping.append(idx)
    original_train = [key(r) for r in original if r["group"] == "train"]
    expanded_train = [key(r) for r in expanded if r["group"] == "train"]
    if expanded_train[:len(original_train)] != original_train:
        raise ValueError("Training subsets are not nested")
    if {key(r) for r in original if r["group"] != "train"} != {key(r) for r in expanded if r["group"] != "train"}:
        raise ValueError("Frozen validation membership changed")
    return np.asarray(mapping, dtype=np.int64)


def array_row_digest(array, indices):
    h = hashlib.sha256()
    for start in range(0, len(indices), 64):
        h.update(np.ascontiguousarray(array[indices[start:start + 64]]).tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("train-cache", "eval-cache", "split-dir", "models", "output", "backup"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--train-rows", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=[6, 9, 12, 18, 24])
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed-cache", type=Path, help="Preserve all existing states/features; replay only new training rows")
    args = p.parse_args()
    if args.output.exists() or args.backup.exists() or args.output == args.backup:
        raise ValueError("Use two fresh, separate destinations")
    if args.batch_size < 1 or args.max_seconds < 1 or len(set(args.layers)) != len(args.layers) or not all(0 < l < 36 for l in args.layers):
        raise ValueError("Invalid collection limits/layers")
    used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError("Requested GPU is occupied; refusing to launch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel

    if transformers.__version__ != "4.57.1":
        raise ValueError("Require the existing Transformers 4.57.1 replay environment")
    torch.set_num_threads(4)
    torch.manual_seed(929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, ev_config, training_audit = select_rows(args.train_cache, args.eval_cache, args.split_dir, args.train_rows, args.smoke)
    models = json.loads(args.models.read_text())
    if models != ev_config["models"]:
        raise ValueError("Source model identity changed")
    seed_complete, seed_rows, reuse = None, [], np.array([], dtype=np.int64)
    if args.seed_cache:
        seed_complete = json.loads((args.seed_cache / "COMPLETE.json").read_text())
        if not seed_complete["passed"] or args.smoke:
            raise ValueError("Expansion requires a completed nonsmoke seed cache")
        for name, expected in seed_complete["binding"].items():
            if Path(name).name != name or sha256(args.seed_cache / name) != expected:
                raise ValueError("Seed cache hash mismatch")
        seed_config = json.loads((args.seed_cache / "config.json").read_text())
        if seed_config["smoke"] or seed_config["models"] != models or seed_config["layers"] != args.layers:
            raise ValueError("Seed collection protocol differs")
        if any(seed_config[k] != v for k, v in {"dtype": "bfloat16", "attention": "sdpa", "tf32": False}.items()):
            raise ValueError("Seed numerical protocol differs")
        seed_rows = json.loads((args.seed_cache / "source_rows.json").read_text())
        reuse = seed_mapping(seed_rows, rows)
        if len(reuse) >= len(rows):
            raise ValueError("Expansion has no new training states")
    for path in (args.output, args.backup):
        path.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"models": models, "rows": len(rows), "groups": dict(Counter(r["group"] for r in rows)),
                   "dtype": "bfloat16", "feature_storage": "float16", "attention": "sdpa", "tf32": False,
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "script_sha256": sha256(__file__), "module_sha256": sha256("dflash/midverify.py"),
                   "source_bindings": {str(root / "COMPLETE.json"): sha256(root / "COMPLETE.json") for root in (args.train_cache, args.eval_cache)},
                   "gpu_name": torch.cuda.get_device_name(), "training_source_audit": training_audit,
                   "candidate_protocol": "Saved B16 candidates unchanged. Fresh target labels and h_(j-1)^L from SAME forward; original labels are audit-only.",
                   "draft_baseline_protocol": "Recompute full-prefix DFlash confidence OF SAVED candidates; may differ numerically from original incremental cached draft. Record argmax disagreement.",
                   "layer_convention": "L means after L decoder layers, before final RMSNorm; hidden_L[j-1] predicts candidate_ids[j-1] (one-based candidate j).",
                   "selection": "first 2000 (or --train-rows) rows of the audited immutable 10k training index; all eligible fixed evaluation states",
                   "draft_stat_columns": ["candidate_logprob", "entropy", "candidate_minus_top_logit"],
                   "lens": "untuned target final RMSNorm + original LM head; diagnostic full-vocabulary compute cost NOT free",
                   "scope": "greedy fixed-state offline replay, not throughput or deployed correctness"})
    config["reused_rows"] = len(reuse)
    if args.seed_cache:
        config["seed_completion_sha256"] = sha256(args.seed_cache / "COMPLETE.json")
        config["seed_row_mapping"] = reuse.tolist()
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "source_rows.json", rows)
    for name in ("config.json", "source_rows.json"):
        backup_file(args.output / name, args.backup / name)
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    if target.config.num_hidden_layers != 36 or target.config.hidden_size != 2560:
        raise ValueError("Wrong target architecture")
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    started = time.monotonic()
    # Length sorting changes compute order only; arrays retain immutable row IDs.
    reused = set(reuse.tolist())
    order = sorted((i for i in range(len(rows)) if i not in reused), key=lambda i: (rows[i]["prefix_length"], i))
    if args.seed_cache and any(rows[i]["group"] != "train" for i in order):
        raise ValueError("Expansion may only replay new training states")
    shards, parity, completed = [], [], len(reuse)
    print("MODELS_READY", json.dumps({"groups": config["groups"], "batch_size": args.batch_size}), flush=True)
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            if time.monotonic() - started > args.max_seconds:
                raise RuntimeError("Time bound reached; partial shards preserved, no COMPLETE")
            take = order[start:start + args.batch_size]
            current = [rows[i] for i in take]
            result = replay_batch(current, target, draft, args.layers)
            result["row_ids"] = np.asarray(take, dtype=np.int64)
            result["accepted_len"] = accepted_lengths(result["matches"])
            # Independent unpadded numerical control on spread-out batches.
            if start // args.batch_size in (0, 1, len(order) // args.batch_size // 2, (len(order) - 1) // args.batch_size):
                for local in sorted(set([0, len(take) - 1])):
                    one = replay_batch([current[local]], target, draft, args.layers)
                    h_error = {}
                    for layer in args.layers:
                        a = result[f"hidden_L{layer}"][local].astype(np.float32)
                        b = one[f"hidden_L{layer}"][0].astype(np.float32)
                        h_error[str(layer)] = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
                    parity.append({"row": take[local], "batched_accepted": int(result["accepted_len"][local]),
                        "single_accepted": int(accepted_lengths(one["matches"])[0]),
                        "argmax_differences": int((result["target_argmax"][local] != one["target_argmax"][0]).sum()),
                        "hidden_relative_l2": h_error})
                    if max(h_error.values()) > .08:
                        raise RuntimeError("Batch/single hidden discrepancy exceeds numerical smoke tolerance")
            name = f"chunks/batch_{start // args.batch_size:05d}.npz"
            shard = args.output / name
            shard.parent.mkdir(exist_ok=True)
            if any(not np.isfinite(v).all() for v in result.values()):
                raise ValueError("Nonfinite captured values")
            np.savez(shard, **result)
            backup_file(shard, args.backup / name)
            shards.append({"path": name, "sha256": sha256(shard), "row_ids": take})
            completed += len(take)
            progress = {"completed": completed, "planned": len(rows), "elapsed_s": time.monotonic() - started,
                        "newly_replayed": completed - len(reuse), "reused_rows": len(reuse),
                        "states_per_s": (completed - len(reuse)) / max(time.monotonic() - started, 1),
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
            atomic_json(args.output / "progress.json", progress)
            backup_file(args.output / "progress.json", args.backup / "progress.json")
            if start // args.batch_size % 8 == 0 or completed == len(rows):
                print(json.dumps(progress), flush=True)
    # Materialize training arrays only after every durable chunk exists.
    arrays = {}
    seen = np.zeros(len(rows), dtype=np.int64)
    if args.seed_cache:
        for name in seed_complete["binding"]:
            if not name.endswith(".npy"):
                continue
            source = np.load(args.seed_cache / name, mmap_mode="r", allow_pickle=False)
            if source.shape[0] != len(reuse):
                raise ValueError("Seed array row count mismatch")
            arrays[Path(name).stem] = np.lib.format.open_memmap(args.output / name, mode="w+",
                dtype=source.dtype, shape=(len(rows), *source.shape[1:]))
            for start in range(0, len(reuse), 64):
                arrays[Path(name).stem][reuse[start:start + 64]] = source[start:start + 64]
        seen[reuse] = 1
    for receipt in shards:
        path = args.output / receipt["path"]
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Chunk checksum changed")
        with np.load(path, allow_pickle=False) as chunk:
            if args.seed_cache and set(chunk.files) - {"row_ids"} != set(arrays):
                raise ValueError("New and frozen feature schemas differ")
            idx = chunk["row_ids"]
            if idx.tolist() != receipt["row_ids"]:
                raise ValueError("Chunk index mismatch")
            seen[idx] += 1
            for key in chunk.files:
                if key == "row_ids":
                    continue
                value = chunk[key]
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(args.output / f"{key}.npy", mode="w+",
                        dtype=value.dtype, shape=(len(rows), *value.shape[1:]))
                if value.dtype != arrays[key].dtype or value.shape[1:] != arrays[key].shape[1:]:
                    raise ValueError("Chunk array dtype/shape mismatch")
                arrays[key][idx] = value
    if not np.all(seen == 1):
        raise ValueError("Missing/duplicate materialized rows")
    preservation = {}
    if args.seed_cache:
        for key, value in arrays.items():
            source = np.load(args.seed_cache / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            if source.dtype != value.dtype or source.shape[1:] != value.shape[1:]:
                raise ValueError("Frozen array type/shape changed")
            preservation[key] = {}
            for group in ("train", "calibration", "assessment"):
                idx = np.asarray([i for i, r in enumerate(seed_rows) if r["group"] == group])
                original_digest = array_row_digest(source, idx)
                if array_row_digest(value, reuse[idx]) != original_digest:
                    raise ValueError("Frozen array bytes changed")
                preservation[key][group] = original_digest
        if sha256(args.seed_cache / "COMPLETE.json") != config["seed_completion_sha256"]:
            raise ValueError("Seed changed during collection")
    records, terminal = [], []
    for i, row in enumerate(rows):
        accepted = int(arrays["accepted_len"][i])
        is_terminal = any(t in eos for t in row["draft_ids"][:accepted])
        terminal.append(is_terminal)
        records.append({k: v for k, v in row.items() if k != "prefix_token_ids"} | {
            "accepted_len": accepted, "label_replay_match": accepted == row["old_accepted_len"],
            "anchor_replay_match": bool(arrays["anchor_match"][i]), "replayed_accepted_eos": is_terminal,
            "draft_argmax_matches_saved": int((arrays["draft_argmax"][i] == arrays["candidate_ids"][i]).sum())})
    if args.seed_cache:
        original_records = json.loads((args.seed_cache / "rows.json").read_text())
        for old, idx in zip(original_records, reuse):
            if {k: v for k, v in old.items() if k != "row"} != {k: v for k, v in records[idx].items() if k != "row"}:
                raise ValueError("Frozen labels/metadata changed")
    audit = {"passed": not any(terminal), "rows": len(rows), "groups": config["groups"],
             "label_replay_matches": sum(r["label_replay_match"] for r in records),
             "anchor_replay_matches": sum(r["anchor_replay_match"] for r in records),
             "draft_argmax_match_fraction": sum(r["draft_argmax_matches_saved"] for r in records) / (15 * len(rows)),
             "fresh_accepted_eos_states": int(sum(terminal)), "batch_single_controls": parity,
             "elapsed_s": time.monotonic() - started, "candidate_identity_preserved": True,
             "source_labels_used_for_training": False, "assessment_membership_unchanged": not args.smoke}
    audit.update({"reused_rows": len(reuse), "newly_replayed_rows": len(order),
                  "seed_array_content_bytes_preserved": bool(args.seed_cache),
                  "seed_array_group_sha256": preservation})
    atomic_json(args.output / "rows.json", records)
    atomic_json(args.output / "chunks.json", shards)
    atomic_json(args.output / "audit.json", audit)
    for a in arrays.values():
        a.flush()
    names = ["config.json", "source_rows.json", "rows.json", "chunks.json", "audit.json"] + [f"{k}.npy" for k in arrays]
    for name in names:
        backup_file(args.output / name, args.backup / name)
    if not audit["passed"]:
        raise RuntimeError("Fresh accepted-EOS states need explicit censoring review; no training launched")
    atomic_json(args.output / "COMPLETE.json", {"passed": True, "rows": len(rows),
                "binding": {name: sha256(args.output / name) for name in names}})
    backup_file(args.output / "COMPLETE.json", args.backup / "COMPLETE.json")
    print("COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
