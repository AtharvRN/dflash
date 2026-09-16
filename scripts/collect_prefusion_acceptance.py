"""Collect paired latest raw/fused features on frozen greedy B16 trajectories."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_context_attention import atomic_json, buffered_backup


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_prompts(manifest, split_dir, reference, train_count, val_count, seed):
    train = set(json.loads((split_dir / "train_prompt_ids.json").read_text())["train_prompt_ids"])
    val = set(json.loads((split_dir / "val_prompt_ids.json").read_text())["val_prompt_ids"])
    train, val = set(map(int, train)), set(map(int, val))
    cal = set(map(int, reference["calibration_prompt_ids"]))
    if train & val or not cal <= val:
        raise ValueError("Inconsistent canonical/calibration prompt membership")
    pools = {"train": [], "calibration": [], "assessment": []}
    seen = set()
    content_groups = {}
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = int(row["manifest_index"])
            if pid in seen:
                raise ValueError("Duplicate manifest_index")
            seen.add(pid)
            group = "train" if pid in train else "calibration" if pid in cal else "assessment" if pid in val else None
            if group:
                digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                content_groups.setdefault(digest, set()).add(group)
                row = {**row, "content_sha256": digest}
                pools[group].append(row)
    # Identical message content under different IDs must not cross experiment groups.
    pools = {group: [r for r in rows if len(content_groups[r["content_sha256"]]) == 1]
             for group, rows in pools.items()}
    # Preserve the existing prompt-level calibration assignment; do not re-split rows.
    counts = {"train": train_count, "calibration": max(2, val_count // 5)}
    counts["assessment"] = val_count - counts["calibration"]
    rng = np.random.default_rng(seed)
    selected = []
    for group, rows in pools.items():
        if counts[group] < 1 or len(rows) < counts[group]:
            raise ValueError(f"Insufficient {group} prompts")
        order = rng.permutation(len(rows))[:counts[group]]
        selected.extend((group, rows[int(i)]) for i in order)
    return selected


def observable_label(block, accepted, eos, remaining):
    """Exclude terminal/capped accepted prefixes instead of inventing a rejection."""
    if remaining < 16:
        return False
    return not any(int(t) in eos for t in block[0, :accepted + 1].tolist())


@torch.inference_mode()
def collect_prompt(row, target, draft, tokenizer, args):
    from transformers import DynamicCache
    from dflash.model import extract_context_feature

    ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt").cuda()
    n = ids.shape[1]
    if n > args.max_prompt_tokens:
        return None, {"skipped": "prompt_length", "prompt_tokens": n}
    pos = torch.arange(n + args.max_new_tokens + 16, device="cuda")[None]
    tc, dc = DynamicCache(), DynamicCache()
    output = target(ids, position_ids=pos[:, :n], past_key_values=tc,
                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
    pending = extract_context_feature(output.hidden_states, draft.target_layer_ids)
    anchor = output.logits[:, -1:].argmax(-1)
    del output
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    raw, fused, labels, cycles, starts, anchors, prefix_hashes = [], [], [], [], [], [], []
    sequence = torch.cat([ids, anchor], dim=1)
    start = n
    captured = []
    # Capture the fusion actually used by this same draft forward, avoiding a second GEMM.
    hook = draft.hidden_norm.register_forward_hook(lambda module, inputs, out: captured.append(out[:, -1:].detach()))
    try:
        for cycle in range(args.max_cycles):
            if int(anchor) in eos or start + 16 > n + args.max_new_tokens:
                break
            if pending.shape[1] != start - dc.get_seq_length() or tc.get_seq_length() != start:
                raise RuntimeError("Pending target features or KV cache misaligned")
            before = pending[:, -1:].clone()
            block = torch.full((1, 16), draft.mask_token_id, device="cuda", dtype=torch.long)
            block[:, :1] = anchor
            captured.clear()
            hidden = draft(target_hidden=pending, noise_embedding=target.model.embed_tokens(block),
                           position_ids=pos[:, dc.get_seq_length():start+16], past_key_values=dc,
                           use_cache=True, is_causal=False)
            block[:, 1:] = target.lm_head(hidden[:, 1:]).argmax(-1)
            if len(captured) != 1 or captured[0].shape != (1, 1, draft.config.hidden_size):
                raise RuntimeError("Fusion hook did not capture the latest context vector")
            dc.crop(start)
            output = target(block, position_ids=pos[:, start:start+16], past_key_values=tc,
                            use_cache=True, output_hidden_states=True)
            posterior = output.logits.argmax(-1)
            accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(-1).sum())
            if observable_label(block, accepted, eos, n + args.max_new_tokens - start):
                raw.append(before[0].float().cpu().to(torch.float16).numpy())
                fused.append(captured[0][0].float().cpu().to(torch.float16).numpy())
                labels.append(accepted)
                cycles.append(cycle)
                starts.append(start)
                anchors.append(int(anchor))
                prefix_hashes.append(hashlib.sha256(sequence[0].cpu().numpy().astype(np.int64).tobytes()).hexdigest())
            if any(int(t) in eos for t in block[0, :accepted+1].tolist()):
                break
            committed = accepted + 1
            pending = extract_context_feature(output.hidden_states, draft.target_layer_ids)[:, :committed].contiguous()
            anchor = posterior[:, accepted:accepted+1]
            sequence = torch.cat([sequence[:, :start], block[:, :committed], anchor], dim=1)
            start += committed
            tc.crop(start)
            del output, hidden, posterior
    finally:
        hook.remove()
    if not labels:
        return None, {"skipped": "no_observable_cycles", "prompt_tokens": n}
    data = {"raw_features": np.stack(raw), "features": np.stack(fused),
            "accepted_len": np.asarray(labels, dtype=np.int64), "cycle_id": np.asarray(cycles, dtype=np.int64),
            "prefix_length": np.asarray(starts, dtype=np.int64), "anchor_id": np.asarray(anchors, dtype=np.int64),
            "prefix_sha256": np.asarray(prefix_hashes), "trajectory_token_ids": sequence[0].cpu().numpy()}
    if not np.isfinite(data["raw_features"]).all() or not np.isfinite(data["features"]).all():
        raise ValueError("Nonfinite collected features")
    return data, {"rows": len(labels), "prompt_tokens": n, "generated_tokens": start-n}


def materialize(shards, destination, raw_dim, fused_dim):
    destination.mkdir()
    n = sum(s["rows"] for s in shards)
    if not n:
        raise ValueError(f"Empty {destination.name} split")
    arrays = {}
    for name, dtype, shape in (("raw_features", np.float16, (n, 1, raw_dim)),
                               ("features", np.float16, (n, 1, fused_dim)),
                               ("mask", np.uint8, (n, 1)), ("accepted_len", np.int64, (n,)),
                               ("row_index", np.int64, (n, 5))):
        arrays[name] = np.lib.format.open_memmap(destination / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    calibration = np.zeros(n, dtype=bool)
    offset = 0
    for shard in shards:
        with np.load(shard["path"]) as data:
            count = shard["rows"]
            sl = slice(offset, offset+count)
            for name in ("raw_features", "features", "accepted_len"):
                arrays[name][sl] = data[name]
            arrays["mask"][sl] = 1
            arrays["row_index"][sl] = np.column_stack([np.zeros(count, dtype=np.int64),
                np.arange(offset, offset+count), np.full(count, shard["prompt_id"]),
                data["cycle_id"], data["accepted_len"]])
            calibration[sl] = shard["group"] == "calibration"
            offset += count
    for array in arrays.values():
        array.flush()
    if destination.name == "val":
        if not calibration.any() or calibration.all():
            raise ValueError("Missing calibration or assessment data")
        np.save(destination / "calibration.npy", calibration)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split-dir", "reference-cache-manifest", "model", "draft-model", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--persistent-dir", type=Path, required=True)
    p.add_argument("--train-prompts", type=int, default=512)
    p.add_argument("--val-prompts", type=int, default=256)
    p.add_argument("--max-cycles", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=913)
    args = p.parse_args()
    if min(args.train_prompts, args.max_cycles, args.max_prompt_tokens) < 1 or args.val_prompts < 4 or args.max_new_tokens < 16:
        raise ValueError("Invalid sampling limits")
    reference = json.loads(args.reference_cache_manifest.read_text())
    selected = select_prompts(args.manifest, args.split_dir, reference, args.train_prompts, args.val_prompts, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "shards").mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    from dflash.model import DFlashDraftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
             attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(args.draft_model, torch_dtype=torch.bfloat16,
             attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    raw_dim, fused_dim = draft.fc.in_features, draft.fc.out_features
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({"target_layer_ids": draft.target_layer_ids, "input_dim": raw_dim, "fused_dim": fused_dim,
        "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hashes": {str(path): sha256(path) for path in [args.manifest, args.reference_cache_manifest,
            args.split_dir/"train_prompt_ids.json", args.split_dir/"val_prompt_ids.json",
            args.model/"config.json", args.draft_model/"config.json"]},
        "selected_prompts": [{"prompt_id": int(r["manifest_index"]), "group": g,
                              "content_sha256": r["content_sha256"]} for g,r in selected],
        "feature_position": "latest committed token at start-1; anchor at start excluded",
        "labels": "direct greedy B16 prefix acceptance; accepted EOS and output-cap states excluded",
        "temperature": 0, "thinking": False, "parameter_models_frozen": True})
    config["model_files_sha256"] = {str(path): sha256(path) for directory in (args.model, args.draft_model)
        for path in sorted(directory.iterdir()) if path.is_file() and path.suffix in (".json", ".safetensors")}
    atomic_json(args.output_dir / "config.json", config)
    # Persist the exact fusion weights for audit/reconstruction, not for predictor training.
    torch.save({"fc": draft.fc.state_dict(), "hidden_norm": draft.hidden_norm.state_dict(),
                "rms_norm_eps": draft.config.rms_norm_eps}, args.output_dir / "fusion.pt")
    started, records = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = []
        def backup(paths):
            for path in paths:
                buffered_backup(path, args.persistent_dir / path.relative_to(args.output_dir))
        futures.append(pool.submit(backup, [args.output_dir / "config.json", args.output_dir / "fusion.pt"]))
        for i, (group, row) in enumerate(selected):
            data, info = collect_prompt(row, target, draft, tokenizer, args)
            pid = int(row["manifest_index"])
            record = {"group": group, "prompt_id": pid, **info}
            if data is not None:
                path = args.output_dir / "shards" / f"{pid}.npz"
                np.savez(path, **data)
                record.update({"path": str(path), "sha256": sha256(path)})
                futures.append(pool.submit(backup, [path]))
            records.append(record)
            progress = {"prompts_processed": i+1, "prompts_total": len(selected),
                "rows": sum(r.get("rows", 0) for r in records), "elapsed_s": time.monotonic()-started,
                "latest": record}
            atomic_json(args.output_dir / "progress.json", progress)
            # Immutable per-prompt receipts survive interruption along with each paired feature shard.
            receipt = args.output_dir / "shards" / f"{pid}.json"
            atomic_json(receipt, record)
            futures.append(pool.submit(backup, [receipt]))
            for future in futures:
                if future.done():
                    future.result()
            print(json.dumps(progress), flush=True)
        del target, draft
        torch.cuda.empty_cache()
        train_rows = materialize([r for r in records if r["group"] == "train" and "path" in r],
                                 args.output_dir / "train", raw_dim, fused_dim)
        val_rows = materialize([r for r in records if r["group"] != "train" and "path" in r],
                               args.output_dir / "val", raw_dim, fused_dim)
        manifest = {"format": "dflash_prefusion_cache_v1", "input_kind": "paired_raw_and_fused",
            "input_dim": raw_dim, "fused_dim": fused_dim, "num_slots": 15, "context_window": 1,
            "train_rows": train_rows, "val_rows": val_rows, "target_layer_ids": config["target_layer_ids"],
            "collection_config": config, "shards": records, "elapsed_s": time.monotonic()-started}
        atomic_json(args.output_dir / "manifest.json", manifest)
        futures.append(pool.submit(backup, list((args.output_dir / "train").glob("*.npy")) +
            list((args.output_dir / "val").glob("*.npy")) + [args.output_dir / "manifest.json", args.output_dir / "progress.json"]))
        for future in futures:
            future.result()
    print(json.dumps({"complete": True, "train_rows": train_rows, "val_rows": val_rows}), flush=True)


if __name__ == "__main__":
    main()
