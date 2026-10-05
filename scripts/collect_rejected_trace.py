"""Consecutive-cycle rejected-state features and fresh actual B2--B16 labels.

Previous memory is captured from the SAME B16 reference draft/verify forwards,
before cache cropping, at every cycle (including unsampled cycles). All current
inputs are snapshotted before the current draft runs. Source evaluation states
are read-only identity checks, never a source of new labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json, select_groups, select_training_groups
from scripts.gpu_runtime import configure_gpu_runtime

BLOCKS = list(range(2, 17))
WIDTH = 2560
MODEL_REVISIONS = {"target": "1cfa9a7208912126459214e8b04321603b3df60c",
                   "draft": "b74e3a329c4d963783143b1e970d95b002be72bd"}
ARRAY_SPEC = {
    "features": (np.float16, (WIDTH,)),
    "anchor_embedding": (np.float16, (WIDTH,)),
    "rejected_anchor_embedding": (np.float16, (WIDTH,)),
    "previous_meta": (np.float32, (3,)),
    "trace_draft": (np.float16, (15, WIDTH)),
    "trace_target": (np.float16, (15, WIDTH)),
    "trace_mask": (np.uint8, (15,)),
    "trace_token_ids": (np.int64, (15,)),
    "trace_offsets": (np.int16, (15,)),
    "actual": (np.int16, (15,)),
}


def prefix_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def accepted_prefix(candidate, posterior):
    return next((i for i, (x, y) in enumerate(zip(candidate, posterior)) if x != y), min(len(candidate), len(posterior)))


def empty_arrays(n):
    result = {k: np.zeros((n, *shape), dtype=dtype) for k, (dtype, shape) in ARRAY_SPEC.items()}
    result["trace_token_ids"].fill(-1)
    return result


def rejected_indices(accepted, block=16):
    """Old draft row i pairs with target row i-1; skip wrong anchor itself."""
    if not 0 <= accepted < block:
        raise ValueError("Invalid acceptance count")
    draft_rows = np.arange(accepted + 2, block, dtype=np.int64)
    return draft_rows, draft_rows - 1


def capture_memory(target, block, draft_hidden, target_hidden, accepted):
    import torch

    device = draft_hidden.device
    width = draft_hidden.shape[-1]
    memory = {
        "trace_draft": torch.zeros((15, width), dtype=torch.float16, device=device),
        "trace_target": torch.zeros((15, width), dtype=torch.float16, device=device),
        "trace_mask": torch.zeros(15, dtype=torch.uint8, device=device),
        "trace_token_ids": torch.full((15,), -1, dtype=torch.int64, device=device),
        "trace_offsets": torch.zeros(15, dtype=torch.int16, device=device),
        "rejected_anchor_embedding": torch.zeros(width, dtype=torch.float16, device=device),
        "previous_meta": torch.tensor([1., block.shape[1]/16, accepted/15], dtype=torch.float32, device=device),
    }
    draft_rows, target_rows = rejected_indices(accepted, block.shape[1])
    n = len(draft_rows)
    if n:
        di = torch.as_tensor(draft_rows, device=device)
        ti = torch.as_tensor(target_rows, device=device)
        memory["trace_draft"][:n] = draft_hidden[0, di].to(torch.float16)
        memory["trace_target"][:n] = target_hidden[0, ti].to(torch.float16)
        memory["trace_mask"][:n] = 1
        memory["trace_token_ids"][:n] = block[0, di]
        memory["trace_offsets"][:n] = torch.arange(1, n+1, device=device, dtype=torch.int16)
    if accepted < block.shape[1]-1:
        memory["rejected_anchor_embedding"] = target.model.embed_tokens(block[:, accepted+1])[0].to(torch.float16).clone()
    return memory


def collect_prompt(args, row, target, draft, tokenizer, wanted=None, *, cache_factory=None, extract=None):
    import torch

    if cache_factory is None:
        from transformers import DynamicCache
        cache_factory = DynamicCache
    if extract is None:
        from dflash.model import extract_context_feature
        extract = extract_context_feature
    device = target.device
    pid, group = int(row["manifest_index"]), row["group"]
    with torch.inference_mode():
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt").to(device)
        prompt_n = ids.shape[1]
        if prompt_n > args.max_prompt_tokens:
            return [], empty_arrays(0), {"prompt_id": pid, "skipped": "prompt_length", "tokens": prompt_n,
                                         "missing_source_cycles": sorted(map(int, wanted or {}))}
        positions = torch.arange(prompt_n+args.max_new_tokens+16, device=device)[None]
        tc, dc = cache_factory(), cache_factory()
        pre = target(ids, position_ids=positions[:, :prompt_n], past_key_values=tc,
                     use_cache=True, output_hidden_states=True, logits_to_keep=1)
        hidden = extract(pre.hidden_states, draft.target_layer_ids)
        sequence = torch.cat([ids, pre.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        del pre
        eos = target.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        offsets = np.linspace(0, max(0, args.max_new_tokens-32), args.states_per_prompt, dtype=int)
        next_offset, start, cycle = 0, prompt_n, 0
        previous, memory = None, None
        records, arrays, timings, seen = [], [], [], set()
        rng = random.Random(args.seed+pid)

        def propose(b, cache, pending):
            block = torch.full((1, b), draft.mask_token_id, dtype=torch.long, device=device)
            block[:, 0] = sequence[:, start]
            h = draft(target_hidden=pending, noise_embedding=target.model.embed_tokens(block),
                      position_ids=positions[:, cache.get_seq_length():start+b],
                      past_key_values=cache, use_cache=True, is_causal=False)
            block[:, 1:] = target.lm_head(h[:, 1:]).argmax(-1)
            cache.crop(start)
            return block, h

        def verify(block, cache, full=False):
            out = target(block, position_ids=positions[:, start:start+block.shape[1]],
                         past_key_values=cache, use_cache=True, output_hidden_states=full)
            post = out.logits.argmax(-1)
            a = int((block[:, 1:] == post[:, :-1]).cumprod(-1).sum())
            return out, post, a

        while start < prompt_n+args.max_new_tokens and int(sequence[0, start]) not in eos:
            enough_slots = start-prompt_n <= args.max_new_tokens-16
            selected = enough_slots and ((cycle in wanted) if wanted is not None else
                (next_offset < len(offsets) and start-prompt_n >= offsets[next_offset]))
            pending = hidden
            if selected:
                # Detach the prior-cycle feature record BEFORE any current proposal.
                current = empty_arrays(1)
                current["features"][0] = draft.hidden_norm(draft.fc(pending[:, -1:]))[0, 0].to(torch.float16).cpu().numpy()
                current["anchor_embedding"][0] = target.model.embed_tokens(sequence[:, start])[0].to(torch.float16).cpu().numpy()
                if memory is not None:
                    for key, value in memory.items():
                        current[key][0] = value.cpu().numpy()
                tc_snapshot, dc_snapshot = copy.deepcopy(tc), copy.deepcopy(dc)
                source = wanted.get(cycle) if wanted is not None else None
                prefix = sequence[0, :start+1].tolist()
                record = {"prompt_id": str(pid), "group": group, "source": row.get("source"),
                    "cycle": cycle, "prefix_length": start, "prefix_token_ids": prefix,
                    "prefix_sha256": prefix_hash(prefix), "generated_before_anchor": start-prompt_n,
                    "has_previous": previous is not None, "previous_cycle": cycle-1 if previous else -1,
                    "previous_B": 16 if previous else 0, "previous_A": previous["accepted"] if previous else -1,
                    "previous": previous, "input_capture": "before_current_draft",
                    "reference_prefix_match": source is None or prefix == source["prefix_token_ids"],
                    "source_eligible": source is None or source["eligible"],
                    "source_prefix_sha256": source["prefix_sha256"] if source else None}
            baseline, dh = propose(16, dc, pending)
            output, posterior, accepted = verify(baseline, tc, True)
            if selected:
                outcomes = {"16": {"accepted": accepted, "draft_ids": baseline[0, 1:].tolist()}}
                order = list(range(2, 16))
                rng.shuffle(order)
                for b in order:
                    block, _ = propose(b, copy.deepcopy(dc_snapshot), pending)
                    alt, _, a = verify(block, copy.deepcopy(tc_snapshot))
                    outcomes[str(b)] = {"accepted": a, "draft_ids": block[0, 1:].tolist()}
                    del alt
                # B4/B8/B12 unchanged-candidate controls diagnose verifier-width differences.
                for b in (4, 8, 12):
                    alt, _, a = verify(baseline[:, :b].contiguous(), copy.deepcopy(tc_snapshot))
                    outcomes[str(b)]["truncated_accepted"] = a
                    del alt
                reverse = bool(row.get("audit_prompt")) and len(records) == 0
                if reverse:
                    for b in reversed(BLOCKS):
                        block, _ = propose(b, copy.deepcopy(dc_snapshot), pending)
                        alt, _, a = verify(block, copy.deepcopy(tc_snapshot))
                        if block[0, 1:].tolist() != outcomes[str(b)]["draft_ids"] or a != outcomes[str(b)]["accepted"]:
                            raise ValueError("Alternative cache/order contamination")
                        del alt
                canonical_checked = disagreements = 0
                if bool(row.get("audit_prompt")) and len(records) < 2:
                    canonical_cache = copy.deepcopy(tc_snapshot)
                    token = sequence[:, start:start+1]
                    canonical = []
                    for j in range(15):
                        one = target(token, position_ids=positions[:, start+j:start+j+1], past_key_values=canonical_cache,
                                     use_cache=True, logits_to_keep=1)
                        token = one.logits[:, -1].argmax(-1, keepdim=True)
                        canonical.append(int(token))
                        if int(token) in eos:
                            break
                    for b in BLOCKS:
                        candidate = outcomes[str(b)]["draft_ids"]
                        observed = min(len(candidate), len(canonical))
                        disagreements += accepted_prefix(candidate, canonical) != min(outcomes[str(b)]["accepted"], observed)
                        canonical_checked += 1
                    del one, canonical_cache
                terminal = any(any(t in eos for t in o["draft_ids"][:o["accepted"]]) for o in outcomes.values())
                exclusions = (["accepted_eos"] if terminal else [])
                if not record["reference_prefix_match"]:
                    exclusions.append("reference_prefix_drift")
                if not record["source_eligible"]:
                    exclusions.append("source_excluded")
                record.update(outcomes=outcomes, eligible=not exclusions, exclusion=",".join(exclusions) or None,
                              canonical_checked=canonical_checked, canonical_disagreements=disagreements,
                              reverse_order_checked=reverse, truncated_verify_checks=3)
                current["actual"][0] = [outcomes[str(b)]["accepted"] for b in BLOCKS]
                records.append(record)
                arrays.append(current)
                seen.add(cycle)
                next_offset += 1
                del tc_snapshot, dc_snapshot
            # Preserve just the previous B16's aligned rejected rows, not its live KV.
            if str(device).startswith("cuda"):
                t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0.record()
            memory = capture_memory(target, baseline, dh, output.hidden_states[-1], accepted)
            if str(device).startswith("cuda"):
                t1.record()
                timings.append((t0, t1))
            old_prefix = sequence[0, :start+1].tolist()
            previous = {"cycle": cycle, "block_size": 16, "prefix_length": start,
                        "prefix_token_ids": old_prefix, "prefix_sha256": prefix_hash(old_prefix),
                        "draft_ids": baseline[0, 1:].tolist(), "accepted": accepted,
                        "posterior_ids": posterior[0].tolist(), "correction_token_id": int(posterior[0, accepted]),
                        "trace_capture": "same_forward", "terminal": any(int(t) in eos for t in baseline[0, :accepted+1])}
            committed = accepted+1
            sequence = torch.cat([sequence[:, :start], baseline[:, :committed], posterior[:, accepted:accepted+1]], dim=1)
            hidden = extract(output.hidden_states, draft.target_layer_ids)[:, :committed].contiguous()
            start += committed
            tc.crop(start)
            cycle += 1
            del output, posterior, dh
            if any(t in eos for t in sequence[0, prompt_n:].tolist()):
                break
            if wanted is not None and seen == set(wanted):
                break
            if wanted is None and next_offset == len(offsets):
                break
        if timings:
            torch.cuda.synchronize()
        capture_ms = [a.elapsed_time(b) for a, b in timings]
        summary = {"prompt_id": pid, "cycles": cycle, "states": len(records), "prompt_tokens": prompt_n,
                   "generated_tokens": min(sequence.shape[1]-prompt_n, args.max_new_tokens),
                   "missing_source_cycles": sorted(set(wanted or {})-seen),
                   "trace_capture_cuda_ms": capture_ms,
                   "trace_timing_scope": "CUDA stream elapsed allocation/gather/copy for prior memory; excludes dataset D2H, JSON/proof recording, and prediction; not serving latency"}
        return records, ({k: np.concatenate([a[k] for a in arrays]) for k in ARRAY_SPEC} if arrays else empty_arrays(0)), summary


def backup_file(path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix+".tmp")
    shutil.copyfile(path, temporary)
    if sha256(temporary) != sha256(path):
        raise ValueError("Backup checksum failed")
    temporary.replace(dest)


def load_reference(root):
    complete = json.loads((root/"COMPLETE.json").read_text())
    if not complete["sample_complete"]:
        raise ValueError("Incomplete evaluation source")
    for name, expected in complete["binding"].items():
        if sha256(root/name) != expected:
            raise ValueError("Evaluation source completion mismatch")
    result = {}
    for receipt in json.loads((root/"receipts.json").read_text()):
        for name, expected in receipt["files"].items():
            if Path(name).name != name or sha256(root/name) != expected:
                raise ValueError("Evaluation source shard mismatch")
        pid = int(receipt["prompt_id"])
        states = json.loads((root/f"prompt_{pid}.json").read_text())["states"]
        if pid in result or len({r["cycle"] for r in states}) != len(states):
            raise ValueError("Duplicate evaluation source prompt or cycle")
        if any(prefix_hash(r["prefix_token_ids"]) != r["prefix_sha256"] for r in states):
            raise ValueError("Evaluation source prefix proof mismatch")
        result[pid] = {r["cycle"]: r for r in states}
    return result, sha256(root/"COMPLETE.json")


def validate_models(models):
    for name, revision in MODEL_REVISIONS.items():
        item = models[name]
        if not isinstance(item, dict) or item.get("revision") != revision:
            raise ValueError(f"Unexpected {name} model revision")
        if Path(item["path"]).name != revision:
            raise ValueError(f"{name} path is not the pinned snapshot")


_WORKER = None


def initialize_worker(options, models):
    global _WORKER
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel

    if transformers.__version__ != "4.57.1":
        raise ValueError("Require Transformers 4.57.1 for pinned collector")
    torch.set_num_threads(2)
    torch.manual_seed(options["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def model_path(name):
        item = models[name]
        return item["path"] if isinstance(item, dict) else item
    tokenizer = AutoTokenizer.from_pretrained(model_path("target"), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(model_path("target"), torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True,
                device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(model_path("draft"), torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True,
                device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval().requires_grad_(False)
    _WORKER = (argparse.Namespace(**options), target, draft, tokenizer)


def run_worker(task):
    args, target, draft, tokenizer = _WORKER
    row, wanted = task
    return collect_prompt(args, row, target, draft, tokenizer, wanted)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pilot-manifest", "split-dir", "eval-cache", "models", "output", "backup"):
        p.add_argument("--"+name, type=Path, required=True)
    gpu_args = p.add_mutually_exclusive_group(required=True)
    gpu_args.add_argument("--gpu", type=int, help="Physical GPU index outside Slurm only")
    gpu_args.add_argument("--use-visible-gpu", action="store_true",
                          help="Use the single Slurm-assigned GPU without changing CUDA_VISIBLE_DEVICES")
    gpu_args.add_argument("--use-container-gpu", action="store_true",
                          help="Use the single Kubernetes-allocated GPU UUID after validating container isolation")
    p.add_argument("--workers", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--training-rows", type=int, default=2000)
    p.add_argument("--limit-training-prompts", type=int, default=600)
    p.add_argument("--states-per-prompt", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Read-only input/model/split/source checks; no GPU or output writes")
    args = p.parse_args()
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Use fresh independent temporary and durable destinations")
    if min(args.training_rows, args.limit_training_prompts, args.states_per_prompt, args.max_seconds) < 1:
        raise ValueError("Invalid collection limits")
    if not args.smoke and (args.training_rows != 2000 or args.states_per_prompt != 8 or args.max_new_tokens != 256 or args.max_prompt_tokens != 2048):
        raise ValueError("Non-smoke run must preserve the 2k / 8-state / 256-output / 2048-prompt protocol")
    selected_train = select_training_groups(args.manifest, args.split_dir, args.seed, args.limit_training_prompts)
    selected_eval = select_groups(args.manifest, args.pilot_manifest, args.split_dir, 928)
    frozen = {g: [int(r["manifest_index"]) for r in selected_eval if r["group"] == g] for g in ("calibration", "assessment")}
    if args.smoke:
        selected_eval = [r for g in frozen for r in [r for r in selected_eval if r["group"] == g][:2]]
    references, reference_binding = load_reference(args.eval_cache)
    selected = selected_train+selected_eval
    for group in ("train", "calibration", "assessment"):
        first = [r for r in selected if r["group"] == group][:2]
        for row in first:
            row["audit_prompt"] = True
    models = json.loads(args.models.read_text())
    validate_models(models)
    canonical = {g: list(map(int, json.loads((args.split_dir/f"{g}_prompt_ids.json").read_text())[f"{g}_prompt_ids"])) for g in ("train", "val")}
    for row in selected_eval:
        pid = int(row["manifest_index"])
        if pid not in references or any(s["group"] != row["group"] for s in references[pid].values()):
            raise ValueError("Evaluation source missing prompt or changed its group")
    if args.preflight:
        print(json.dumps({"preflight_passed": True, "gpu_accessed": False,
            "selected_training_prompts": len(selected_train),
            "evaluation": {g: {"prompts": sum(r["group"] == g for r in selected_eval),
                "source_rows": sum(len(references[int(r["manifest_index"])]) for r in selected_eval if r["group"] == g),
                "source_eligible_rows": sum(sum(s["eligible"] for s in references[int(r["manifest_index"])].values())
                                            for r in selected_eval if r["group"] == g)} for g in frozen},
            "model_revisions": MODEL_REVISIONS, "reference_completion_sha256": reference_binding}, indent=2))
        return
    # Parent-only idle check happens before loading any worker's models.
    gpu_runtime = configure_gpu_runtime(args.gpu, args.use_visible_gpu,
                                        use_container_gpu=args.use_container_gpu, require_gpu=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config = {**options, "schema_version": "dflash_rejected_trace_v1", "blocks": BLOCKS,
              "gpu_runtime": gpu_runtime,
              "models": models, "prompt_ids": [int(r["manifest_index"]) for r in selected],
              "prompt_groups": {str(r["manifest_index"]): r["group"] for r in selected},
              "prompt_content_hashes": {str(r["manifest_index"]): r["content_sha256"] for r in selected},
              "canonical_train_prompt_ids": canonical["train"], "canonical_val_prompt_ids": canonical["val"],
              "frozen_calibration_prompt_ids": frozen["calibration"], "frozen_assessment_prompt_ids": frozen["assessment"],
              "reference_completion_sha256": reference_binding,
              "input_hashes": {str(path): sha256(path) for path in (args.manifest, args.pilot_manifest, args.models,
                              args.split_dir/"train_prompt_ids.json", args.split_dir/"val_prompt_ids.json")},
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": sha256(Path(__file__)), "dtype": "bfloat16", "attention": "sdpa", "thinking": False,
              "dependency_sha256": {name: sha256(Path(__file__).resolve().parents[1]/name) for name in
                  ("dflash/model.py", "scripts/collect_policy_granularity.py", "scripts/audit_block_headroom.py", "scripts/gpu_runtime.py")},
              "temperature": 0, "tf32": False, "training_selection": "first 2000 eligible rows in immutable receipt order; finish bounded in-flight prompts",
              "trace": "same-forward B16 previous cycle; STRICTLY AFTER first rejected draft token; target row i-1 paired with draft row i; packed offsets1..count",
              "previous_meta": "has_previous, previous_B/16, previous_A/15; all zeros at cycle0"}
    atomic_json(args.output/"config.json", config)
    backup_file(args.output/"config.json", args.backup/"config.json")
    receipts, totals, capture_times = [], Counter(), []
    missing, exclusions, disagreements = [], Counter(), 0
    evaluation_coverage = {g: {"source_rows": sum(len(references[int(r["manifest_index"])]) for r in selected_eval if r["group"] == g),
                            "source_eligible_rows": sum(sum(s["eligible"] for s in references[int(r["manifest_index"])].values())
                                                        for r in selected_eval if r["group"] == g),
                            "replayed_rows": 0, "eligible_rows": 0, "prefix_drift_rows": 0}
                           for g in ("calibration", "assessment")}
    started = time.monotonic()
    failure = None
    def commit(row, result):
        nonlocal disagreements
        records, arrays, prompt_summary = result
        pid = int(row["manifest_index"])
        for name in ARRAY_SPEC:
            if len(arrays[name]) != len(records):
                raise ValueError("Unaligned collected arrays")
        json_name, npz_name = f"prompt_{pid}.json", f"prompt_{pid}.npz"
        atomic_json(args.output/json_name, {"prompt_id": pid, "group": row["group"], "states": records, "prompt_summary": prompt_summary})
        temporary = args.output/(npz_name+".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(args.output/npz_name)
        receipt = {"prompt_id": pid, "group": row["group"], "source": row.get("source"), "states": len(records),
                   "eligible_states": sum(r["eligible"] for r in records),
                   "files": {name: sha256(args.output/name) for name in (json_name, npz_name)}}
        # Shards land durably before their receipt. An interrupted backup cannot bless an incomplete shard.
        for name in receipt["files"]:
            backup_file(args.output/name, args.backup/name)
        atomic_json(args.output/f"receipt_{pid}.json", receipt)
        backup_file(args.output/f"receipt_{pid}.json", args.backup/f"receipt_{pid}.json")
        receipts.append(receipt)
        atomic_json(args.output/"receipts.json", receipts)
        backup_file(args.output/"receipts.json", args.backup/"receipts.json")
        totals[row["group"]] += receipt["eligible_states"]
        if row["group"] != "train":
            coverage = evaluation_coverage[row["group"]]
            coverage["replayed_rows"] += len(records)
            coverage["eligible_rows"] += receipt["eligible_states"]
            coverage["prefix_drift_rows"] += sum(not r["reference_prefix_match"] for r in records)
        disagreements += sum(r["canonical_disagreements"] for r in records)
        exclusions.update(r["exclusion"] for r in records if not r["eligible"])
        missing.extend({"prompt_id": pid, "cycle": c} for c in prompt_summary["missing_source_cycles"])
        capture_times.extend(prompt_summary["trace_capture_cuda_ms"] if "trace_capture_cuda_ms" in prompt_summary else [])
        progress = {"processed_prompts": len(receipts), "eligible": dict(totals), "elapsed_seconds": time.monotonic()-started,
                    "last_prompt_id": pid, "canonical_disagreements": disagreements, "missing_evaluation_states": len(missing)}
        atomic_json(args.output/"progress.json", progress)
        backup_file(args.output/"progress.json", args.backup/"progress.json")
        print("PROGRESS", json.dumps(progress), flush=True)
        if disagreements:
            raise ValueError("Canonical greedy disagreement: preserve evidence, stop before training")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(options, models)) as pool:
            for phase, planned in (("train", selected_train), ("evaluation", selected_eval)):
                cursor, pending = 0, []
                while cursor < len(planned) or pending:
                    stop = time.monotonic()-started >= args.max_seconds or (phase == "train" and totals["train"] >= args.training_rows)
                    while not stop and cursor < len(planned) and len(pending) < args.workers:
                        row = planned[cursor]
                        wanted = references[int(row["manifest_index"])] if phase == "evaluation" else None
                        if args.smoke and wanted is not None:
                            wanted = {k:v for k,v in wanted.items() if v["generated_before_anchor"] <= args.max_new_tokens-16}
                        pending.append((row, pool.submit(run_worker, (row, wanted))))
                        cursor += 1
                    if not pending:
                        break
                    row, future = pending.pop(0)
                    commit(row, future.result())
                    if stop and not pending:
                        break
                if time.monotonic()-started >= args.max_seconds:
                    break
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        seen_eval = {r["prompt_id"] for r in receipts if r["group"] != "train"}
        complete = (failure is None and totals["train"] >= args.training_rows and
                    seen_eval == {int(r["manifest_index"]) for r in selected_eval} and not disagreements)
        summary = {"sample_complete": complete, "states": sum(r["states"] for r in receipts),
                   "eligible_states": sum(totals.values()), "group_eligible": dict(totals), "prompts": len(receipts),
                   "canonical_disagreements": disagreements, "exclusions": dict(exclusions),
                   "evaluation_coverage": evaluation_coverage,
                   "missing_evaluation_states": missing, "failure": failure, "elapsed_seconds": time.monotonic()-started,
                   "trace_capture_cuda_ms": {"count": len(capture_times), "mean": float(np.mean(capture_times)) if capture_times else None,
                                             "p50": float(np.median(capture_times)) if capture_times else None},
                   "reference_data_unchanged": sha256(args.eval_cache/"COMPLETE.json") == reference_binding}
        atomic_json(args.output/"collection_summary.json", summary)
        backup_file(args.output/"collection_summary.json", args.backup/"collection_summary.json")
        if (args.output/"receipts.json").exists():
            atomic_json(args.output/"COMPLETE.json", {"sample_complete": complete, "states": summary["states"],
                         "eligible_states": summary["eligible_states"], "binding": {name: sha256(args.output/name)
                         for name in ("config.json", "receipts.json", "collection_summary.json")}})
            backup_file(args.output/"COMPLETE.json", args.backup/"COMPLETE.json")
    if not complete:
        raise RuntimeError("Incomplete collection: training is blocked")
    from scripts.audit_rejected_trace_cache import load_cache
    _, _, _, audit = load_cache(args.backup)
    atomic_json(args.backup/"audit.json", audit)
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
