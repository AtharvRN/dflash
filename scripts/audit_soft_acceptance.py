"""Training-only, exact-prefix B16 replay audit; no predictor fitting or policy tuning.

Fresh greedy proposals, target labels, TV overlap and margins share one replay.
Temperature-one TV is stochastic distribution overlap, NOT greedy acceptance.
Rejected suffix labels are off-path diagnostics, not on-path observations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256
from scripts.collect_midverify_probe import pack_rows, backup_file
from scripts.collect_policy_granularity import atomic_json
from scripts.collect_rejected_trace import prefix_hash, validate_models
from scripts.gpu_runtime import configure_gpu_runtime


def select_training_rows(root, count=200, seed=1005):
    """Uniform training-only sample from the first 2000 receipt-ordered eligible rows."""
    complete = json.loads((root / "COMPLETE.json").read_text())
    if not complete.get("sample_complete"):
        raise ValueError("Source must be complete")
    if set(complete["binding"]) != {"config.json", "receipts.json", "collection_summary.json"}:
        raise ValueError("Unexpected completion bindings")
    for name, digest in complete["binding"].items():
        if sha256(root / name) != digest:
            raise ValueError("Source binding mismatch")
    config = json.loads((root / "config.json").read_text())
    validate_models(config["models"])
    train = set(map(int, config["canonical_train_prompt_ids"]))
    val = set(map(int, config["canonical_val_prompt_ids"]))
    if train & val:
        raise ValueError("Source split leakage")
    rows, seen, bindings = [], set(), {}
    for receipt in json.loads((root / "receipts.json").read_text()):
        if receipt["group"] != "train":
            continue
        pid = int(receipt["prompt_id"])
        name = f"prompt_{pid}.json"
        if pid not in train or pid in val or sha256(root / name) != receipt["files"][name]:
            raise ValueError("Training provenance mismatch")
        bindings[name] = receipt["files"][name]
        for row in json.loads((root / name).read_text())["states"]:
            if row["group"] != "train" or int(row["prompt_id"]) != pid:
                raise ValueError("Changed training group")
            key = (pid, row["cycle"])
            if key in seen or prefix_hash(row["prefix_token_ids"]) != row["prefix_sha256"]:
                raise ValueError("Duplicate state or invalid prefix hash")
            seen.add(key)
            if row["eligible"]:
                rows.append({k: row[k] for k in ("prompt_id", "cycle", "group", "source", "prefix_length",
                            "prefix_token_ids", "prefix_sha256")} | {
                            "draft_ids": row["outcomes"]["16"]["draft_ids"],
                            "old_accepted_len": row["outcomes"]["16"]["accepted"]})
    rows = rows[:2000]
    if not 1 <= count <= len(rows):
        raise ValueError("Insufficient audited training rows")
    indices = np.sort(np.random.default_rng(seed).choice(len(rows), count, replace=False))
    selected = [dict(rows[i], row=j, source_training_index=int(i)) for j, i in enumerate(indices)]
    return selected, {"completion_sha256": sha256(root / "COMPLETE.json"),
                      "source_pool_rows": len(rows), "json_sha256": bindings}


def label_statistics(draft_logits, target_logits, candidates):
    """Compute labels without assuming that distribution overlap implies argmax match."""
    import torch
    if draft_logits.shape != target_logits.shape or draft_logits.shape[:-1] != candidates.shape:
        raise ValueError("Logit/candidate shape mismatch")
    qlog, plog = draft_logits.float().log_softmax(-1), target_logits.float().log_softmax(-1)
    q, p = qlog.exp(), plog.exp()
    if not torch.isfinite(q).all() or not torch.isfinite(p).all():
        raise ValueError("Nonfinite distribution")
    overlap = torch.minimum(q, p).sum(-1)
    tv_overlap = 1 - .5 * (q-p).abs().sum(-1)
    if not torch.allclose(overlap, tv_overlap, atol=2e-6, rtol=2e-5):
        raise ValueError("TV identity failure")
    picked = target_logits.float().gather(-1, candidates[..., None]).squeeze(-1)
    top, top_ids = target_logits.float().topk(2, dim=-1)
    competitor = torch.where(top_ids[..., 0] == candidates, top[..., 1], top[..., 0])
    margin = picked - competitor
    matches = target_logits.argmax(-1) == candidates
    ties = margin == 0
    if not torch.equal(matches[~ties], (margin > 0)[~ties]):
        raise ValueError("Margin sign/greedy label mismatch")
    survival = matches.long().cumprod(-1)
    at_risk = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), dim=-1).bool()
    qpicked = qlog.gather(-1, candidates[..., None]).squeeze(-1)
    draft_picked = draft_logits.float().gather(-1, candidates[..., None]).squeeze(-1)
    return {"tv_overlap": overlap, "target_candidate_prob": p.gather(-1, candidates[..., None]).squeeze(-1),
            "target_margin": margin, "target_argmax": target_logits.argmax(-1), "matches": matches,
            "survival": survival, "at_risk": at_risk, "ties": ties,
            "accepted_len": survival.sum(-1),
            "draft_stats": torch.stack((qpicked, -(q*qlog).sum(-1),
                                         draft_picked-draft_logits.float().amax(-1)), dim=-1)}


def replay_batch(rows, target, draft, *, device="cuda", canonical_check=False):
    import torch
    from transformers import DynamicCache
    ids, mask, positions, saved_block, lengths = [torch.as_tensor(x, device=device) for x in pack_rows(rows)]
    early, hooks = {}, []
    for idx in draft.target_layer_ids:
        hooks.append(target.model.layers[idx].register_forward_hook(
            lambda module, inputs, output, idx=idx: early.__setitem__(idx, output.detach())))
    cache = DynamicCache()
    try:
        pre = target(ids, attention_mask=mask, position_ids=positions, past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    anchor_match = pre.logits[:, -1].argmax(-1) == saved_block[:, 0]
    raw = torch.cat([early[i] for i in draft.target_layer_ids], dim=-1)
    early.clear()
    fused = draft.hidden_norm(draft.fc(raw[:, -1:]))[:, 0]
    del pre
    query_positions = lengths[:, None] + torch.arange(16, device=device)[None]
    draft_mask = torch.cat((mask, torch.ones_like(saved_block)), dim=1)
    additive = torch.zeros((len(rows), 1, 1, draft_mask.shape[1]), dtype=torch.bfloat16, device=device)
    additive.masked_fill_(~draft_mask[:, None, None].bool(), torch.finfo(torch.bfloat16).min)
    noise = torch.full_like(saved_block, draft.mask_token_id)
    noise[:, 0] = saved_block[:, 0]
    hidden = draft(target_hidden=raw, noise_embedding=target.model.embed_tokens(noise),
                   position_ids=torch.cat((positions, query_positions), dim=1),
                   attention_mask=additive, use_cache=False, is_causal=False)
    del raw, additive
    draft_logits = target.lm_head(hidden[:, 1:]).float()
    fresh = saved_block.clone()
    fresh[:, 1:] = draft_logits.argmax(-1)
    verified = target(fresh, attention_mask=draft_mask, position_ids=query_positions,
                      past_key_values=cache, use_cache=True)
    # Verify row 0 (anchor input) predicts proposed token 1; never compare row j with input j.
    result = label_statistics(draft_logits, verified.logits[:, :15], fresh[:, 1:])
    eos = target.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    accepted_eos = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for token in eos:
        accepted_eos |= ((fresh[:, 1:] == token) & result["survival"].bool()).any(-1)
    result.update(candidate_ids=fresh[:, 1:], saved_candidate_ids=saved_block[:, 1:],
                  candidate_match=fresh[:, 1:] == saved_block[:, 1:], anchor_match=anchor_match,
                  accepted_eos=accepted_eos, fused=fused.to(torch.float16),
                  draft_hidden=hidden[:, 1:].to(torch.float16),
                  candidate_vectors=target.lm_head.weight[fresh[:, 1:]].to(torch.float16))
    result["canonical_accepted_len"] = torch.full((len(rows),), -1, device=device, dtype=torch.long)
    if canonical_check:
        # Reuse only committed-prefix KV, then run canonical one-token target decoding.
        cache.crop(ids.shape[1])
        token, canonical = fresh[:, :1], []
        for j in range(15):
            one = target(token, attention_mask=draft_mask[:, :ids.shape[1]+j+1],
                         position_ids=query_positions[:, j:j+1], past_key_values=cache,
                         use_cache=True, logits_to_keep=1)
            token = one.logits[:, -1].argmax(-1, keepdim=True)
            canonical.append(token)
        result["canonical_accepted_len"] = (torch.cat(canonical, -1) == fresh[:, 1:]).long().cumprod(-1).sum(-1)
    return {k: v.detach().cpu().numpy() for k, v in result.items()}


_MODELS = None


def initialize(models):
    global _MODELS
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel
    if transformers.__version__ != "4.57.1" or torch.__version__ != "2.13.0+cu130":
        raise ValueError("Pinned Torch/Transformers required")
    torch.set_num_threads(2)
    torch.manual_seed(1005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                  device_map={"": "cuda:0"}, low_cpu_mem_usage=True)
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], **kwargs).eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], **kwargs).eval().requires_grad_(False)
    _MODELS = target, draft


def work(batch):
    import torch
    with torch.inference_mode():
        return replay_batch(batch, *_MODELS, canonical_check=batch[0]["row"] == 0 or batch[0].get("canonical_check", False))


def summarize(arrays, rows):
    result = {"rows": len(rows), "prompts": len({r["prompt_id"] for r in rows}),
              "anchor_matches": int(arrays["anchor_match"].sum()),
              "accepted_eos_rows": int(arrays["accepted_eos"].sum()),
              "all_candidate_ids_match_rows": int(arrays["candidate_match"].all(-1).sum()),
              "acceptance_matches_source_rows": int((arrays["accepted_len"] ==
                    np.array([r["old_accepted_len"] for r in rows])).sum()),
              "mean_fresh_accepted": float(arrays["accepted_len"].mean())}
    if "canonical_accepted_len" in arrays:
        checked = arrays["canonical_accepted_len"] >= 0
        result["canonical_checked_rows"] = int(checked.sum())
        result["canonical_disagreements"] = int((arrays["canonical_accepted_len"][checked] != arrays["accepted_len"][checked]).sum())
    for name, mask in (("at_risk", arrays["at_risk"]), ("off_path_suffix", ~arrays["at_risk"])):
        hard = arrays["matches"][mask]
        result[name] = {"positions": int(mask.sum()), "greedy_match_rate": float(hard.mean()) if hard.size else None,
                        "ties": int(arrays["ties"][mask].sum())}
        for field in ("tv_overlap", "target_candidate_prob", "target_margin"):
            values = arrays[field][mask]
            result[name][field] = {"mean": float(values.mean()) if values.size else None,
                "accepted_mean": float(values[hard].mean()) if hard.any() else None,
                "rejected_mean": float(values[~hard].mean()) if (~hard).any() else None}
        result[name]["tv_above_0_9_but_greedy_rejected"] = int(((arrays["tv_overlap"][mask] > .9) & ~hard).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "models", "output", "backup"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Fresh independent destinations required")
    if args.batch_size < 1 or not 1 <= args.rows <= 200:
        raise ValueError("Audit is bounded to 200 training states")
    rows, binding = select_training_rows(args.source, args.rows)
    models = json.loads(args.models.read_text())
    validate_models(models)
    runtime = configure_gpu_runtime(use_container_gpu=True, require_gpu=True)
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    config = {"schema": "soft_acceptance_audit_v1", "rows": rows, "source_binding": binding,
              "models": models, "runtime": runtime, "workers": args.workers, "batch_size": args.batch_size,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": sha256(Path(__file__)), "temperature_for_soft_labels": 1., "decoding": "greedy",
              "selection": "seed1005 uniform sample from first2000 eligible training rows; no validation access",
              "protocol": "exact saved prefix; fresh B16 draft and target forward; off-path suffix explicit; no training"}
    atomic_json(args.output / "config.json", config)
    backup_file(args.output / "config.json", args.backup / "config.json")
    started, chunks, receipts = time.monotonic(), [], []
    batches = [rows[i:i+args.batch_size] for i in range(0, len(rows), args.batch_size)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(models,)) as pool:
        for i, arrays in enumerate(pool.map(work, batches)):
            path = args.output / f"batch_{i:04d}.npz"
            np.savez(path, **arrays)
            backup_file(path, args.backup / path.name)
            receipts.append({"file": path.name, "sha256": sha256(path), "rows": [r["row"] for r in batches[i]]})
            chunks.append(arrays)
            atomic_json(args.output / "receipts.json", receipts)
            backup_file(args.output / "receipts.json", args.backup / "receipts.json")
            print("PROGRESS", json.dumps({"rows": sum(len(r["rows"]) for r in receipts),
                  "elapsed_seconds": time.monotonic()-started}), flush=True)
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    summary = summarize(arrays, rows)
    summary.update(elapsed_seconds=time.monotonic()-started, trained_models=0,
                   limitations=["Training-only diagnostic, not held-out policy evidence", "TV labels refer to temperature1 distributions",
                                "Old outcomes are replay controls only; all new labels paired with fresh candidates",
                                "Full-prefix replay can differ from incremental cached execution"])
    atomic_json(args.output / "summary.json", summary)
    backup_file(args.output / "summary.json", args.backup / "summary.json")
    if summary.get("canonical_disagreements", 0):
        raise ValueError("Canonical target check failed; summary preserved, no completion marker")
    atomic_json(args.output / "COMPLETE.json", {"complete": True, "binding": {
        name: sha256(args.output / name) for name in ("config.json", "receipts.json", "summary.json")}})
    backup_file(args.output / "COMPLETE.json", args.backup / "COMPLETE.json")
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
