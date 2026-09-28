"""Capture causal verifier distributions at saved prefixes, with fresh B16 labels.

This is inference-only diagnostic replay, not a resumed generation trajectory.
Original labels are retained only for replay comparison, never used as new labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_acceptance_geometry import digest


def history_positions(start, prompt_tokens, window=16):
    positions = np.arange(max(0, start - window), start)
    # Original collection computed just the last prompt logit, then all verify logits.
    return positions, positions >= prompt_tokens - 1


def confidence(logits, ids):
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    p = lp.exp()
    top = p.topk(2, -1).values
    chosen = lp.gather(-1, ids[..., None]).squeeze(-1)
    result = torch.stack((-(p * lp).sum(-1), chosen.exp(),
                          top[..., 0] - top[..., 1], chosen), -1)
    return result.cpu().numpy(), p


def select_states(cache, analysis, limit):
    manifest = json.loads((cache / "manifest.json").read_text())
    old_summary = json.loads((analysis / "summary.json").read_text())
    if digest(cache / "manifest.json") != old_summary["audit"]["cache_manifest_sha256"]:
        raise ValueError("The source cache does not match the plotted cache")
    selected = np.load(analysis / "plot_coordinates.npz", allow_pickle=False)
    take = np.arange(len(selected["rows"]))
    if limit and limit < len(take):
        take = np.unique(np.linspace(0, len(take) - 1, limit, dtype=int))
    index = np.concatenate([np.load(cache / g / "row_index.npy") for g in ("train", "val")])
    shards = {s["prompt_id"]: s for s in manifest["shards"] if s.get("rows")}
    loaded, rows = {}, []
    for i in take:
        original_row, prompt = int(selected["rows"][i]), int(selected["prompt_id"][i])
        shard = shards[prompt]
        if prompt not in loaded:
            path = cache / "shards" / Path(shard["path"]).name
            if digest(path) != shard["sha256"]:
                raise ValueError("Shard hash mismatch")
            with np.load(path, allow_pickle=False) as npz:
                loaded[prompt] = {k: npz[k].copy() for k in npz.files}
        data = loaded[prompt]
        cycle = int(selected["cycle_id"][i])
        matches = np.flatnonzero(data["cycle_id"] == cycle)
        if len(matches) != 1:
            raise ValueError("Ambiguous cycle")
        j = int(matches[0])
        start, anchor = int(data["prefix_length"][j]), int(data["anchor_id"][j])
        prefix = data["trajectory_token_ids"][:start + 1].astype(np.int64)
        original_a = int(data["accepted_len"][j])
        if digest_bytes(prefix) != str(data["prefix_sha256"][j]) or prefix[-1] != anchor:
            raise ValueError("Prefix/anchor mismatch")
        if not np.array_equal(index[original_row, 2:5], [prompt, cycle, original_a]):
            raise ValueError("Original row alignment failure")
        if original_a != selected["accepted_len"][i]:
            raise ValueError("Original plotted label mismatch")
        rows.append({"original_row": original_row, "prompt_id": prompt, "cycle_id": cycle,
                     "group": str(selected["group"][i]), "source": str(selected["source"][i]),
                     "prefix_length": start, "prompt_tokens": shard["prompt_tokens"],
                     "anchor_id": anchor, "prefix_sha256": digest_bytes(prefix),
                     "original_accepted_len": original_a, "prefix": prefix,
                     "original_fused": data["features"][j, 0]})
    return rows, manifest


def digest_bytes(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "analysis", "models", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=1800)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing replay")
    if args.limit < 0 or args.max_seconds < 1:
        raise ValueError("Invalid limit")
    used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError("Requested GPU is occupied; no job launched")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache
    from dflash.model import DFlashDraftModel, extract_context_feature
    torch.set_num_threads(4)
    torch.manual_seed(927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    states, manifest = select_states(args.cache, args.analysis, args.limit)
    models = json.loads(args.models.read_text())
    args.output.mkdir(parents=True)
    config = {"models": models, "states_planned": len(states), "block_size": 16,
              "source_cache": str(args.cache), "source_analysis": str(args.analysis),
              "source_manifest_sha256": digest(args.cache / "manifest.json"),
              "source_coordinates_sha256": digest(args.analysis / "plot_coordinates.npz"),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": digest(__file__), "torch": torch.__version__,
              "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
              "dtype": "bfloat16", "attention": "sdpa", "tf32": False,
              "confidence_columns": ["entropy", "prob_of_known_token", "top1_top2_prob_margin", "logprob_of_known_token"],
              "probability_temperature": 1.0, "decoding": "greedy",
              "protocol": "Replay identical saved prefix including original anchor, recompute fused features and current B16 label together. Primary input distribution is at start-1, before processing anchor. No old/new label join.",
              "history_protocol": "Last 16 committed positions; prompt logits other than the last prompt position are masked because they were not computed by original collection.",
              "noncausal_arrays": ["draft_confidence", "current_verifier_confidence"]}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).cuda().eval().requires_grad_(False)
    if draft.target_layer_ids != manifest["target_layer_ids"]:
        raise ValueError("Target-layer mismatch")
    for name, model in (("target", target), ("draft", draft)):
        expected = manifest["collection_config"]["hashes"][f"/tmp/dflash-prefusion-{name}/config.json"]
        if digest(Path(models[name]["path"]) / "config.json") != expected:
            raise ValueError("Original model config mismatch")
    n, width, vocab = len(states), target.config.hidden_size, target.config.vocab_size
    specs = {"predraft_probabilities": (np.float32, (n, vocab)),
             "fused": (np.float16, (n, width)), "target_final": (np.float16, (n, width)),
             "anchor_embedding": (np.float16, (n, width)),
             "predraft_history": (np.float32, (n, 16, 4)),
             "predraft_history_mask": (np.uint8, (n, 16)),
             "draft_confidence": (np.float32, (n, 15, 4)),
             "current_verifier_confidence": (np.float32, (n, 15, 4))}
    arrays = {k: np.lib.format.open_memmap(args.output / f"{k}.npy", mode="w+", dtype=d, shape=s)
              for k, (d, s) in specs.items()}
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    capture = []
    hook = draft.hidden_norm.register_forward_hook(lambda module, inputs, out: capture.append(out[:, -1:].detach().clone()))
    started, records = time.monotonic(), []
    print(f"MODELS_READY states={n}", flush=True)
    try:
        with torch.inference_mode(), (args.output / "rows.jsonl").open("w") as stream:
            for i, state in enumerate(states):
                if time.monotonic() - started > args.max_seconds:
                    raise RuntimeError("Time bound reached; partial replay preserved without COMPLETE")
                prefix, start = state["prefix"], state["prefix_length"]
                ids = torch.tensor(prefix[None], dtype=torch.long, device="cuda")
                tc = DynamicCache()
                positions, valid = history_positions(start, state["prompt_tokens"])
                pre = target(ids[:, :-1], past_key_values=tc, use_cache=True,
                             output_hidden_states=True, logits_to_keep=len(positions))
                if tc.get_seq_length() != start:
                    raise ValueError("Prefill cache alignment failure")
                stats, history_p = confidence(pre.logits[0], ids[0, torch.as_tensor(positions + 1, device="cuda")])
                prob = history_p[-1]
                arrays["predraft_probabilities"][i] = prob.cpu().numpy()
                arrays["predraft_history"][i] = 0
                arrays["predraft_history_mask"][i] = 0
                arrays["predraft_history"][i, -len(positions):] = stats * valid[:, None]
                arrays["predraft_history_mask"][i, -len(positions):] = valid
                arrays["target_final"][i] = pre.hidden_states[-1][0, -1].float().cpu().numpy()
                arrays["anchor_embedding"][i] = target.model.embed_tokens(ids[:, -1:])[0, 0].float().cpu().numpy()
                anchor_match = int(prob.argmax()) == state["anchor_id"]
                hidden = extract_context_feature(pre.hidden_states, draft.target_layer_ids)
                del pre, prob, history_p
                tokens = torch.full((1, 16), draft.mask_token_id, dtype=torch.long, device="cuda")
                tokens[:, :1] = ids[:, -1:]
                capture.clear()
                drafted = draft(target_hidden=hidden, noise_embedding=target.model.embed_tokens(tokens),
                    position_ids=torch.arange(start + 16, device="cuda")[None],
                    past_key_values=DynamicCache(), use_cache=True, is_causal=False)
                if len(capture) != 1 or capture[0].shape != (1, 1, width):
                    raise ValueError("Fusion capture failure")
                fused = capture[0][0, 0].float().cpu().numpy()
                arrays["fused"][i] = fused
                draft_logits = target.lm_head(drafted[:, 1:])[0]
                tokens[:, 1:] = draft_logits.argmax(-1)
                arrays["draft_confidence"][i] = confidence(draft_logits, tokens[0, 1:])[0]
                verified = target(tokens, past_key_values=tc, use_cache=True)
                posterior = verified.logits[0, :-1].argmax(-1)
                matches = (tokens[0, 1:] == posterior).cpu().numpy()
                accepted = int(np.cumprod(matches).sum())
                arrays["current_verifier_confidence"][i] = confidence(verified.logits[0, :-1], tokens[0, 1:])[0]
                terminal = any(int(t) in eos for t in tokens[0, :accepted + 1])
                original = state["original_fused"].astype(np.float32)
                row = {k: v for k, v in state.items() if k not in ("prefix", "original_fused")}
                row.update({"row": i, "accepted_len": accepted, "anchor_replay_match": anchor_match,
                            "terminal": terminal, "eligible": anchor_match and not terminal,
                            "label_replay_match": accepted == state["original_accepted_len"],
                            "fused_cosine_vs_original": float(np.dot(fused, original) / (np.linalg.norm(fused) * np.linalg.norm(original))),
                            "draft_ids": tokens[0, 1:].cpu().tolist(),
                            "target_argmax_ids": posterior.cpu().tolist()})
                stream.write(json.dumps(row) + "\n")
                records.append(row)
                del hidden, drafted, draft_logits, verified, tc
                capture.clear()
                if (i + 1) % 16 == 0 or i == n - 1:
                    for value in arrays.values():
                        value.flush()
                    stream.flush()
                    print(json.dumps({"rows": i + 1, "total": n, "elapsed_s": time.monotonic() - started,
                                      "label_matches": sum(r["label_replay_match"] for r in records),
                                      "anchor_matches": sum(r["anchor_replay_match"] for r in records)}), flush=True)
    finally:
        hook.remove()
    audit = {"rows": n, "eligible": sum(r["eligible"] for r in records),
             "anchor_replay_matches": sum(r["anchor_replay_match"] for r in records),
             "label_replay_matches": sum(r["label_replay_match"] for r in records),
             "mean_fused_cosine": float(np.mean([r["fused_cosine_vs_original"] for r in records])),
             "minimum_fused_cosine": min(r["fused_cosine_vs_original"] for r in records),
             "elapsed_seconds": time.monotonic() - started,
             "binding": {name: digest(args.output / name) for name in
                         ["config.json", "rows.jsonl"] + [k + ".npy" for k in arrays]}}
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite saved values")
    if not np.allclose(arrays["predraft_probabilities"].sum(-1), 1, atol=1e-5):
        raise ValueError("Probability normalization failure")
    (args.output / "COMPLETE.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
