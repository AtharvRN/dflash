"""Compare actual short drafts with clipped B16 labels at identical greedy states."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.model import DFlashDraftModel, extract_context_feature
from dflash.policy import DFlashV2HorizonBlockPolicy


def prefix_matches(candidate, posterior):
    return next((i for i, (a, b) in enumerate(zip(candidate, posterior)) if a != b),
                min(len(candidate), len(posterior)))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_prompts(manifest, split_file, count, seed):
    split = json.loads(split_file.read_text())
    allowed = set(map(str, split["val_prompt_ids"]))
    rows = []
    seen = set()
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["manifest_index"])
            if key in allowed:
                if key in seen:
                    raise ValueError(f"Duplicate manifest_index {key}")
                seen.add(key)
                rows.append(row)
    random.Random(seed).shuffle(rows)
    if not rows:
        raise ValueError("No prompts overlap the fixed validation IDs")
    return rows[:count]


def summarize(rows, bootstrap=1000):
    eligible = [r for r in rows if r["eligible"]]
    output = {"states": len(rows), "eligible_states": len(eligible),
              "prompts": len({r["prompt_id"] for r in rows}), "blocks": {}, "policies": {}}
    if not eligible:
        return output
    groups = sorted({r["prompt_id"] for r in eligible})
    group_index = {p: i for i, p in enumerate(groups)}
    rng = np.random.default_rng(0)
    draws = rng.integers(len(groups), size=(bootstrap, len(groups)))
    blocks = sorted(set.intersection(*(set(r["outcomes"]) for r in eligible)), key=int)
    for b in blocks:
        actual = np.array([r["outcomes"][b]["accepted"] for r in eligible], dtype=float)
        proxy = np.array([min(r["outcomes"]["16"]["accepted"], int(b) - 1) for r in eligible])
        baseline = np.array([r["outcomes"]["16"]["accepted"] for r in eligible], dtype=float)
        error = actual - proxy
        # Resample whole prompts, keeping all their paired states together.
        totals = np.zeros((len(groups), 5))
        for r, e, base in zip(eligible, error, baseline):
            totals[group_index[r["prompt_id"]]] += [abs(e), e, e != 0, 1, base]
        boot = totals[draws].sum(axis=1)
        ci = lambda v: np.quantile(v, [.025, .975]).tolist()
        output["blocks"][b] = {
            "mean_actual_accepted": float(actual.mean()), "mean_proxy_accepted": float(proxy.mean()),
            "label_mae": float(abs(error).mean()), "signed_error": float(error.mean()),
            "mismatch_rate": float((error != 0).mean()),
            "mae_ci95": ci(boot[:, 0] / boot[:, 3]),
            "signed_error_ci95": ci(boot[:, 1] / boot[:, 3]),
            "mismatch_ci95": ci(boot[:, 2] / boot[:, 3]),
            "actual_retention": float(actual.sum() / max(baseline.sum(), 1)),
            "proxy_retention": float(proxy.sum() / max(baseline.sum(), 1)),
            "retention_distortion_ci95": ci(boot[:, 1] / np.maximum(boot[:, 4], 1)),
            "draft_token_prefix_mismatch_rate": float(np.mean([
                r["outcomes"][b]["draft_ids"] != r["outcomes"]["16"]["draft_ids"][:int(b)-1]
                for r in eligible])),
        }
    for name in eligible[0]["policies"]:
        budget = np.array([r["policies"][name] - 1 for r in eligible], dtype=float)
        actual = np.array([r["outcomes"][str(r["policies"][name])]["accepted"] for r in eligible])
        baseline = np.array([r["outcomes"]["16"]["accepted"] for r in eligible])
        proxy = np.minimum(baseline, budget)
        totals = np.zeros((len(groups), 2))
        for r, delta, base in zip(eligible, actual-proxy, baseline):
            totals[group_index[r["prompt_id"]]] += [delta, base]
        boot = totals[draws].sum(axis=1)
        output["policies"][name] = {
            "mean_budget": float(budget.mean()), "mean_actual_accepted": float(actual.mean()),
            "mean_proxy_accepted": float(proxy.mean()),
            "actual_retention": float(actual.sum() / max(baseline.sum(), 1)),
            "proxy_retention": float(proxy.sum() / max(baseline.sum(), 1)),
            "actual_aggregate_ratio": float(actual.sum() / budget.sum()),
            "proxy_aggregate_ratio": float(proxy.sum() / budget.sum()),
            "actual_mean_cycle_ratio": float(np.mean(actual / budget)),
            "proxy_mean_cycle_ratio": float(np.mean(proxy / budget)),
            "retention_distortion_ci95": np.quantile(boot[:, 0] / np.maximum(boot[:, 1], 1), [.025, .975]).tolist(),
        }
    output["validation_checks"] = {
        "reverse_order_states": sum(r["reverse_order_checked"] for r in rows),
        "canonical_checked_outcomes": sum(r["canonical_checked"] for r in rows),
        "canonical_disagreements": sum(r["canonical_disagreements"] for r in rows),
        "truncated_verify_checks": sum(r["truncated_verify_checks"] for r in rows),
    }
    return output


class Policies:
    def __init__(self, args, draft):
        self.direct = None
        self.entropy = None
        if args.direct_checkpoint:
            self.direct = DFlashV2HorizonBlockPolicy(checkpoint_path=args.direct_checkpoint, alpha=args.alpha)
            if self.direct.config["architecture"] != "last_mlp":
                raise ValueError("Only the last-fused MLP is supported in this diagnostic")
            self.direct.reset(draft, "cuda")
        if args.entropy_checkpoint:
            ckpt = torch.load(args.entropy_checkpoint, map_location="cpu", weights_only=False)
            c = ckpt["config"]
            self.entropy = nn.Sequential(
                nn.Linear(c["input_dim"], c["proj_dim"]), nn.GELU(), nn.LayerNorm(c["proj_dim"]), nn.Dropout(0),
                nn.Linear(c["proj_dim"], c["hidden_size"]), nn.GELU(), nn.LayerNorm(c["hidden_size"]), nn.Dropout(0),
                nn.Linear(c["hidden_size"], c["num_slots"]),
            ).cuda().eval()
            self.entropy.load_state_dict({k.removeprefix("net."): v for k, v in ckpt["model"].items()})
            self.mean = ckpt["target_mean"].cuda()
            self.std = ckpt["target_std"].cuda()
        self.threshold = args.entropy_threshold

    def predict(self, fused):
        choices = {}
        if self.direct:
            self.direct.history.clear()
            self.direct.history.append(fused[0].float().cpu())
            choices["direct"] = self.direct.select_block_size()
        if self.entropy is not None:
            values = (self.entropy(fused.float()) * self.std + self.mean)[0]
            budget = int((values <= self.threshold).cumprod(0).sum())
            choices["entropy"] = max(1, budget) + 1
        return choices


@torch.inference_mode()
def run_prompt(args, row, target, draft, tokenizer, policies, count_before):
    ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt").cuda()
    n = ids.shape[1]
    if n > args.max_prompt_tokens:
        return [], {"prompt_id": str(row["manifest_index"]), "skipped": "prompt_length", "tokens": n}
    positions = torch.arange(n + args.max_new_tokens + 32, device="cuda").unsqueeze(0)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    prefill = target(ids, position_ids=positions[:, :n], past_key_values=target_cache,
                     use_cache=True, output_hidden_states=True, logits_to_keep=1)
    hidden = extract_context_feature(prefill.hidden_states, draft.target_layer_ids)
    sequence = torch.cat([ids, prefill.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    del prefill
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    offsets = np.linspace(0, args.max_new_tokens - 16, args.states_per_prompt, dtype=int).tolist()
    next_offset, start, cycle = 0, n, 0
    rows = []
    rng = random.Random(args.seed + int(row["manifest_index"]))

    def propose(b, dc, pending):
        block = torch.full((1, b), draft.mask_token_id, dtype=torch.long, device="cuda")
        block[:, 0] = sequence[:, start]
        output = draft(target_hidden=pending, noise_embedding=target.model.embed_tokens(block),
                       position_ids=positions[:, dc.get_seq_length():start+b],
                       past_key_values=dc, use_cache=True, is_causal=False)
        block[:, 1:] = target.lm_head(output[:, 1:, :]).argmax(-1)
        dc.crop(start)
        return block

    def verify(block, tc, save_hidden=False):
        b = block.shape[1]
        output = target(block, position_ids=positions[:, start:start+b], past_key_values=tc,
                        use_cache=True, output_hidden_states=save_hidden)
        posterior = output.logits.argmax(-1)
        accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(-1).sum())
        return output, posterior, accepted

    while start < n + args.max_new_tokens and int(sequence[0, start]) not in eos:
        selected = (next_offset < len(offsets) and start - n >= offsets[next_offset]
                    and start - n <= args.max_new_tokens - 16)
        pending = hidden
        if selected:
            tc_snapshot = copy.deepcopy(target_cache)
            dc_snapshot = copy.deepcopy(draft_cache)
            fused = draft.hidden_norm(draft.fc(pending[:, -1:]))[:, 0]
            choices = policies.predict(fused)
        baseline = propose(16, draft_cache, pending)
        output, posterior, accepted = verify(baseline, target_cache, True)
        if selected:
            outcomes = {"16": {"accepted": accepted, "draft_ids": baseline[0, 1:].tolist()}}
            block_sizes = sorted(set(args.blocks) | set(choices.values()))
            order = [b for b in block_sizes if b != 16]
            rng.shuffle(order)
            for b in order:
                block = propose(b, copy.deepcopy(dc_snapshot), pending)
                alt_output, _, alt_a = verify(block, copy.deepcopy(tc_snapshot))
                outcomes[str(b)] = {"accepted": alt_a, "draft_ids": block[0, 1:].tolist()}
                del alt_output
            state_index = count_before + len(rows)
            reverse_checked = state_index < args.reverse_check_states
            if reverse_checked:
                for b in reversed(block_sizes):
                    block = propose(b, copy.deepcopy(dc_snapshot), pending)
                    alt_output, _, alt_a = verify(block, copy.deepcopy(tc_snapshot))
                    assert block[0, 1:].tolist() == outcomes[str(b)]["draft_ids"], "Draft cache/order contamination"
                    assert alt_a == outcomes[str(b)]["accepted"], "Target cache/order contamination"
                    del alt_output
            canonical_checked = canonical_disagreements = truncated_checks = 0
            if state_index < args.canonical_check_states:
                tc = copy.deepcopy(tc_snapshot)
                token = sequence[:, start:start+1]
                canonical = []
                for k in range(15):
                    one = target(token, position_ids=positions[:, start+k:start+k+1], past_key_values=tc,
                                 use_cache=True, logits_to_keep=1)
                    token = one.logits[:, -1].argmax(-1, keepdim=True)
                    canonical.append(int(token))
                    if int(token) in eos:
                        break
                for b in block_sizes:
                    candidate = outcomes[str(b)]["draft_ids"]
                    observed = min(len(candidate), len(canonical))
                    canonical_checked += 1
                    canonical_disagreements += prefix_matches(candidate, canonical) != min(outcomes[str(b)]["accepted"], observed)
                    if b != 16:
                        truncated_output, _, trunc_a = verify(baseline[:, :b], copy.deepcopy(tc_snapshot))
                        assert trunc_a == min(accepted, b-1), "Target verification prefix differs across shapes"
                        truncated_checks += 1
                        del truncated_output
                del one, tc
            terminal = any(any(t in eos for t in o["draft_ids"][:o["accepted"]]) for o in outcomes.values())
            prefix = sequence[0, :start+1].tolist()
            record = {"prompt_id": str(row["manifest_index"]), "cycle": cycle, "prefix_length": start,
                      "generated_before_anchor": start-n, "source": row.get("source"), "category": row.get("split"),
                      "prefix_sha256": hashlib.sha256(np.asarray(prefix, dtype=np.int64).tobytes()).hexdigest(),
                      "prefix_token_ids": prefix, "outcomes": outcomes, "policies": choices,
                      "eligible": not terminal, "exclusion": "accepted_eos" if terminal else None,
                      "reverse_order_checked": reverse_checked, "canonical_checked": canonical_checked,
                      "canonical_disagreements": canonical_disagreements, "truncated_verify_checks": truncated_checks}
            rows.append((record, fused[0].cpu().to(torch.float16).numpy()))
            next_offset += 1
            del tc_snapshot, dc_snapshot, fused
            if count_before + len(rows) >= args.max_states:
                break
        committed = accepted + 1
        sequence = torch.cat([sequence[:, :start], baseline[:, :committed], posterior[:, accepted:accepted+1]], dim=1)
        hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[:, :committed].contiguous()
        start += committed
        target_cache.crop(start)
        cycle += 1
        if any(t in eos for t in sequence[0, n:].tolist()):
            break
        del output, posterior
    return rows, {"prompt_id": str(row["manifest_index"]), "states": len(rows), "cycles": cycle,
                  "prompt_tokens": n, "generated_tokens": min(sequence.shape[1]-n, args.max_new_tokens)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "validation-ids", "output-dir", "model", "draft-model"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--persistent-dir", type=Path)
    p.add_argument("--direct-checkpoint", type=Path)
    p.add_argument("--entropy-checkpoint", type=Path)
    p.add_argument("--alpha", type=float, default=.92)
    p.add_argument("--entropy-threshold", type=float, default=2.6000006198883057)
    p.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 12, 16])
    p.add_argument("--max-states", type=int, default=2000)
    p.add_argument("--max-prompts", type=int, default=500)
    p.add_argument("--states-per-prompt", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--reverse-check-states", type=int, default=5)
    p.add_argument("--canonical-check-states", type=int, default=10)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--attn-implementation", default="sdpa")
    return p.parse_args()


def main():
    args = parse_args()
    if 16 not in args.blocks or any(b < 2 or b > 16 for b in args.blocks):
        raise ValueError("Blocks must include 16 and lie in [2,16]")
    if args.max_new_tokens < 16 or min(args.max_states, args.max_prompts, args.states_per_prompt) < 1:
        raise ValueError("Invalid sample/generation limits")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; completed runs are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.persistent_dir:
        args.persistent_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rows = select_prompts(args.manifest, args.validation_ids, args.max_prompts, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"prompt_ids": [str(r["manifest_index"]) for r in rows], "temperature": 0,
                   "thinking": False, "ground_truth": "direct verification for every actual draft",
                   "trajectory": "B16 reference; policies evaluated at identical states, not closed-loop",
                   "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    for name, path in (("model_config", args.model/"config.json"), ("draft_config", args.draft_model/"config.json")):
        config[name] = json.loads(path.read_text())
    atomic_json(args.output_dir / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
             attn_implementation=args.attn_implementation, local_files_only=True).cuda().eval()
    draft = DFlashDraftModel.from_pretrained(args.draft_model, torch_dtype=torch.bfloat16,
             attn_implementation=args.attn_implementation, local_files_only=True).cuda().eval()
    policies = Policies(args, draft)
    all_rows, prompts, features = [], [], []
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=1)
    backups = []
    def backup(paths):
        if args.persistent_dir:
            for path in paths:
                tmp = args.persistent_dir / (path.name + ".tmp")
                shutil.copy2(path, tmp)
                tmp.replace(args.persistent_dir / path.name)
    backups.append(pool.submit(backup, [args.output_dir / "config.json"]))
    try:
        for i, row in enumerate(rows):
            batch, progress = run_prompt(args, row, target, draft, tokenizer, policies, len(all_rows))
            prompts.append(progress)
            shard = args.output_dir / f"prompt_{row['manifest_index']}.json"
            atomic_json(shard, {"progress": progress, "states": [x[0] for x in batch]})
            paths = [shard]
            if batch:
                feat = args.output_dir / f"prompt_{row['manifest_index']}_fused.npy"
                np.save(feat, np.stack([x[1] for x in batch]))
                paths.append(feat)
            backups.append(pool.submit(backup, paths))
            all_rows.extend(x[0] for x in batch)
            elapsed = time.monotonic() - started
            status = {"prompts_processed": i+1, "states": len(all_rows), "elapsed_s": round(elapsed, 2),
                      "states_per_s": len(all_rows)/max(elapsed, 1), "latest": progress}
            atomic_json(args.output_dir / "progress.json", status)
            print(json.dumps(status), flush=True)
            if len(all_rows) >= args.max_states:
                break
        result = summarize(all_rows)
        result.update({"elapsed_s": time.monotonic()-started, "prompt_progress": prompts,
                       "sample_complete": len(all_rows) >= args.max_states})
        atomic_json(args.output_dir / "summary.json", result)
        backups.append(pool.submit(backup, [args.output_dir / "summary.json", args.output_dir / "progress.json"]))
        for future in backups:
            future.result()
        print(json.dumps({k: v for k, v in result.items() if k != "prompt_progress"}, indent=2), flush=True)
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
