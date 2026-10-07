"""Greedy Transformers reference and matched-state collector, not serving code."""
from __future__ import annotations

import copy
import hashlib
import time

import torch
from transformers import DynamicCache

from .history_policy import History, blocks_checked
from .model import extract_context_feature


def valid_verifier_entropy(logits, accepted):
    """Rows 0..A predict accepted proposals and correction/bonus on valid prefixes.

    Row A+1 would consume the first rejected proposal; never include it.
    For full acceptance all B rows are valid. Accumulate in FP32, in nats.
    """
    if logits.ndim != 3 or logits.shape[0] != 1 or not 0 <= accepted < logits.shape[1]:
        raise ValueError("invalid verifier logits or accepted length")
    x = logits[0, :accepted+1].float()
    if not torch.isfinite(x).all():
        raise ValueError("non-finite verifier logits")
    logp = x.log_softmax(-1)
    return float((-(logp.exp()*logp).sum(-1).mean()).clamp_min(0).item())


@torch.inference_mode()
def generate_history(model, target, input_ids, *, blocks, choose, max_new_tokens,
                     stop_token_ids=(), collect=False, prompt_id="", group="assessment",
                     max_cycles=0):
    """choose(history_snapshot, cycle) runs BEFORE any current-cycle drafting.

    collect=True independently redrafts every arm from cloned identical caches.
    Only the chosen arm commits KV/features and updates the history. Randomized
    collection behavior can therefore cover histories other than fixed B16.
    """
    blocks = blocks_checked(blocks)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("single nonempty prompt required")
    if max_new_tokens < 1 or max_cycles < 0:
        raise ValueError("invalid generation limits")
    device, n = input_ids.device, input_ids.shape[1]
    limit, width = n + max_new_tokens, max(blocks)
    sequence = torch.full((1, limit+width+1), model.mask_token_id, device=device, dtype=torch.long)
    sequence[:, :n] = input_ids
    positions = torch.arange(sequence.shape[1], device=device).unsqueeze(0)
    tc, dc = DynamicCache(), DynamicCache()
    stops = set(stop_token_ids or ())
    history, records = History(), []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = target(input_ids, position_ids=positions[:, :n], past_key_values=tc,
                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
    sequence[:, n] = output.logits[:, -1].argmax(-1)
    pending = extract_context_feature(output.hidden_states, model.target_layer_ids)
    del output
    start, cycle, terminal = n, 0, False

    def run_block(b, target_cache, draft_cache):
        block = torch.full((1, b), model.mask_token_id, device=device, dtype=torch.long)
        block[:, 0] = sequence[:, start]
        hidden = model(target_hidden=pending, noise_embedding=target.model.embed_tokens(block),
                       position_ids=positions[:, draft_cache.get_seq_length():start+b],
                       past_key_values=draft_cache, use_cache=True, is_causal=False)
        block[:, 1:] = target.lm_head(hidden[:, 1:]).argmax(-1)
        draft_cache.crop(start)
        verified = target(block, position_ids=positions[:, start:start+b],
                          past_key_values=target_cache, use_cache=True, output_hidden_states=True)
        posterior = verified.logits.argmax(-1)
        a = int((block[:, 1:] == posterior[:, :-1]).cumprod(-1).sum().item())
        committed = torch.cat((block[:, :a+1], posterior[:, a:a+1]), dim=1)
        terminal = any(int(t) in stops for t in committed[0].tolist())
        return block, verified, posterior, a, terminal

    while start < limit-1 and int(sequence[0, start]) not in stops:
        if max_cycles and cycle >= max_cycles:
            break
        snapshot = history.snapshot()
        b = choose(dict(snapshot), cycle)
        if b not in blocks:
            raise ValueError("controller selected an uncalibrated block size")
        prefix = sequence[0, :start+1].tolist()
        record = {"schema_version": 1, "prompt_id": str(prompt_id), "group": group,
                  "cycle": cycle, "prefix_length_including_anchor": start+1,
                  "prefix_sha256": hashlib.sha256(",".join(map(str, prefix)).encode()).hexdigest(),
                  "history": snapshot, "selected_block": b, "outcomes": {},
                  "outcome_kind": "actual_redraft" if collect else "closed_loop_selected_only"}
        terminal_any = False
        if collect:
            for arm in blocks:
                if arm == b:
                    continue
                probe, verified, posterior, a, terminal = run_block(arm, copy.deepcopy(tc), copy.deepcopy(dc))
                record["outcomes"][str(arm)] = {"accepted": a}
                terminal_any |= terminal
                del probe, verified, posterior
        block, output, posterior, a, terminal = run_block(b, tc, dc)
        entropy = valid_verifier_entropy(output.logits, a)
        record["outcomes"][str(b)] = {"accepted": a}
        # Reserve room for the bonus for EVERY counterfactual arm. No clipped labels.
        capped = start + width >= limit
        record.update(terminal_or_capped=bool(terminal_any or terminal or capped),
                      eligible=not (terminal_any or terminal or capped),
                      selected_entropy_nats=entropy, selected_accepted=a,
                      selected_progress=a+1)
        sequence[:, start:start+a+1] = block[:, :a+1]
        sequence[:, start+a+1] = posterior[:, a]
        start += a+1
        tc.crop(start)
        pending = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :a+1]
        history.observe(b, a, entropy)
        records.append(record)
        cycle += 1
        del output, posterior, block
        if terminal:
            break
    ids = sequence[:, :min(start+1, limit)]
    for i, token in enumerate(ids[0, n:].tolist()):
        if token in stops:
            ids = ids[:, :n+i+1]
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"output_ids": ids, "cycles": records, "elapsed_s": time.perf_counter()-started,
            "scope": "collection with counterfactual replay" if collect else "single-request reference rollout; not SGLang",
            "cycle_limited": bool(max_cycles and cycle >= max_cycles and start < limit-1 and not terminal)}
