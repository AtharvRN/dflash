"""Same-prefix numerical control outside SGLang; never a speed benchmark.

Replay a saved target-verification state using Transformers SDPA. Duplicate the
identical request to change GEMM shape without any ragged packing, then repeat in
FP32. The cached prefix is held fixed, including its original BF16 rounding.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from profile_sglang_latency import ROOT, atomic_json, check_gpu, command


def compare(a, b):
    import torch
    a, b = a.float(), b.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Invalid reference tensors")
    return {"shape": list(a.shape), "max_abs": float((a-b).abs().max()),
            "relative_l2": float((a-b).norm()/b.norm().clamp_min(1e-12)),
            "per_token_relative_l2": ((a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)).tolist(),
            "bitwise_equal": bool(torch.equal(a, b))}


def summarize(hidden, logits, tokens, reference_hidden=None, reference_logits=None):
    prediction = logits.argmax(-1)
    accepted = int((tokens[1:] == prediction[:-1]).int().cumprod(0).sum())
    result = {"A": accepted, "bonus": int(prediction[accepted]), "top1": prediction.tolist()}
    if reference_hidden is not None:
        result["hidden"] = compare(hidden, reference_hidden)
        result["hidden_groups"] = [compare(a, b) for a, b in
                                    zip(hidden.split(2560, dim=-1), reference_hidden.split(2560, dim=-1))]
        result["logits"] = compare(logits, reference_logits)
        result["top1_mismatches"] = int((prediction != reference_logits.argmax(-1)).sum())
    return result


def replay_queries(state, batch, block, device):
    """Preserve every real query; add only causally subsequent dummy queries."""
    import torch
    real = len(state["input_ids"])
    if batch < 1 or not real <= block <= 32:
        raise ValueError("Invalid replay query shape")
    positions = torch.arange(state["prefix_length"], state["prefix_length"]+block,
                             device=device)
    if not torch.equal(positions[:real].cpu(), state["positions"].cpu()):
        raise ValueError("Saved positions must immediately follow the committed prefix")
    ids = torch.zeros(block, device=device, dtype=torch.long)
    ids[:real] = state["input_ids"].to(device)
    return ids[None].expand(batch, -1), positions[None].expand(batch, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-batch-size", type=int, default=90)
    parser.add_argument("--shape-block-size", type=int, default=0,
                        help="Append causal dummy queries in the batched control to match an exact execution-row count")
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.shape_batch_size <= 128:
        raise ValueError("Use a fresh output and bounded batch size")
    lock = (ROOT / f"gpu_{args.gpu}_actual_block.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = check_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True)
    models = json.loads((ROOT / "models.json").read_text())
    state = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if state["role"] != "target" or len(state["prefix_kv"]) != 36:
        raise ValueError("Expected the pinned Qwen3-4B target state")
    real = len(state["input_ids"])
    shape_block = args.shape_block_size or real
    if not real <= shape_block <= 32:
        raise ValueError("Padded block must preserve the real prefix and be <=32")
    digest = hashlib.sha256()
    with args.artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    config = {"artifact": str(args.artifact), "artifact_sha256": digest.hexdigest(),
              "gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__,
              "model": models["target"], "shape_batch_size": args.shape_batch_size,
              "shape_block_size": shape_block,
              "prefix_length": state["prefix_length"], "tokens": state["input_ids"].tolist(),
              "hidden_tuple_indices": [2, 10, 18, 26, 34],
              "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "code_commit": command(["git", "rev-parse", "HEAD"]).strip(),
              "limits": "Same saved BF16 prefix; not recomputed FP32 prefix, not an SGLang parity certificate. Padded queries are causal target-only dummies, not new draft candidates."}
    atomic_json(args.output / "config.json", config)
    started = time.monotonic()
    outputs, summary = {}, {}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            models["target"]["path"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).eval().cuda()
        for dtype in (torch.bfloat16, torch.float32):
            model.to(dtype=dtype)
            for batch, block in ((1, real), (args.shape_batch_size, shape_block)):
                label = f"{str(dtype).split('.')[-1]}_c{batch}"
                cache = DynamicCache()
                for layer, chunks in enumerate(state["prefix_kv"]):
                    # Expanded views share the immutable prefix until HF appends
                    # suffix KV. No C separate copies are staged up front.
                    k, v = [torch.cat([pair[j] for pair in chunks], dim=0).permute(1, 0, 2)
                            .unsqueeze(0).to(device="cuda", dtype=dtype).expand(batch, -1, -1, -1)
                            for j in (0, 1)]
                    cache.update(k, v, layer)
                if cache.get_seq_length() != state["prefix_length"]:
                    raise AssertionError("Saved prefix cache length mismatch")
                ids, positions = replay_queries(state, batch, block, "cuda")
                mask = torch.ones((batch, state["prefix_length"]+ids.shape[1]), device="cuda", dtype=torch.long)
                with torch.inference_mode():
                    out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                past_key_values=cache, use_cache=True, output_hidden_states=True)
                    hidden = torch.cat([out.hidden_states[i][0, :real] for i in config["hidden_tuple_indices"]], dim=-1).cpu()
                    logits = out.logits[0, :real].cpu()
                    identical_replicas = all(torch.equal(out.logits[i], out.logits[0]) for i in range(batch))
                outputs[label] = {"hidden": hidden, "logits": logits}
                summary[label] = {
                    "batch_size": batch, "block_size": block, "real_queries_per_request": real,
                    "vs_saved_packed": summarize(hidden, logits, state["input_ids"], state["actual_hidden"], state["actual_logits"]),
                    "vs_saved_independent": summarize(hidden, logits, state["input_ids"], state["independent_hidden"], state["independent_logits"]),
                    "replicas_bitwise_equal_logits": identical_replicas}
                single = outputs[f"{str(dtype).split('.')[-1]}_c1"]
                summary[label]["vs_single_same_dtype"] = summarize(hidden, logits, state["input_ids"], single["hidden"], single["logits"])
                del out, cache, ids, positions, mask, k, v
                gc.collect()
                torch.cuda.empty_cache()
                atomic_json(args.output / "summary.json", summary)
                print(json.dumps({"case": label, "hidden_vs_single": summary[label]["vs_single_same_dtype"]["hidden"],
                                  "top1": summary[label]["vs_single_same_dtype"]["top1"]}), flush=True)
        torch.save(outputs, args.output / "outputs.pt")
        atomic_json(args.output / "COMPLETE.json", {"elapsed_s": time.monotonic()-started})
    except BaseException as error:
        atomic_json(args.output / "FAILED.json", {"error": repr(error), "elapsed_s": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
