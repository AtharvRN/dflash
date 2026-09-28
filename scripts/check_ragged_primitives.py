"""Reference checks against recovered production helpers (not model correctness)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from typing import List, Optional

import torch


def load_worker(source, device):
    if device == "cuda":
        sys.path.insert(0, str(source / "python"))
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
        cls = DFlashWorkerV2
    else:
        # Local CPU environment need not have SGLang/CUDA installed. Execute the
        # original method AST verbatim; only the eager acceptance branch is used.
        path = source / "python/sglang/srt/speculative/dflash_worker_v2.py"
        tree = ast.parse(path.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2")
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_ragged_accept_bonus")
        module = ast.Module(body=[ast.ClassDef(name="CPUWorker", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
        namespace = {"torch": torch, "Optional": Optional, "List": List,
                     "logger": logging.getLogger(__name__)}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        cls = namespace["CPUWorker"]
    worker = cls.__new__(cls)
    worker.device = torch.device(device)
    worker.block_size = 16
    worker._mask_token_id = 151669
    worker._use_triton_accept_bonus = device == "cuda"
    worker._accept_bonus_buffer_cap = 0
    worker._accept_bonus_buffer_slot = 0
    worker._accept_len_buf = None
    worker._rel_2d_cache = {}
    worker._ragged_verify_buffer_cap = 0
    worker._ragged_verify_req_cap = 0
    worker._ragged_verify_buffer_slot = 0
    worker._ragged_verify_buffers = {}
    return worker


def equal(actual, expected, label):
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise AssertionError(f"{label}: {actual.cpu().tolist()} != {expected.cpu().tolist()}")


def acceptance_case(worker, lengths, accepts, seed):
    rng = random.Random(seed)
    candidates, target, offsets = [], [], [0]
    expected_bonus, expected_out, expected_indices = [], [], []
    for b, a in zip(lengths, accepts):
        row = [rng.randrange(100, 50000) for _ in range(b)]
        pred = [rng.randrange(50001, 100000) for _ in range(b)]
        pred[:a] = row[1:a+1]
        # Matches AFTER the first failure must never be accepted.
        if a < b-1:
            pred[a+1:b-1] = row[a+2:b]
        expected_bonus.append(pred[a])
        expected_out.extend(row[1:a+1] + [pred[a]])
        expected_indices.extend(range(offsets[-1], offsets[-1]+a+1))
        candidates.extend(row)
        target.extend(pred)
        offsets.append(len(candidates))
    tensor = lambda x: torch.tensor(x, dtype=torch.int64, device=worker.device)
    result = worker._compute_ragged_accept_bonus(
        draft_tokens=tensor(candidates), target_predict=tensor(target),
        block_lens=tensor(lengths), offsets=tensor(offsets))
    for label, actual, expected in zip(
        ["acceptance", "commit length", "bonus", "compact output", "commit indices"],
        result, [accepts, [a+1 for a in accepts], expected_bonus, expected_out, expected_indices]):
        equal(actual, expected, label)
    if worker.device.type == "cuda" and not worker._use_triton_accept_bonus:
        raise AssertionError("GPU test silently fell back from Triton")


def packing_case(worker, lengths, seed, extra_bucket):
    rng = random.Random(seed)
    bs = len(lengths)
    tensor = lambda x: torch.tensor(x, dtype=torch.int64, device=worker.device)
    prefixes = [rng.randrange(0, 241) for _ in lengths]
    pool_ids = rng.sample(range(bs+7), bs)
    anchors = [rng.randrange(100, 50000) for _ in lengths]
    req_to_token = torch.arange((bs+7)*272, device=worker.device, dtype=torch.int32).view(bs+7, 272) + 73
    ids, pos, loc, offsets, projection = worker._make_ragged_block(
        verified_id=tensor(anchors), prefix_lens=tensor(prefixes), block_lens=tensor(lengths),
        req_pool_indices=tensor(pool_ids), req_to_token=req_to_token)
    expected_ids, expected_pos, expected_loc, expected_proj, expected_offsets = [], [], [], [], [0]
    for i, b in enumerate(lengths):
        expected_ids.extend([anchors[i]] + [worker._mask_token_id]*(b-1))
        expected_pos.extend(range(prefixes[i], prefixes[i]+b))
        expected_loc.extend(pool_ids[i]*272 + p + 73 for p in range(prefixes[i], prefixes[i]+b))
        expected_proj.extend(range(expected_offsets[-1]+1, expected_offsets[-1]+b))
        expected_offsets.append(expected_offsets[-1]+b)
    for actual, expected, label in zip([ids, pos, loc, offsets, projection],
                                     [expected_ids, expected_pos, expected_loc, expected_offsets, expected_proj],
                                     ["draft IDs", "positions", "KV locations", "offsets", "projection indices"]):
        equal(actual, expected, label)
    # Distinct per-position candidate IDs expose accidental row/index mixing.
    drafted = torch.arange(sum(lengths), device=worker.device, dtype=torch.int64) + 9000
    bucket = min(16, (sum(lengths)+bs-1)//bs + extra_bucket)
    graph_ids, graph_pos, graph_loc, graph_lens, real_indices, chosen = worker._make_graph_padded_ragged_verify(
        draft_tokens=drafted, prefix_lens=tensor(prefixes), block_lens=tensor(lengths), offsets=offsets,
        req_pool_indices=tensor(pool_ids), req_to_token=req_to_token,
        max_block_size=16, graph_num_tokens_per_batch=bucket, real_total=sum(lengths))
    assert chosen == bucket and graph_ids.numel() == bs*bucket
    expected_lens = lengths.copy()
    remaining = bs*bucket - sum(lengths)
    for i in reversed(range(bs)):
        addition = min(16-lengths[i], remaining)
        expected_lens[i] += addition
        remaining -= addition
    assert remaining == 0
    equal(graph_lens, expected_lens, "graph physical lengths")
    equal(graph_ids[real_indices], drafted, "real-token gather")
    equal(graph_pos[real_indices], expected_pos, "real positions")
    equal(graph_loc[real_indices], expected_loc, "real KV locations")
    offset = 0
    for i, physical in enumerate(expected_lens):
        equal(graph_pos[offset:offset+physical], range(prefixes[i], prefixes[i]+physical), "physical positions")
        equal(graph_loc[offset:offset+physical],
              [pool_ids[i]*272+p+73 for p in range(prefixes[i], prefixes[i]+physical)], "physical KV locations")
        equal(graph_ids[offset+lengths[i]:offset+physical],
              [worker._mask_token_id]*(physical-lengths[i]), "dummy suffix")
        offset += physical
    # Returning clones in the test prevents intentional ping-pong buffer reuse
    # from affecting assertions on subsequent calls.


def run(source, device):
    worker = load_worker(source, device)
    counts = {"acceptance_batches": 0, "acceptance_request_cases": 0, "packing_batches": 0}
    lengths = [b for b in range(1, 17) for _ in range(b)]
    accepts = [a for b in range(1, 17) for a in range(b)]
    acceptance_case(worker, lengths, accepts, 932)
    counts["acceptance_batches"] += 1
    counts["acceptance_request_cases"] += len(lengths)
    # Growth, shrinking, reordering, different parity, all-reject/all-accept and
    # several buffer wraparounds, including non-power-of-two batch sizes.
    rng = random.Random(933)
    for j, bs in enumerate([1, 2, 3, 8, 16, 31, 64, 65, 7, 128, 4, 1]*4):
        lengths = [rng.randrange(1, 17) for _ in range(bs)]
        accepts = [0 if j % 3 == 0 else b-1 if j % 3 == 1 else rng.randrange(b) for b in lengths]
        acceptance_case(worker, lengths, accepts, 1000+j)
        counts["acceptance_batches"] += 1
        counts["acceptance_request_cases"] += bs
        if device == "cuda":
            packing_case(worker, lengths, 2000+j, j % 3)
            counts["packing_batches"] += 1
    if device == "cuda":
        for j, lengths in enumerate([[1], [16], [1]*64, [16]*64, [1, 16]*32, list(range(1, 17))]):
            packing_case(worker, lengths, 3000+j, 0)
            counts["packing_batches"] += 1
        torch.cuda.synchronize()
    path = source / "python/sglang/srt/speculative/dflash_worker_v2.py"
    return {"success": True, "device": device, "counts": counts,
            "worker_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "limits": ["primitive correctness only", "no attention, full-model logits, scheduler or KV lifetime validation",
                       "GPU path imports actual SGLang class; CPU path executes original acceptance method AST"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior check results")
    result = run(args.source.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
