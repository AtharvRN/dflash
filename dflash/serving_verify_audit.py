"""Opt-in CPU-side length audit, after the normal result D2H synchronization."""
import json
import os

_handle = None


def make_record(reqs, lengths, commit_lens, packed_tokens, executed_tokens, graph):
    assert len(reqs) == len(lengths) == len(commit_lens)
    assert sum(lengths) <= packed_tokens <= executed_tokens
    rows = []
    for req, length, commit in zip(reqs, lengths, commit_lens):
        assert 1 <= commit <= length <= 16
        rows.append({"rid": req.rid, "verify_length": int(length), "accepted": int(commit)-1,
                     "counted": not req.is_retracted and not req.finished()})
    return {"requests": rows, "selected_tokens": sum(lengths),
            "packed_tokens": int(packed_tokens), "executed_tokens": int(executed_tokens),
            "cuda_graph": bool(graph)}


def record_result(reqs, result, commit_lens):
    global _handle
    if result.dflash_verify_lens is None:
        return
    assert result.dflash_verify_lens.is_cpu
    row = make_record(reqs, result.dflash_verify_lens.tolist(), commit_lens,
                      result.dflash_verify_packed_tokens, result.dflash_verify_executed_tokens,
                      result.can_run_cuda_graph)
    if _handle is None:
        _handle = open(os.environ["SGLANG_DFLASH_VERIFY_AUDIT_PATH"], "x", buffering=1)
    _handle.write(json.dumps(row, separators=(",", ":")) + "\n")
