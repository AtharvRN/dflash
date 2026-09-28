"""Non-synchronizing CUDA-event and host-call spans for pinned DFlash v1.

CUDA-event time is stream elapsed time, NOT a sum of kernel execution times.
Nested spans are retained; exclusive values are computed by the report, not by
summing overlapping parent/child measurements. Clean runs do not load this hook.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import atexit
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

ACTIVE = None
PENDING = []
MODE = os.environ["DFLASH_COMPONENT_PROFILE"]
ROOT = Path(os.environ["DFLASH_PROFILE_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)
HANDLE = None
COUNTER = 0


@contextmanager
def span(name):
    if ACTIVE is None:
        yield
        return
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    parent = ACTIVE["stack"][-1] if ACTIVE["stack"] else None
    index = len(ACTIVE["spans"])
    item = {"name": name, "parent": parent, "start": start, "end": end}
    ACTIVE["spans"].append(item)
    ACTIVE["stack"].append(index)
    start.record()
    begin = time.perf_counter_ns()
    annotation_name = "DFLASH/" + name
    if name.endswith("_cycle"):
        annotation_name += f"|B={ACTIVE['block_size']}|C={ACTIVE['batch_size']}|cycle={ACTIVE['cycle']}"
    annotation = torch.profiler.record_function(annotation_name) if MODE == "trace" else nullcontext()
    try:
        with annotation:
            yield
    finally:
        item["host_call_ms"] = (time.perf_counter_ns() - begin) / 1e6
        end.record()
        ACTIVE["stack"].pop()


def drain(force=False):
    global HANDLE
    if force and PENDING:
        torch.cuda.synchronize()
    while PENDING and PENDING[0]["spans"][0]["end"].query():
        item = PENDING.pop(0)
        item.pop("stack")
        for s in item["spans"]:
            s["stream_elapsed_ms"] = s.pop("start").elapsed_time(s.pop("end"))
        if HANDLE is None:
            HANDLE = (ROOT / f"cycles_{os.getpid()}.jsonl").open("a", buffering=1)
        HANDLE.write(json.dumps(item, sort_keys=True) + "\n")


def wrap(function, name):
    @wraps(function)
    def timed(*args, **kwargs):
        with span(name):
            out = function(*args, **kwargs)
        if ACTIVE is not None and name in {"draft_forward", "target_forward"}:
            for field in ("can_run_cuda_graph", "can_run_graph"):
                value = getattr(out, field, None)
                if value is not None:
                    ACTIVE[name + "_graph"] = bool(value)
        return out
    return timed


def install(module):
    cls = module.DFlashWorker
    source = inspect.getsource(module)
    if not all(hasattr(cls, name) for name in (
        "_prepare_for_speculative_decoding", "_greedy_sample_from_vocab_parallel_head",
        "_append_target_hidden_to_draft_kv", "forward_batch_generation")):
        raise RuntimeError("Unexpected DFlash implementation; do not use these boundaries")
    (ROOT / f"hook_{os.getpid()}.json").write_text(json.dumps({
        "worker_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_path": module.__file__, "mode": MODE, "torch": torch.__version__,
        "instrumentation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "CUDA events on current stream; no per-stage synchronization",
    }, indent=2))
    original_init = cls.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.draft_model_runner.forward = wrap(self.draft_model_runner.forward, "draft_forward")
        self.target_worker.forward_batch_generation = wrap(
            self.target_worker.forward_batch_generation, "target_forward")

    cls.__init__ = initialize
    for method, name in (
        ("_prepare_for_speculative_decoding", "draft_and_verify_prepare"),
        ("_greedy_sample_from_vocab_parallel_head", "draft_projection_argmax"),
        ("_append_target_hidden_to_draft_kv", "draft_kv_upkeep"),
    ):
        setattr(cls, method, wrap(getattr(cls, method), name))
    verify = module.DFlashVerifyInput
    verify.prepare_for_verify = wrap(verify.prepare_for_verify, "target_verify_prepare")
    verify.verify = wrap(verify.verify, "acceptance_and_target_kv_commit")
    original_cycle = cls.forward_batch_generation

    @wraps(original_cycle)
    def cycle(self, batch, **kwargs):
        global ACTIVE, COUNTER
        drain()
        if ACTIVE is not None:
            raise RuntimeError("Unexpected recursive DFlash cycle")
        prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        COUNTER += 1
        try:
            label = json.loads((ROOT / "control.json").read_text())["label"]
        except FileNotFoundError:
            label = "startup"
        lengths = getattr(batch, "seq_lens_cpu", None)
        lengths = lengths.tolist() if lengths is not None else None
        ACTIVE = {"cycle": COUNTER, "label": label,
                  "phase": "prefill" if prefill else "decode",
                  "batch_size": int(batch.batch_size()), "block_size": int(self.block_size),
                  "prefix_lens": lengths, "spans": [], "stack": []}
        current = ACTIVE
        try:
            with span("prefill_cycle" if prefill else "decode_cycle"):
                result = original_cycle(self, batch, **kwargs)
            current["accepted_drafts"] = int(result.num_correct_drafts or 0)
            values = getattr(result, "num_correct_drafts_per_req_cpu", None)
            if values is not None:
                current["accepted_per_request"] = list(map(int, values))
            current["target_graph"] = bool(result.can_run_cuda_graph)
            PENDING.append(current)
            return result
        finally:
            ACTIVE = None
            drain()

    cls.forward_batch_generation = cycle
    atexit.register(lambda: drain(force=True))
