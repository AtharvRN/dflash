"""CUDA-event replacement for existing v2 phase boundaries, without stage sync.

Stream elapsed time includes stream idle gaps; it is not kernel execution time.
The clean process never imports this module. No runtime source file is edited.
Existing native scalar synchronization remains present in BOTH runs.
"""
import atexit
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(os.environ["DFLASH_V2_PROFILE_DIR"])
ACTIVE = None
PENDING = []
COUNTER = 0
HANDLE = None


def begin(name):
    if ACTIVE is None:
        return None
    item = {"name": name, "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True), "host_start": time.perf_counter_ns()}
    item["start"].record()
    ACTIVE["spans"].append(item)
    return item


def end(name, item):
    if item is not None:
        if name != item["name"]:
            raise AssertionError("Mismatched phase boundary")
        item["end"].record()
        item["host_call_ms"] = (time.perf_counter_ns()-item.pop("host_start"))/1e6


def drain(force=False):
    global HANDLE
    if force and PENDING:
        torch.cuda.synchronize()
    while PENDING and PENDING[0]["spans"][0]["end"].query():
        row = PENDING.pop(0)
        # The device-to-host prefix copy is queued before the timed root.
        if not row.pop("metadata_done").query():
            raise AssertionError("Metadata event should precede the next-cycle drain")
        row["prefix_lens"] = row.pop("prefix_cpu").tolist()
        for item in row["spans"]:
            item["stream_elapsed_ms"] = item.pop("start").elapsed_time(item.pop("end"))
        if HANDLE is None:
            HANDLE = (ROOT/f"cycles_{os.getpid()}.jsonl").open("a", buffering=1)
        HANDLE.write(json.dumps(row, sort_keys=True)+"\n")


def install(module):
    cls = module.DFlashWorkerV2
    source = inspect.getsource(module)
    required = ("draft_block_setup", "draft_model_forward", "draft_token_projection",
                "verify_preparation", "target_verify_forward", "acceptance_bonus", "post_verify_kv_materialize")
    if not all('"'+name+'"' in source for name in required):
        raise RuntimeError("Unexpected v2 phase boundaries")
    (ROOT/f"hook_{os.getpid()}.json").write_text(json.dumps({
        "source_path": module.__file__, "worker_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__,
        "timing": "current-stream CUDA events, no per-stage synchronization; stream elapsed is not kernel time",
        "draft_cuda_graph": False, "native_scalar_synchronization_preserved": True,
    }, indent=2))
    cls._timing_begin = lambda self, name: begin(name)
    cls._timing_end = lambda self, name, item: end(name, item)
    cls._timing_start_cycle = lambda self, **kwargs: None
    cls._timing_finish_cycle = lambda self: None

    def record(self, key, value):
        if ACTIVE is not None:
            ACTIVE["native"][key] = value

    cls._timing_set = record
    original = cls.forward_batch_generation

    @wraps(original)
    def cycle(self, batch, *args, **kwargs):
        global ACTIVE, COUNTER
        drain()
        if ACTIVE is not None:
            raise RuntimeError("Recursive v2 forward")
        prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        # Do not instrument prefill: it is included in clean wall time, separately.
        if prefill or len(batch.seq_lens) == 0:
            return original(self, batch, *args, **kwargs)
        COUNTER += 1
        control = ROOT/"control.json"
        label = json.loads(control.read_text())["label"] if control.exists() else "startup"
        # Copy the ACTUAL device-side committed prefix, not the overlap scheduler's
        # conservative CPU planning lengths. This is outside the timed root.
        prefix_cpu = torch.empty_like(batch.seq_lens, device="cpu", pin_memory=True)
        prefix_cpu.copy_(batch.seq_lens, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        ACTIVE = {"cycle": COUNTER, "label": label, "batch_size": len(batch.seq_lens),
                  "block_size": int(self.block_size), "prefix_cpu": prefix_cpu, "metadata_done": done,
                  "spans": [], "native": {}}
        current = ACTIVE
        root = begin("decode_cycle")
        try:
            result = original(self, batch, *args, **kwargs)
            end("decode_cycle", root)
            current["target_graph"] = bool(result.can_run_cuda_graph)
            PENDING.append(current)
            return result
        finally:
            ACTIVE = None
            drain()

    cls.forward_batch_generation = cycle
    atexit.register(lambda: drain(force=True))
