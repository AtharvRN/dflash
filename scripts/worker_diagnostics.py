"""Task-scoped diagnostics for spawned collection workers; no polling service.

Record process exit status BEFORE executor cleanup sends SIGTERM to peers.
SIGKILL cannot be caught in a worker: its parent-side exit status is evidence of
the signal, not evidence of OOM or of who sent it.
"""
from __future__ import annotations

from datetime import datetime, timezone
import faulthandler
import json
from multiprocessing.context import SpawnContext, SpawnProcess
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback


_EVENT_PATH = None
_CRASH_STREAM = None
_ACTIVE_TASK = None


def exit_details(code):
    result = {"exitcode": code}
    if code is not None and code < 0:
        number = -code
        result["termination_signal"] = number
        try:
            result["termination_signal_name"] = signal.Signals(number).name
        except ValueError:
            result["termination_signal_name"] = "unknown"
    return result


def _read(path):
    try:
        return path.read_text().strip()
    except OSError as exc:
        return {"unavailable": str(exc)}


def process_resources(pid=None, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Best-effort Linux RSS/limits and cgroup counters, without privileged logs."""
    pid = os.getpid() if pid is None else pid
    result = {"pid": pid}
    status = _read(proc_root/str(pid)/"status")
    keys = {"Name", "State", "Pid", "PPid", "VmPeak", "VmSize", "VmHWM", "VmRSS", "VmSwap", "Threads"}
    result["status"] = ({k: v.strip() for line in status.splitlines() if ":" in line
                         for k, v in [line.split(":", 1)] if k in keys}
                        if isinstance(status, str) else status)
    memory = _read(proc_root/"meminfo")
    result["host_memory"] = ({k: v.strip() for line in memory.splitlines() if ":" in line
                              for k, v in [line.split(":", 1)]
                              if k in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}}
                             if isinstance(memory, str) else memory)
    membership = _read(proc_root/str(pid)/"cgroup")
    result["cgroup_membership"] = membership
    if isinstance(membership, str):
        for line in membership.splitlines():
            if not line.startswith("0::"):
                continue
            directory = cgroup_root/line[3:].lstrip("/")
            if not directory.resolve().is_relative_to(cgroup_root.resolve()):
                continue
            names = ("memory.current", "memory.max", "memory.peak", "memory.events",
                     "memory.events.local", "pids.current", "pids.max", "pids.events")
            result["cgroup_v2"] = {"path": str(directory), **{n: _read(directory/n) for n in names}}
    if pid == os.getpid():
        result["resource_limits"] = {name: list(resource.getrlimit(getattr(resource, name)))
                                     for name in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_RSS", "RLIMIT_NOFILE")}
    return result


def append_event(path, event, **payload):
    """One append per event; failures must not obscure the original exception."""
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                  "unix_time": time.time(), "observer_pid": os.getpid(), **payload}
        with path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True)+"\n")
            stream.flush()
    except Exception as exc:
        print(f"DIAGNOSTIC_WRITE_FAILED {path}: {exc!r}", file=sys.stderr, flush=True)


def worker_event(event, task=None, **payload):
    global _ACTIVE_TASK
    if task is not None:
        _ACTIVE_TASK = task
    append_event(_EVENT_PATH, event, task=_ACTIVE_TASK, **payload)


def controller_event(root, event, processes=(), **payload):
    if root is None:
        return
    observed = []
    for process in processes:
        try:
            observed.append({"pid": process.pid, **exit_details(process.exitcode),
                             "resources": process_resources(process.pid)})
        except (ValueError, OSError) as exc:
            observed.append({"unavailable": repr(exc)})
    append_event(Path(root)/"controller.jsonl", event, processes=observed,
                 controller_resources=process_resources(), **payload)


def _worker_signal(number, frame):
    # Do not swallow a termination request or convert its observed exit status
    # into an ordinary Python error code. SIGKILL is deliberately not catchable.
    worker_event("signal_received", number=number, name=signal.Signals(number).name)
    if _CRASH_STREAM is not None:
        faulthandler.dump_traceback(file=_CRASH_STREAM, all_threads=True)
        _CRASH_STREAM.flush()
    signal.signal(number, signal.SIG_DFL)
    os.kill(os.getpid(), number)


class DiagnosticSpawnProcess(SpawnProcess):
    def __init__(self, *args, diagnostics_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics_root = Path(diagnostics_root)

    def _parent_event(self, event, **payload):
        append_event(self.diagnostics_root/"parent_lifecycle.jsonl", event,
                     worker_pid=self.pid, **exit_details(self.exitcode), **payload)

    def start(self):
        super().start()
        self._parent_event("spawned")

    def terminate(self):
        # ProcessPoolExecutor terminates ALL peers when one worker dies. This
        # pre-cleanup record distinguishes an already-dead worker from peers.
        self._parent_event("terminate_requested", resources=process_resources(self.pid))
        super().terminate()

    def kill(self):
        self._parent_event("kill_requested", resources=process_resources(self.pid))
        super().kill()

    def join(self, timeout=None):
        super().join(timeout)
        self._parent_event("joined")

    def run(self):
        global _EVENT_PATH, _CRASH_STREAM
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        prefix = self.diagnostics_root/f"worker_{os.getpid()}"
        _EVENT_PATH = prefix.with_suffix(".jsonl")
        with prefix.with_suffix(".log").open("a", buffering=1) as stream:
            _CRASH_STREAM = stream
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            for output in (sys.stdout, sys.stderr):
                if hasattr(output, "reconfigure"):
                    output.reconfigure(line_buffering=True)
            faulthandler.enable(file=stream, all_threads=True)
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(number, _worker_signal)
            worker_event("worker_started", parent_pid=os.getppid(), hostname=socket.gethostname(),
                         resources=process_resources())
            try:
                super().run()
            except BaseException as exc:
                worker_event("uncaught_exception", error=repr(exc), traceback=traceback.format_exc(),
                             resources=process_resources())
                raise
            else:
                worker_event("worker_returned", resources=process_resources())
            finally:
                faulthandler.disable()
                _CRASH_STREAM = None


class DiagnosticSpawnContext(SpawnContext):
    """Standard spawn context with lifecycle logging, not a new scheduler."""
    def __init__(self, root):
        self.diagnostics_root = Path(root)

    def Process(self, *args, **kwargs):
        return DiagnosticSpawnProcess(*args, diagnostics_root=self.diagnostics_root, **kwargs)
