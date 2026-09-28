from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest

from scripts.worker_diagnostics import DiagnosticSpawnContext, exit_details, process_resources, worker_event


def normal_task():
    worker_event("test_task", task={"prompt_id": 123})
    return 42


def wait_for_termination(ready):
    worker_event("test_waiting", task={"prompt_id": 456})
    ready.set()
    signal.pause()


def abrupt_sigkill():
    worker_event("test_before_sigkill")
    os.kill(os.getpid(), signal.SIGKILL)


def announce_ready(queue, barrier):
    queue.put(os.getpid())
    barrier.wait(timeout=20)


def uncaught_error():
    raise RuntimeError("intentional disposable diagnostic test")


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class WorkerDiagnosticsTests(unittest.TestCase):
    def test_exit_code_labels_do_not_infer_oom(self):
        self.assertEqual(exit_details(None), {"exitcode": None})
        self.assertEqual(exit_details(7), {"exitcode": 7})
        self.assertEqual(exit_details(-9), {"exitcode": -9, "termination_signal": 9,
                                          "termination_signal_name": "SIGKILL"})

    def test_read_only_linux_resource_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc, cg = root/"proc", root/"cgroup"
            (proc/"123").mkdir(parents=True)
            (cg/"test.slice").mkdir(parents=True)
            (proc/"123/status").write_text("Name:\ttest\nVmRSS:\t256 kB\nSecret:\tignored\n")
            (proc/"meminfo").write_text("MemAvailable: 1000 kB\n")
            (proc/"123/cgroup").write_text("0::/test.slice\n")
            (cg/"test.slice/memory.events").write_text("oom 2\noom_kill 1\n")
            snapshot = process_resources(123, proc, cg)
            self.assertEqual(snapshot["status"], {"Name": "test", "VmRSS": "256 kB"})
            self.assertEqual(snapshot["cgroup_v2"]["memory.events"], "oom 2\noom_kill 1")
            self.assertIn("unavailable", snapshot["cgroup_v2"]["memory.max"])

    def test_normal_executor_does_not_change_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with ProcessPoolExecutor(max_workers=1, mp_context=DiagnosticSpawnContext(root)) as pool:
                self.assertEqual(pool.submit(normal_task).result(timeout=20), 42)
            lifecycle = events(root/"parent_lifecycle.jsonl")
            self.assertTrue(any(r["event"] == "joined" and r["exitcode"] == 0 for r in lifecycle))
            worker = events(next(root.glob("worker_*.jsonl")))
            self.assertTrue(any(r["event"] == "test_task" and r["task"]["prompt_id"] == 123 for r in worker))

    def test_uncaught_exception_has_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = DiagnosticSpawnContext(root).Process(target=uncaught_error)
            p.start()
            p.join(timeout=20)
            self.assertEqual(p.exitcode, 1)
            record = next(r for r in events(root/f"worker_{p.pid}.jsonl") if r["event"] == "uncaught_exception")
            self.assertIn("RuntimeError", record["traceback"])

    def test_sigterm_is_logged_and_preserves_signal_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctx = DiagnosticSpawnContext(root)
            ready = ctx.Event()
            p = ctx.Process(target=wait_for_termination, args=(ready,))
            p.start()
            try:
                self.assertTrue(ready.wait(timeout=20))
                p.terminate()
                p.join(timeout=20)
                self.assertEqual(p.exitcode, -signal.SIGTERM)
                before = next(r for r in events(root/"parent_lifecycle.jsonl") if r["event"] == "terminate_requested")
                self.assertIsNone(before["exitcode"])
                received = next(r for r in events(root/f"worker_{p.pid}.jsonl") if r["event"] == "signal_received")
                self.assertEqual(received["number"], signal.SIGTERM)
                self.assertEqual(received["task"]["prompt_id"], 456)
                self.assertIn("wait_for_termination", (root/f"worker_{p.pid}.log").read_text())
            finally:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=10)

    def test_sigkill_records_dead_worker_before_pool_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctx = DiagnosticSpawnContext(root)
            ready, barrier = ctx.Queue(), ctx.Barrier(2)
            with ProcessPoolExecutor(max_workers=2, mp_context=ctx, initializer=announce_ready,
                                     initargs=(ready, barrier)) as pool:
                initial = [pool.submit(normal_task) for _ in range(2)]
                self.assertNotEqual(ready.get(timeout=20), ready.get(timeout=20))
                self.assertEqual([f.result(timeout=20) for f in initial], [42, 42])
                with self.assertRaises(BrokenProcessPool):
                    pool.submit(abrupt_sigkill).result(timeout=20)
            ready.close()
            ready.join_thread()
            lifecycle = events(root/"parent_lifecycle.jsonl")
            before = [r for r in lifecycle if r["event"] == "terminate_requested"]
            # Python versions may retry cleanup; preserve the first observation
            # for each worker rather than assuming one terminate call per PID.
            first = {}
            for record in before:
                first.setdefault(record["worker_pid"], record)
            self.assertEqual(len(first), 2)
            self.assertEqual({r["exitcode"] for r in first.values()}, {None, -signal.SIGKILL})
            dead = next(r for r in first.values() if r["exitcode"] is not None)
            self.assertEqual(dead["termination_signal_name"], "SIGKILL")
            self.assertTrue(any(r["event"] == "joined" and r["exitcode"] == -signal.SIGKILL for r in lifecycle))
            self.assertTrue(any(r["event"] == "joined" and r["exitcode"] == -signal.SIGTERM for r in lifecycle))


if __name__ == "__main__":
    unittest.main()
