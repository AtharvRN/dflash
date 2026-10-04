"""No-GPU tests for scheduler mapping, manual selection, and CPU fallback."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.gpu_runtime import configure_gpu_runtime


class GpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.inventory = "<nvidia_smi_log><gpu><minor_number>6</minor_number><uuid>GPU-aabbccdd-0011</uuid></gpu></nvidia_smi_log>"
        self.occupancy = "4, 24, 0\n"
        query_patch = mock.patch("scripts.gpu_runtime.subprocess.check_output", side_effect=
            lambda command, **kwargs: self.inventory if command == ["nvidia-smi", "-q", "-x"] else self.occupancy)
        self.query = query_patch.start()
        self.addCleanup(query_patch.stop)

    def slurm(self, visible="0", allocated="6"):
        os.environ.update(SLURM_JOB_ID="123", SLURM_JOB_GPUS=allocated, CUDA_VISIBLE_DEVICES=visible)

    def test_remapped_ordinal_preserved_and_physical_allocation_queried(self):
        self.slurm()
        original = dict(os.environ)
        result = configure_gpu_runtime(use_visible_gpu=True)
        self.assertEqual(dict(os.environ), original)
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["allocated_device_minor"], 6)
        self.assertIsNone(result["physical_gpu_index"])
        self.assertEqual(result["nvidia_smi_index"], 4)
        self.assertEqual(result["cuda_visible_devices"], "0")
        self.assertIn("--id=GPU-aabbccdd-0011", self.query.call_args.args[0])
        self.assertNotIn("--id=0", self.query.call_args.args[0])

    def test_uuid_preserved(self):
        self.slurm(visible="GPU-aabbccdd-0011")
        result = configure_gpu_runtime(use_visible_gpu=True)
        self.assertEqual(result["cuda_visible_devices"], "GPU-aabbccdd-0011")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-aabbccdd-0011")

    def test_manual_selection_outside_slurm(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        result = configure_gpu_runtime(6)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(result["inherited_cuda_visible_devices"], "1")
        self.assertEqual(result["mode"], "manual")

    def test_manual_selection_forbidden_inside_slurm(self):
        self.slurm()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            configure_gpu_runtime(5)
        self.query.assert_not_called()
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_mutually_exclusive_modes(self):
        self.slurm()
        with self.assertRaisesRegex(ValueError, "not both"):
            configure_gpu_runtime(6, True)
        self.query.assert_not_called()

    def test_cpu_default_does_not_query_gpu_even_in_slurm(self):
        self.slurm()
        result = configure_gpu_runtime()
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.query.assert_not_called()

    def test_collector_requires_gpu(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            configure_gpu_runtime(require_gpu=True)
        self.query.assert_not_called()

    def test_missing_or_invalid_job_fails_closed(self):
        for job_id in (None, "", "not-a-job"):
            with self.subTest(job_id=job_id):
                self.slurm()
                if job_id is None:
                    del os.environ["SLURM_JOB_ID"]
                else:
                    os.environ["SLURM_JOB_ID"] = job_id
                with self.assertRaisesRegex(ValueError, "SLURM_JOB_ID"):
                    configure_gpu_runtime(use_visible_gpu=True)
        self.query.assert_not_called()

    def test_missing_multiple_and_invalid_physical_allocations_fail_closed(self):
        for allocation in ("", "0,6", "0-7", "8", "GPU-aabb", "6(S:0)"):
            with self.subTest(allocation=allocation):
                self.slurm(allocated=allocation)
                with self.assertRaisesRegex(ValueError, "SLURM_JOB_GPUS"):
                    configure_gpu_runtime(use_visible_gpu=True)
        self.query.assert_not_called()

    def test_missing_multiple_and_invalid_visibility_fail_closed(self):
        for visibility in ("", "0,1", "-1", "NoDevFiles", "0 ", "MIG-aabb"):
            with self.subTest(visibility=visibility):
                self.slurm(visible=visibility)
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    configure_gpu_runtime(use_visible_gpu=True)
        self.slurm()
        del os.environ["CUDA_VISIBLE_DEVICES"]
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
            configure_gpu_runtime(use_visible_gpu=True)
        self.query.assert_not_called()

    def test_busy_gpu_does_not_change_visibility(self):
        for measurement in ("4, 1025, 0", "4, 24, 11"):
            with self.subTest(measurement=measurement):
                self.slurm()
                self.occupancy = measurement
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    configure_gpu_runtime(use_visible_gpu=True)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_malformed_occupancy_fails_closed(self):
        for measurement in ("4, N/A, N/A", "4, 24, 0\n5, 24, 0", "4, 24, 101", "4, -1, 0"):
            with self.subTest(measurement=measurement):
                self.slurm()
                self.occupancy = measurement
                with self.assertRaises(RuntimeError):
                    configure_gpu_runtime(use_visible_gpu=True)

    def test_missing_device_minor_fails_closed(self):
        self.slurm()
        self.inventory = self.inventory.replace("<minor_number>6", "<minor_number>5")
        with self.assertRaisesRegex(RuntimeError, "device minor"):
            configure_gpu_runtime(use_visible_gpu=True)

    def test_duplicate_minor_and_malformed_xml_fail_closed(self):
        self.slurm()
        for inventory in ("<broken", "<nvidia_smi_log><gpu><minor_number>6</minor_number></gpu><gpu><minor_number>6</minor_number></gpu></nvidia_smi_log>"):
            with self.subTest(inventory=inventory):
                self.inventory = inventory
                with self.assertRaises(RuntimeError):
                    configure_gpu_runtime(use_visible_gpu=True)

    def test_query_failure_propagates_without_changing_visibility(self):
        self.slurm()
        self.query.side_effect = OSError("nvidia-smi unavailable")
        with self.assertRaises(OSError):
            configure_gpu_runtime(use_visible_gpu=True)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_collector_preflight_skips_gpu_validation_and_query(self):
        from scripts.collect_rejected_trace import main
        # No Slurm env and no GPU query are necessary for a read-only preflight.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["collect_rejected_trace.py", "--use-visible-gpu", "--preflight"]
            for flag in ("manifest", "pilot-manifest", "split-dir", "eval-cache", "models", "output", "backup"):
                argv += ["--"+flag, str(root/flag)]
            with mock.patch("sys.argv", argv), \
                 mock.patch("scripts.collect_rejected_trace.select_training_groups", return_value=[]), \
                 mock.patch("scripts.collect_rejected_trace.select_groups", return_value=[]), \
                 mock.patch("scripts.collect_rejected_trace.load_reference", return_value=({}, "refhash")), \
                 mock.patch("scripts.collect_rejected_trace.validate_models"), \
                 mock.patch("scripts.collect_rejected_trace.configure_gpu_runtime") as configure, \
                 mock.patch.object(Path, "read_text", side_effect=["{}", '{"train_prompt_ids":[]}', '{"val_prompt_ids":[]}']), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                main()
            configure.assert_not_called()
            self.query.assert_not_called()
            self.assertIn('"gpu_accessed": false', output.getvalue())
            self.assertFalse((root/"output").exists())


if __name__ == "__main__":
    unittest.main()
