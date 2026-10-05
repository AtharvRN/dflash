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


class ContainerGpuRuntimeTests(unittest.TestCase):
    UUID = "GPU-12345678-1234-1234-1234-123456789abc"
    OTHER_UUID = "GPU-abcdef12-1234-1234-1234-123456789abc"

    def setUp(self):
        environment = mock.patch.dict(os.environ, {
            "KUBERNETES_SERVICE_HOST": "10.0.0.1", "DFLASH_POD_UID": "test-pod-uid",
            "NVIDIA_VISIBLE_DEVICES": self.UUID,
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.inventory = self.UUID+"\n"
        self.occupancy = "6, 24, 0\n"
        patch = mock.patch("scripts.gpu_runtime.subprocess.check_output", side_effect=
            lambda command, **kwargs: self.inventory if "--query-gpu=uuid" in command else self.occupancy)
        self.query = patch.start()
        self.addCleanup(patch.stop)

    def test_single_allocation_uuid_selected_with_unset_visibility(self):
        result = configure_gpu_runtime(use_container_gpu=True, require_gpu=True)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], self.UUID)
        self.assertEqual(result["mode"], "kubernetes_container")
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["container_allocated_gpu_uuid"], self.UUID)
        self.assertEqual(result["kubernetes_pod_uid"], "test-pod-uid")
        self.assertEqual(result["nvidia_smi_index"], 6)
        self.assertIsNone(result["physical_gpu_index"])
        self.assertIn("--id="+self.UUID, self.query.call_args.args[0])

    def test_sole_ordinal_normalized_to_proven_allocated_uuid(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        result = configure_gpu_runtime(use_container_gpu=True)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], self.UUID)
        self.assertEqual(result["inherited_cuda_visible_devices"], "0")

    def test_matching_uuid_preserved(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.UUID
        original = dict(os.environ)
        configure_gpu_runtime(use_container_gpu=True)
        self.assertEqual(dict(os.environ), original)

    def test_requires_kubernetes_signal_and_downward_api_identity(self):
        for key in ("KUBERNETES_SERVICE_HOST", "DFLASH_POD_UID"):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: ""}):
                with self.assertRaisesRegex(ValueError, "Kubernetes.*DFLASH_POD_UID"):
                    configure_gpu_runtime(use_container_gpu=True)
        self.query.assert_not_called()

    def test_refuses_multiple_ordinal_mig_or_all_allocation(self):
        for allocation in ("all", "none", "0", "GPU-short", "MIG-"+self.UUID,
                           self.UUID+","+self.OTHER_UUID, " "+self.UUID):
            with self.subTest(allocation=allocation), mock.patch.dict(os.environ, {"NVIDIA_VISIBLE_DEVICES": allocation}):
                with self.assertRaisesRegex(ValueError, "NVIDIA_VISIBLE_DEVICES"):
                    configure_gpu_runtime(use_container_gpu=True)
        self.query.assert_not_called()

    def test_runtime_inventory_discovery_without_legacy_uuid(self):
        for allocation in (None, "", "void"):
            with self.subTest(allocation=allocation), mock.patch.dict(os.environ):
                if allocation is None:
                    os.environ.pop("NVIDIA_VISIBLE_DEVICES", None)
                else:
                    os.environ["NVIDIA_VISIBLE_DEVICES"] = allocation
                result = configure_gpu_runtime(use_container_gpu=True, require_gpu=True)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], self.UUID)
                self.assertEqual(result["container_uuid_source"], "single_visible_inventory")
                self.assertEqual(result["nvidia_visible_devices"], allocation)

    def test_discovery_requires_one_valid_full_device(self):
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
        for inventory in ("", "MIG-"+self.UUID, "GPU-short", self.UUID+"\n"+self.OTHER_UUID):
            with self.subTest(inventory=inventory):
                self.inventory = inventory
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    configure_gpu_runtime(use_container_gpu=True)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_discovery_preserves_explicit_cuda_restrictions(self):
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
        for visible in ("", "-1", "1", "0,1", self.OTHER_UUID):
            with self.subTest(visible=visible), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    configure_gpu_runtime(use_container_gpu=True)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible)

    def test_refuses_disabled_multiple_foreign_or_nonzero_visibility(self):
        for visible in ("", "-1", "1", "0,1", "all", self.OTHER_UUID, self.UUID+",0"):
            with self.subTest(visible=visible), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    configure_gpu_runtime(use_container_gpu=True)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible)
        self.query.assert_not_called()

    def test_inventory_must_contain_only_the_allocated_gpu(self):
        for inventory in ("", self.OTHER_UUID, self.UUID+"\n"+self.OTHER_UUID, self.UUID+"\n"+self.UUID):
            with self.subTest(inventory=inventory):
                self.inventory = inventory
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    configure_gpu_runtime(use_container_gpu=True)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_occupancy_failure_does_not_change_visibility(self):
        for occupancy in ("6, 1025, 0", "6, 24, 11", "N/A", "6, 24, 0\n7, 24, 0"):
            with self.subTest(occupancy=occupancy):
                self.occupancy = occupancy
                with self.assertRaises(RuntimeError):
                    configure_gpu_runtime(use_container_gpu=True)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_query_failure_does_not_change_visibility(self):
        self.query.side_effect = OSError("nvidia-smi unavailable")
        with self.assertRaises(OSError):
            configure_gpu_runtime(use_container_gpu=True)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_modes_mutually_exclusive(self):
        for kwargs in ({"gpu": 0}, {"use_visible_gpu": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Choose"):
                configure_gpu_runtime(use_container_gpu=True, **kwargs)
        self.query.assert_not_called()

    def test_manual_override_forbidden_in_kubernetes(self):
        with self.assertRaisesRegex(ValueError, "forbidden inside Kubernetes"):
            configure_gpu_runtime(gpu=0)
        self.query.assert_not_called()

    def test_slurm_and_kubernetes_allocation_cannot_be_combined(self):
        os.environ["SLURM_JOB_ID"] = "123"
        with self.assertRaisesRegex(ValueError, "Slurm allocation"):
            configure_gpu_runtime(use_container_gpu=True)
        self.query.assert_not_called()

    def test_cpu_default_never_queries_or_uses_allocated_gpu(self):
        result = configure_gpu_runtime()
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.query.assert_not_called()

    def test_container_preflight_remains_read_only_without_allocation(self):
        from scripts.collect_rejected_trace import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["collect_rejected_trace.py", "--use-container-gpu", "--preflight"]
            for flag in ("manifest", "pilot-manifest", "split-dir", "eval-cache", "models", "output", "backup"):
                argv += ["--"+flag, str(root/flag)]
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch("sys.argv", argv), \
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
            self.assertFalse((root/"backup").exists())


if __name__ == "__main__":
    unittest.main()
