"""Fail-closed shell prechecks; these never access a GPU or run the pipeline."""
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_rejected_trace_predictor.sh"
SBATCH = ROOT / "scripts/run_rejected_trace_slurm.sbatch"


class SlurmLauncherTests(unittest.TestCase):
    def launch(self, script=LAUNCHER, **values):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("SLURM_", "DFLASH_", "KUBERNETES_", "NVIDIA_")) and key not in
               {"GPU", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "RUN_ID", "MODE", "WORKERS", "EXPECTED_COMMIT"}}
        env.update(values)
        return subprocess.run(["bash", str(script)], env=env, cwd=ROOT,
                              capture_output=True, text=True, timeout=10)

    def test_shell_syntax(self):
        for script in (LAUNCHER, SBATCH):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_manual_requires_explicit_gpu(self):
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose an idle GPU explicitly", result.stderr)

    def test_slurm_rejects_manual_override(self):
        result = self.launch(SLURM_JOB_ID="123", SLURM_JOB_GPUS="5",
                             CUDA_VISIBLE_DEVICES="0", GPU="5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not set GPU", result.stderr)

    def test_slurm_requires_single_allocation_and_visibility(self):
        for allocated, visible in (("", "0"), ("4,5", "0"), ("5", ""), ("5", "0,1")):
            with self.subTest(allocated=allocated, visible=visible):
                result = self.launch(SLURM_JOB_ID="123", SLURM_JOB_GPUS=allocated,
                                     CUDA_VISIBLE_DEVICES=visible)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Slurm requires one allocated GPU", result.stderr)

    def test_fixture_strips_ambient_cluster_and_path_overrides(self):
        ambient = {
            "KUBERNETES_SERVICE_HOST": "10.0.0.1", "DFLASH_POD_UID": "ambient-pod",
            "NVIDIA_VISIBLE_DEVICES": "all", "CUDA_VISIBLE_DEVICES": "0",
            "DFLASH_TASK_ROOT": "/never-use-this-ambient-root",
            "DFLASH_DATA_ROOT": "/never-use-this-ambient-data",
            "DFLASH_PYTHON": "/never-use-this-ambient-python",
            "DFLASH_MODELS": "/never-use-this-ambient-models",
            "DFLASH_EVAL_CACHE": "/never-use-this-ambient-cache",
            "SLURM_JOB_ID": "123", "GPU": "5", "MODE": "invalid-ambient-mode",
        }
        with mock.patch.dict(os.environ, ambient):
            result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose an idle GPU explicitly", result.stderr)
        self.assertNotIn("Kubernetes requires", result.stderr)

    def test_kubernetes_requires_host_and_pod_identity_together(self):
        for values in ({"KUBERNETES_SERVICE_HOST": "10.0.0.1"},
                       {"DFLASH_POD_UID": "pod-123"},
                       {"KUBERNETES_SERVICE_HOST": "10.0.0.1", "DFLASH_POD_UID": ""},
                       {"KUBERNETES_SERVICE_HOST": "", "DFLASH_POD_UID": "pod-123"}):
            with self.subTest(values=values):
                result = self.launch(**values)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Kubernetes requires its pod UID", result.stderr)
                self.assertNotIn("df:", result.stderr)

    def test_kubernetes_rejects_manual_and_slurm_allocation_overrides(self):
        for overrides in ({"GPU": "0"}, {"SLURM_JOB_ID": "123"},
                          {"GPU": "5", "SLURM_JOB_ID": "123", "SLURM_JOB_GPUS": "5"}):
            with self.subTest(overrides=overrides):
                result = self.launch(KUBERNETES_SERVICE_HOST="10.0.0.1",
                                     DFLASH_POD_UID="pod-123", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("do not set GPU or Slurm overrides", result.stderr)
                self.assertNotIn("df:", result.stderr)

    def test_run_id_cannot_escape_destination(self):
        result = self.launch(RUN_ID="../different-run", GPU="5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid run ID", result.stderr)

    def test_batch_script_requires_slurm(self):
        result = self.launch(script=SBATCH)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be submitted with sbatch", result.stderr)

    def test_batch_script_requires_pin(self):
        result = self.launch(script=SBATCH, SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=str(ROOT))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set EXPECTED_COMMIT", result.stderr)


if __name__ == "__main__":
    unittest.main()
