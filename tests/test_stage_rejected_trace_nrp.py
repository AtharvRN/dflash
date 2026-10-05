import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage_rejected_trace_nrp import main


class StagingTests(unittest.TestCase):
    def test_copy_hashes_inputs_materializes_models_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "local"
            files = ["manifests/qwen3_4b_instruct_100k_messages.jsonl",
                     "runs/prefusion_pilot_20260915/cache/manifest.json",
                     "splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719/train_prompt_ids.json",
                     "runs/policy_granularity_20260927/cache/COMPLETE.json"]
            for name in files:
                p = source / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("source-evidence")
            weights = root / "weights"
            weights.write_bytes(b"example weights")
            model = root / "revision"
            model.mkdir()
            (model / "model.safetensors").symlink_to(weights)
            registry = root / "models.json"
            registry.write_text(json.dumps({"target": {"revision": "revision", "path": str(model)}}))
            args = ["stage", "--source", str(source), "--destination", str(destination), "--models", str(registry)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            for name in files:
                self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())
            copied = destination / "models/target/revision/model.safetensors"
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_bytes(), weights.read_bytes())
            self.assertEqual(len(json.loads((destination / "staging.json").read_text())["input_sha256"]), 4)


if __name__ == "__main__":
    unittest.main()
