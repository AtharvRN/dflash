import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from dflash.context_attention import ContextAcceptancePredictor, ResidualContextAcceptancePredictor, acceptance_nll, acceptance_survival, choose_budget
from scripts.prepare_context_attention_cache import materialize, partition_staged, scan_shard, split_rows, validate_context_masks, validate_feature_kind
from scripts.train_context_attention import calibrate, proxy_metrics, make_model


class ContextAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def model(self, queries=15):
        return ContextAcceptancePredictor(input_dim=8, context_window=6, num_queries=queries,
                    num_layers=2, num_heads=2, ff_width=16, dropout=0).eval()

    def test_both_output_shapes_gradients_and_reload(self):
        for queries in (1, 15):
            model = self.model(queries)
            x = torch.randn(3, 6, 8, requires_grad=True)
            logits = model(x, torch.ones(3, 6))
            self.assertEqual(logits.shape, (3, 15))
            acceptance_nll(logits, torch.tensor([0, 7, 15])).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(model.queries.grad.abs().sum()), 0)
            clone = self.model(queries)
            clone.load_state_dict(model.state_dict())
            torch.testing.assert_close(logits, clone(x.detach(), torch.ones(3, 6)))

    def test_masked_features_and_padding_do_not_change_prediction(self):
        for queries in (1, 15):
            model = self.model(queries)
            content = torch.randn(2, 3, 8)
            expected = model(content, torch.ones(2, 3))
            left = torch.cat([torch.full_like(content, float("nan")), content], 1)
            right = torch.cat([content, torch.full_like(content, 1e10)], 1)
            torch.testing.assert_close(expected, model(left, torch.tensor([[0,0,0,1,1,1]]*2)), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(expected, model(right, torch.tensor([[1,1,1,0,0,0]]*2)), atol=1e-6, rtol=1e-5)

    def test_no_cross_request_mixing_and_context_order_matters(self):
        model = self.model()
        x = torch.randn(3, 6, 8)
        batched = model(x, torch.ones(3, 6))
        alone = torch.cat([model(row[None], torch.ones(1, 6)) for row in x])
        torch.testing.assert_close(batched, alone, atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(batched, model(x.flip(1), torch.ones(3, 6)), atol=1e-7, rtol=1e-7))

    def test_all_masked_rejected_and_anchor_optional(self):
        model = self.model()
        x = torch.randn(1, 6, 8)
        with self.assertRaises(ValueError):
            model(x, torch.zeros(1, 6))
        with self.assertRaises(ValueError):
            model(x, torch.full((1, 6), .5))
        normal = model(x, torch.ones(1, 6))
        anchored = model(x, torch.ones(1, 6), torch.randn(1, 8))
        self.assertFalse(torch.allclose(normal, anchored))

    def test_likelihood_all_lengths_and_censoring_gradients(self):
        z = torch.zeros(16, 15, requires_grad=True)
        y = torch.arange(16)
        nll = acceptance_nll(z, y, "none")
        torch.testing.assert_close(nll, torch.minimum(y+1, torch.tensor(15)).float()*math.log(2))
        nll.sum().backward()
        for i in range(16):
            torch.testing.assert_close(z.grad[i, :i], torch.full_like(z.grad[i, :i], -.5))
            if i < 15:
                self.assertEqual(float(z.grad[i, i]), .5)
                self.assertEqual(float(z.grad[i, i+1:].abs().sum()), 0)
        self.assertEqual(float(z.grad[15].sum()), -7.5)

    def test_extreme_logits_monotonicity_and_budget(self):
        logits = torch.tensor([[1000., -1000., 1000.], [-1000., 1000., -1000.]])
        self.assertTrue(torch.isfinite(acceptance_nll(logits, torch.tensor([3, 0]))))
        s = acceptance_survival(logits)
        self.assertTrue((s[:, 1:] <= s[:, :-1]).all())
        budgets = [choose_budget(s, a) for a in (.5, .8, .95, 1.)]
        self.assertTrue(all((a <= b).all() for a, b in zip(budgets, budgets[1:])))
        self.assertTrue(all(((b >= 1) & (b <= 3)).all() for b in budgets))
        self.assertTrue((choose_budget(torch.zeros(2, 15), 1.0) == 15).all())
        with self.assertRaises(ValueError):
            acceptance_nll(logits, torch.tensor([3.5, 0.]))

    def test_tiny_overfit(self):
        for queries in (1, 15):
            model = self.model(queries).train()
            x = torch.randn(4, 6, 8)
            y = torch.tensor([0, 3, 8, 15])
            opt = torch.optim.Adam(model.parameters(), lr=.02)
            initial = float(acceptance_nll(model(x, torch.ones(4, 6)), y).detach())
            for _ in range(100):
                opt.zero_grad()
                loss = acceptance_nll(model(x, torch.ones(4, 6)), y)
                loss.backward()
                opt.step()
            self.assertLess(float(loss.detach()), initial*.2)

    def test_residual_starts_at_baseline_and_freezes_it(self):
        from types import SimpleNamespace
        args = SimpleNamespace(dropout=.4, layers=1, heads=2, ff_width=16)
        info = {"input_dim":8, "context_window":6, "num_slots":15}
        model = make_model("residual_attention", info, args).train()
        x, mask, y = torch.randn(4,6,8), torch.ones(4,6), torch.tensor([0,3,8,15])
        state = {k:v.clone() for k,v in model.baseline.state_dict().items()}
        torch.testing.assert_close(model(x,mask), model.baseline(x,mask), atol=0, rtol=0)
        self.assertFalse(model.baseline.training)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            opt.zero_grad()
            acceptance_nll(model(x,mask), y).backward()
            opt.step()
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        self.assertGreater(float(model.correction.queries.grad.abs().sum()), 0)
        for k,v in model.baseline.state_dict().items():
            torch.testing.assert_close(v, state[k], atol=0, rtol=0)
        model.eval()
        clone = make_model("residual_attention", info, args).eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x,mask), clone(x,mask), atol=0, rtol=0)

    def test_last_only_correction_ignores_earlier_features(self):
        from types import SimpleNamespace
        model = make_model("residual_last_only", {"input_dim":8,"context_window":6,"num_slots":15},
                           SimpleNamespace(dropout=0.,layers=1,heads=2,ff_width=16)).eval()
        torch.nn.init.normal_(model.correction.head.weight)
        x, mask = torch.randn(2,6,8), torch.tensor([[0,1,1,1,0,0],[1,1,1,1,1,1]])
        changed = x.clone()
        changed[0,:3] = torch.randn(3,8)*10
        changed[1,:5] = torch.randn(5,8)*10
        torch.testing.assert_close(model(x,mask), model(changed,mask), atol=0, rtol=0)


class ContextDataTest(unittest.TestCase):
    def test_reject_contradictory_features_and_interior_mask_holes(self):
        validate_feature_kind({"predraft_feature_kind": "fused"})
        with self.assertRaises(ValueError):
            validate_feature_kind({"predraft_feature_kind": "raw-target"})
        validate_context_masks(np.array([[0,1,1], [1,1,0]]))
        for mask in ([[1,0,1]], [[0,0,0]], [[1,.5,0]]):
            with self.assertRaises(ValueError):
                validate_context_masks(np.array(mask))

    def test_prompt_splits_and_padded_tail_rows(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard"
            shard.mkdir()
            x = np.arange(5*3*8, dtype=np.float16).reshape(5, 3, 8)
            y = np.array([0, 15, 4, 99, 99])
            for name, array in {"features": x, "mask": np.ones((5,3), np.uint8), "accepted_len": y,
                                "survival": y[:, None] >= np.arange(1,16)}.items():
                np.save(shard / (name+".npy"), array)
            metadata = [{"manifest_index": p, "cycle_id": c, "accepted_draft_len": a}
                        for p,c,a in [(10,0,0), (20,0,15), (10,1,4)]]
            (shard / "metadata.jsonl").write_text("".join(json.dumps(r)+"\n" for r in metadata))
            rows = scan_shard((0, shard))
            self.assertEqual(len(rows), 3)
            train, val = split_rows(rows, {10}, {20}, None, 0)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            materialize([shard], train, root / "train", window=3, width=8, workers=1)
            np.testing.assert_array_equal(np.load(root/"train"/"features.npy"), x[[0,2]])
            partition_staged(root/"train", root/"partition", np.array([1]))
            np.testing.assert_array_equal(np.load(root/"partition"/"features.npy"), x[[2]])
            np.testing.assert_array_equal(np.load(root/"partition"/"row_index.npy"), train[[1]])
            with self.assertRaises(ValueError):
                split_rows(rows, {10,20}, {20}, None, 0)
            with self.assertRaises(ValueError):
                split_rows(rows, {10}, {30}, None, 0)

    def test_calibration_does_not_use_assessment_labels(self):
        s = torch.tensor([[.9,.7,.2], [.7,.2,.1]])
        y = torch.tensor([3, 1])
        selected = calibrate(s, y, .95)
        self.assertGreaterEqual(selected["retention"], .95)
        assessment_a = proxy_metrics(s, torch.tensor([0,0]), selected["alpha"])
        assessment_b = proxy_metrics(s, torch.tensor([3,3]), selected["alpha"])
        self.assertEqual(assessment_a["mean_budget"], assessment_b["mean_budget"])

    def test_training_cli_and_persistent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            rng = np.random.default_rng(1)
            info = {"format": "dflash_context_attention_cache_v1", "input_kind": "predraft_fused",
                    "input_dim": 8, "context_window": 4, "num_slots": 15}
            (cache / "manifest.json").write_text(json.dumps(info))
            for name, offset in (("train", 0), ("val", 100)):
                path = cache / name
                path.mkdir()
                np.save(path/"features.npy", rng.standard_normal((16,4,8)).astype(np.float16))
                np.save(path/"mask.npy", np.ones((16,4), np.uint8))
                np.save(path/"accepted_len.npy", np.arange(16))
                rows = np.stack([np.zeros(16), np.arange(16), np.arange(16)+offset,
                                 np.zeros(16), np.arange(16)], axis=1).astype(np.int64)
                np.save(path/"row_index.npy", rows)
            np.save(cache/"val"/"calibration.npy", np.arange(16)%2 == 0)
            script = Path(__file__).resolve().parents[1]/"scripts"/"train_context_attention.py"
            completed = subprocess.run([sys.executable, str(script), "--cache-dir", str(cache),
                "--output-dir", str(root/"run"), "--persistent-dir", str(root/"persistent"),
                "--device", "cpu", "--epochs", "1", "--layers", "1", "--heads", "2",
                "--ff-width", "16", "--batch-size", "8", "--eval-batch-size", "8", "--workers", "0",
                "--models", "last_mlp", "one_query", "position_queries", "residual_attention", "residual_last_only",
                "--selection-metric", "accept_ratio", "--backup-checkpoints", "final"],
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root/"run"/"summary.json").read_text())
            self.assertEqual(set(result), {"last_mlp", "one_query", "position_queries", "residual_attention", "residual_last_only"})
            self.assertEqual(result, json.loads((root/"persistent"/"summary.json").read_text()))
            for name, metrics in result.items():
                self.assertEqual(metrics["full_validation"]["rows"], 16)
                self.assertEqual(metrics["assessment"]["rows"], 8)
                self.assertTrue((root/"persistent"/name/metrics["best_checkpoint"]).exists())
            self.assertTrue(result["residual_attention"]["baseline_frozen_verified"])
            checkpoint = torch.load(root/"persistent"/"residual_attention"/result["residual_attention"]["best_checkpoint"], weights_only=False)
            initial = json.loads((root/"run"/"residual_attention"/"initial.json").read_text())
            self.assertLessEqual(checkpoint["selection_score"], -initial["calibration_proxy_policy"]["aggregate_accept_ratio"])


if __name__ == "__main__":
    unittest.main()
