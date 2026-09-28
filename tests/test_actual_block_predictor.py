import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dflash.block_response import (BlockResponseMLP, training_targets, choose_response_budget,
    penalty_candidates, response_metrics, calibration_curve, select_response_points)
from scripts.collect_policy_granularity import select_training_groups
from scripts.train_actual_block_predictor import weight_digest, bootstrap, load_training_pairs
from scripts.audit_block_headroom import sha256


class ActualBlockTests(unittest.TestCase):
    def test_architecture_count_and_bounds_without_monotonicity(self):
        model = BlockResponseMLP().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1478415)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.fill_(-10)
            model.net[-1].bias[3] = 10
            y = model(torch.randn(2, 2560))
        self.assertTrue(((y >= 0) & (y <= torch.arange(1, 16))).all())
        self.assertGreater(float(y[0, 3]), float(y[0, 14]))

    def test_labels_differ_only_by_supervision(self):
        a = np.zeros((2, 15), dtype=int)
        a[:, -1] = [8, 2]
        a[1, 4] = 5
        actual, clipped = training_targets(a, "actual"), training_targets(a, "clipped")
        self.assertEqual(actual[1, 4], 5)
        self.assertEqual(clipped[1, 4], 2)
        np.testing.assert_array_equal(actual[:, -1], clipped[:, -1])
        self.assertFalse(np.shares_memory(actual, a))
        with self.assertRaises(ValueError):
            training_targets(a+.2, "actual")

    def test_penalty_decision_matches_exhaustive_and_tie_rule(self):
        rng = np.random.default_rng(1)
        mu = rng.random((12, 15))*np.arange(1, 16)
        for lam in (0, .01, .35, 1, 10):
            actual = choose_response_budget(mu, {"kind": "penalty", "value": lam})
            expected = [max(range(1, 16), key=lambda d: (row[d-1]-lam*d, -d)) for row in mu]
            np.testing.assert_array_equal(actual, expected)
        tied = np.tile(np.arange(1, 16), (2, 1))
        np.testing.assert_array_equal(choose_response_budget(tied, {"kind": "penalty", "value": 1}), [1, 1])

    def test_hull_covers_all_open_interval_decisions(self):
        rng = np.random.default_rng(4)
        mu = rng.random((4, 15))*np.arange(1, 16)
        candidates = penalty_candidates(mu)
        covered = {tuple(choose_response_budget(mu, {"kind": "penalty", "value": float(v)})) for v in candidates}
        all_crossings = {0., 16.}
        for row in mu:
            for i in range(15):
                for j in range(i+1, 15):
                    q = (row[j]-row[i])/(j-i)
                    if 0 < q < 16:
                        all_crossings.add(q)
        edges = sorted(all_crossings)
        for left, right in zip(edges, edges[1:]):
            d = tuple(choose_response_budget(mu, {"kind": "penalty", "value": (left+right)/2}))
            self.assertIn(d, covered)

    def test_calibration_constraint_and_fallback(self):
        a = np.zeros((3, 15), dtype=int)
        a[:, -1] = [4, 8, 12]
        mu = np.zeros_like(a, dtype=float)
        curves = calibration_curve(mu, a)
        points = select_response_points(curves)
        self.assertEqual(points["0.96"]["setting"], {"kind": "full"})
        for target, point in points.items():
            self.assertGreaterEqual(point["retention"], float(target))
            feasible = [p["mean_budget"] for p in curves if p["retention"] >= float(target)]
            self.assertEqual(point["mean_budget"], min(feasible))

    def test_aggregate_metrics_and_paired_bootstrap(self):
        a = np.zeros((4, 15), dtype=int)
        a[:, -1] = [3, 5, 8, 10]
        a[0, 1] = 2
        a[1, 3] = 2
        d = np.array([2, 4, 15, 15])
        self.assertEqual(response_metrics(a, d)["aggregate_accept_ratio"], 22/36)
        result = bootstrap(a, np.array([1, 1, 2, 2]), {"actual_seed_913": d, "clipped_seed_913": d}, count=50)
        diff = result["paired_supervision_differences"]["actual_seed_913_minus_clipped_seed_913"]
        self.assertEqual(diff["ratio_ci95"], [0, 0])

    def test_matched_initialization_and_dropout_with_copy_scoring(self):
        states = []
        x = torch.ones(3, 2560)
        for _ in range(2):
            torch.manual_seed(913)
            model = BlockResponseMLP()
            before = weight_digest(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            for _ in range(2):
                loss = model(x).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                copy.deepcopy(model).eval()(x)
            states.append((before, weight_digest(model.state_dict())))
        self.assertEqual(states[0], states[1])

    def test_training_selection_disjoint_deduplicated_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"manifest_index": i, "messages": [{"role": "user", "content": str(i)}]} for i in range(7)]
            rows[6]["messages"] = rows[0]["messages"]
            rows[3]["messages"] = rows[2]["messages"]
            (root/"manifest.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (root/"train_prompt_ids.json").write_text(json.dumps({"train_prompt_ids": list(range(6))}))
            (root/"val_prompt_ids.json").write_text(json.dumps({"val_prompt_ids": [6]}))
            out = select_training_groups(root/"manifest.jsonl", root, 929, 4)
            self.assertEqual({r["manifest_index"] for r in out}, {1, 2, 4, 5})
            self.assertTrue(all(r["group"] == "train" for r in out))
            self.assertEqual(out, select_training_groups(root/"manifest.jsonl", root, 929, 4))
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                select_training_groups(root/"manifest.jsonl", root, 929, 5)

    def test_training_audit_alignment_tail_and_split_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root/"split"
            split.mkdir()
            (split/"train_prompt_ids.json").write_text(json.dumps({"train_prompt_ids": [10, 11]}))
            (split/"val_prompt_ids.json").write_text(json.dumps({"val_prompt_ids": [99]}))
            evaluation = {"models": {"target": "frozen", "draft": "frozen"}, "blocks": list(range(2, 17)),
                "dtype": "bfloat16", "attention": "sdpa", "tf32": False, "temperature": 0, "thinking": False,
                "states_per_prompt": 8, "max_new_tokens": 256, "max_prompt_tokens": 2048, "feature": "latest causal",
                "prompt_ids": [99], "prompt_content_hashes": {"99": "validation-content"}}
            config = {**evaluation, "collection_kind": "training", "training_rows": 1, "prompt_ids": [10, 11],
                "prompt_content_hashes": {"10": "training-content", "11": "more-training"},
                "prompt_groups": {"10": "train", "11": "train"},
                "input_hashes": {str(split/n): sha256(split/n) for n in ("train_prompt_ids.json", "val_prompt_ids.json")}}
            states = []
            for cycle, prefix in enumerate(([1, 2], [1, 2, 3])):
                states.append({"prompt_id": "10", "group": "train", "source": "nemotron", "cycle": cycle,
                    "prefix_token_ids": prefix, "prefix_length": len(prefix)-1,
                    "prefix_sha256": hashlib.sha256(np.array(prefix, dtype=np.int64).tobytes()).hexdigest(),
                    "eligible": True, "canonical_disagreements": 0,
                    "outcomes": {str(b): {"draft_ids": [7]*(b-1), "accepted": 0} for b in range(2, 17)}})
            (root/"prompt_10.json").write_text(json.dumps({"states": states}))
            np.save(root/"prompt_10_fused.npy", np.ones((2, 2560), dtype=np.float16))
            receipt = {"prompt_id": 10, "group": "train", "source": "nemotron", "states": 2,
                "files": {n: sha256(root/n) for n in ("prompt_10.json", "prompt_10_fused.npy")}}
            (root/"receipt_10.json").write_text(json.dumps(receipt))
            (root/"receipts.json").write_text(json.dumps([receipt]))
            summary = {"sample_complete": True, "states": 2, "eligible_states": 2, "prompts": 1,
                       "canonical_disagreements": 0}
            (root/"collection_summary.json").write_text(json.dumps(summary))
            def bind():
                (root/"config.json").write_text(json.dumps(config))
                (root/"COMPLETE.json").write_text(json.dumps({**summary,
                    "binding": {n: sha256(root/n) for n in ("config.json", "receipts.json", "collection_summary.json")}}))
            bind()
            x, actual, audit, index = load_training_pairs(root, split, evaluation)
            self.assertEqual(x.shape, (1, 2560))
            self.assertEqual(actual.shape, (1, 15))
            self.assertEqual(audit["unused_eligible_tail"], 1)
            self.assertEqual(index[0]["cycle"], 0)
            config["prompt_content_hashes"]["10"] = "validation-content"
            bind()
            with self.assertRaisesRegex(ValueError, "Cross-split"):
                load_training_pairs(root, split, evaluation)
            config["prompt_content_hashes"]["10"] = "training-content"
            config["prompt_groups"]["10"] = "assessment"
            bind()
            with self.assertRaisesRegex(ValueError, "Nontraining"):
                load_training_pairs(root, split, evaluation)
            config["prompt_groups"]["10"] = "train"
            bind()
            np.save(root/"prompt_10_fused.npy", np.zeros((2, 2560), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "shard hash"):
                load_training_pairs(root, split, evaluation)


if __name__ == "__main__":
    unittest.main()
