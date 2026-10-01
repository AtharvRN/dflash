"""CPU execution checks for causal capture, index alignment and replay.

These fake deterministic models test collector plumbing, not GPU numerical
equivalence or the real Qwen/DFlash architecture.
"""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.collect_rejected_trace import (
    MODEL_REVISIONS, WIDTH, capture_memory, collect_prompt, prefix_hash,
    rejected_indices, validate_models,
)
from scripts.audit_rejected_trace_cache import validate_row


class Cache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = min(self.length, length)


class Tokens:
    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]])


def embed(ids):
    return ids.float()[..., None].expand(*ids.shape, WIDTH).clone()


def logits(ids):
    return torch.nn.functional.one_hot(ids.long(), num_classes=256).float()*10


class Target:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=255)
    model = SimpleNamespace(embed_tokens=embed)

    @staticmethod
    def lm_head(hidden):
        return logits(hidden[..., 0])

    def __call__(self, ids, *, past_key_values, position_ids, **kwargs):
        torch.testing.assert_close(position_ids, torch.arange(past_key_values.length,
                                   past_key_values.length+ids.shape[1])[None])
        past_key_values.length += ids.shape[1]
        # Greedy true continuation is always token+1.
        h = embed(ids)+1000
        return SimpleNamespace(logits=logits(ids+1), hidden_states=(h,))


class Draft:
    mask_token_id = 0
    target_layer_ids = [0]
    fc = torch.nn.Identity()
    hidden_norm = torch.nn.Identity()

    def __init__(self, accepted=2):
        self.accepted = accepted

    def __call__(self, *, target_hidden, noise_embedding, past_key_values, position_ids, **kwargs):
        b = noise_embedding.shape[1]
        torch.testing.assert_close(position_ids, torch.arange(past_key_values.length,
            past_key_values.length+target_hidden.shape[1]+b)[None])
        anchor = noise_embedding[:, :1, 0].long()
        candidates = anchor+torch.arange(b)[None]
        # First two proposals agree. Later proposals follow a wrong branch.
        candidates[:, self.accepted+1:] += 1
        past_key_values.length += target_hidden.shape[1]+b
        return embed(candidates)


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(max_prompt_tokens=2048, max_new_tokens=32,
                                    states_per_prompt=3, seed=1001)
        self.row = {"manifest_index": 11, "group": "train", "source": "fake",
                    "messages": [], "audit_prompt": True}

    def collect(self, wanted=None, row=None, args=None, draft=None):
        return collect_prompt(args or self.args, row or self.row, Target(), draft or Draft(),
                              Tokens(), wanted, cache_factory=Cache,
                              extract=lambda hidden, layers: hidden[0])

    def test_all_acceptance_index_pairs_and_padding(self):
        block = torch.arange(16)[None]
        dh = embed(block)+100
        th = embed(block)+200
        for accepted in range(16):
            memory = capture_memory(Target(), block, dh, th, accepted)
            di, ti = rejected_indices(accepted)
            n = max(14-accepted, 0)
            self.assertEqual(len(di), n)
            np.testing.assert_array_equal(ti, di-1)
            np.testing.assert_array_equal(memory["trace_draft"][:n, 0], di+100)
            np.testing.assert_array_equal(memory["trace_target"][:n, 0], ti+200)
            np.testing.assert_array_equal(memory["trace_token_ids"][:n], di)
            np.testing.assert_array_equal(memory["trace_offsets"][:n], np.arange(1, n+1))
            self.assertEqual(int(memory["trace_mask"].sum()), n)
            self.assertTrue((memory["trace_token_ids"][n:] == -1).all())
            self.assertTrue((memory["trace_draft"][n:] == 0).all())
            self.assertTrue((memory["trace_target"][n:] == 0).all())
            expected = accepted+1 if accepted < 15 else 0
            self.assertTrue((memory["rejected_anchor_embedding"] == expected).all())

    def test_previous_cycle_only_and_fresh_actual_labels(self):
        rows, arrays, summary = self.collect()
        self.assertEqual([row["cycle"] for row in rows], [0, 1, 2])
        self.assertEqual(summary["missing_source_cycles"], [])
        self.assertTrue(all(row["canonical_disagreements"] == 0 for row in rows))
        self.assertTrue(rows[0]["reverse_order_checked"])
        self.assertTrue(all(row["eligible"] for row in rows))
        for i, row in enumerate(rows):
            validate_row(row, arrays, i)
        self.assertEqual(rows[0]["previous"], None)
        self.assertTrue((arrays["previous_meta"][0] == 0).all())
        self.assertTrue((arrays["trace_mask"][0] == 0).all())
        self.assertTrue((arrays["rejected_anchor_embedding"][0] == 0).all())
        for i, row in enumerate(rows[1:], start=1):
            prev = row["previous"]
            self.assertEqual(prev["cycle"], row["cycle"]-1)
            self.assertEqual(prev["accepted"], 2)
            self.assertEqual(row["prefix_token_ids"], prev["prefix_token_ids"]+
                             prev["draft_ids"][:2]+[prev["correction_token_id"]])
            self.assertEqual(arrays["anchor_embedding"][i, 0], prev["correction_token_id"])
            self.assertEqual(arrays["rejected_anchor_embedding"][i, 0], prev["draft_ids"][2])
            # Candidate at old block index4 follows the rejected index3.
            self.assertEqual(arrays["trace_draft"][i, 0, 0], prev["draft_ids"][3])
            self.assertEqual(arrays["trace_target"][i, 0, 0], prev["draft_ids"][2]+1000)
            # Current fused feature is the most recent committed target state,
            # not the current (as yet unprocessed) anchor's target hidden state.
            self.assertEqual(arrays["features"][i, 0], row["prefix_token_ids"][-2]+1000)
        np.testing.assert_array_equal(arrays["actual"], np.tile([1]+[2]*14, (3, 1)))

    def test_unsampled_parent_is_not_previous_saved_row(self):
        source, _, _ = self.collect()
        rows, arrays, _ = self.collect({2: source[2]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cycle"], 2)
        self.assertEqual(rows[0]["previous_cycle"], 1)
        self.assertEqual(rows[0]["previous"]["prefix_sha256"], source[1]["prefix_sha256"])
        self.assertEqual(int(arrays["trace_mask"].sum()), 12)

    def test_source_labels_are_never_used(self):
        source, _, _ = self.collect()
        source[2]["outcomes"] = {str(b): {"accepted": 0} for b in range(2, 17)}
        rows, arrays, _ = self.collect({2: source[2]})
        self.assertTrue(rows[0]["eligible"])
        self.assertEqual(arrays["actual"][0, -1], 2)

    def test_reference_drift_is_recorded_and_excluded(self):
        source, _, _ = self.collect()
        altered = copy.deepcopy(source[2])
        altered["prefix_token_ids"][-1] += 1
        altered["prefix_sha256"] = prefix_hash(altered["prefix_token_ids"])
        rows, _, _ = self.collect({2: altered})
        self.assertFalse(rows[0]["eligible"])
        self.assertIn("reference_prefix_drift", rows[0]["exclusion"])

    def test_source_exclusion_is_preserved(self):
        source, _, _ = self.collect()
        source[2]["eligible"] = False
        rows, _, _ = self.collect({2: source[2]})
        self.assertFalse(rows[0]["eligible"])
        self.assertIn("source_excluded", rows[0]["exclusion"])

    def test_missing_source_and_long_prompt_are_explicit(self):
        _, _, summary = self.collect({100: {}})
        self.assertEqual(summary["missing_source_cycles"], [100])
        options = copy.copy(self.args)
        options.max_prompt_tokens = 2
        rows, arrays, summary = self.collect({100: {}}, args=options)
        self.assertFalse(rows)
        self.assertEqual(arrays["features"].shape, (0, WIDTH))
        self.assertEqual(summary["missing_source_cycles"], [100])
        self.assertEqual(summary["skipped"], "prompt_length")

    def test_unpinned_model_registry_rejected(self):
        models = {name: {"revision": revision, "path": "/models/"+revision}
                  for name, revision in MODEL_REVISIONS.items()}
        validate_models(models)
        models["target"]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_models(models)

    def test_full_accept_previous_has_bonus_but_no_wrong_anchor_or_trace(self):
        args = copy.copy(self.args)
        args.max_new_tokens = 64
        rows, arrays, _ = self.collect(args=args, draft=Draft(accepted=15))
        self.assertEqual([row["cycle"] for row in rows], [0, 1, 2])
        for index, row in enumerate(rows):
            validate_row(row, arrays, index)
        self.assertTrue((arrays["actual"] == np.arange(1, 16)[None]).all())
        self.assertTrue((arrays["trace_mask"] == 0).all())
        self.assertTrue((arrays["rejected_anchor_embedding"] == 0).all())
        self.assertEqual(rows[1]["previous"]["accepted"], 15)
        self.assertEqual(rows[1]["prefix_token_ids"][-1], rows[1]["previous"]["posterior_ids"][-1])


if __name__ == "__main__":
    unittest.main()
