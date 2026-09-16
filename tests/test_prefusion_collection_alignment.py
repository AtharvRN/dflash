"""Exercise collection control flow on CPU without importing Transformers."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
REJECTED = -10000.0


def _load_functions(path, names, namespace):
    # Execute the real function bodies, excluding heavyweight module imports/main.
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing test entry points in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class _CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **{**kwargs, "device": "cpu"})

    def full(self, *args, **kwargs):
        return torch.full(*args, **{**kwargs, "device": "cpu"})


class _InputIds(torch.Tensor):
    def cuda(self, *args, **kwargs):
        return self.as_subclass(torch.Tensor)


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([[1, 4, 5]]).as_subclass(_InputIds)


class _Cache:
    def __init__(self):
        self.entries = []
        self.crops = []

    def get_seq_length(self):
        return len(self.entries)

    def crop(self, length):
        self.entries = self.entries[:length]
        self.crops.append((length, tuple(self.entries)))


@dataclass(frozen=True)
class _Step:
    accepted: int
    eos_positions: tuple[int, ...] = ()
    correction: int = 3
    eos_token: int = 9


class _Target:
    def __init__(self, steps, *, first_anchor=2, eos=9, fail_verify=False):
        self.steps = steps
        self.first_anchor = first_anchor
        self.fail_verify = fail_verify
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.model = SimpleNamespace(embed_tokens=lambda ids: ids)
        self.lm_head = lambda hidden: hidden
        self.verifications = []
        self.cache = None

    def __call__(self, ids, position_ids, past_key_values, **kwargs):
        self.cache = past_key_values
        start = self.cache.get_seq_length()
        positions = position_ids[0].tolist()
        assert positions == list(range(start, start + ids.shape[1]))
        prefill = "logits_to_keep" in kwargs
        if not prefill and self.fail_verify:
            raise RuntimeError("injected verifier failure")
        p = position_ids.float().unsqueeze(-1)
        states = [torch.zeros((*ids.shape, 2)),
                  torch.cat([p, p + 100], dim=-1),
                  torch.cat([p + 200, p + 300], dim=-1)]
        posterior = torch.full_like(ids, self.first_anchor)
        if prefill:
            self.cache.entries.extend(("target", p) for p in positions)
        else:
            step = self.steps[len(self.verifications)]
            self.verifications.append({"block": ids.clone(), "cache_before": tuple(self.cache.entries)})
            posterior[:, :-1] = ids[:, 1:]
            posterior[:, step.accepted] = step.correction
            if step.accepted < 15:
                assert step.correction != int(ids[0, step.accepted + 1])
            for state in states[1:]:
                state[:, step.accepted + 1:] = REJECTED
            self.cache.entries.extend(
                ("target" if i <= step.accepted else "rejected", p)
                for i, p in enumerate(positions))
        logits = torch.zeros((*ids.shape, 16))
        logits.scatter_(-1, posterior.unsqueeze(-1), 1)
        return SimpleNamespace(hidden_states=tuple(states), logits=logits)


class _Draft(nn.Module):
    def __init__(self, steps, *, fusion_calls=1, fail_forward=False):
        super().__init__()
        self.steps = steps
        self.fusion_calls = fusion_calls
        self.fail_forward = fail_forward
        self.fc = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1., 0., .5, 0.], [0., .25, 0., 1.]]))
        self.hidden_norm = nn.Identity()
        self.config = SimpleNamespace(hidden_size=2)
        self.target_layer_ids = [0, 1]
        self.mask_token_id = 0
        self.calls = []
        self.cache = None

    def forward(self, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
        self.cache = past_key_values
        cache_length = self.cache.get_seq_length()
        context_length = target_hidden.shape[1]
        start = cache_length + context_length
        assert position_ids[0].tolist() == list(range(cache_length, start + 16))
        assert kwargs["use_cache"] and not kwargs["is_causal"]
        step = self.steps[len(self.calls)]
        record = {"cache_before": tuple(self.cache.entries), "pending": target_hidden.clone(),
                  "block_before": noise_embedding.clone()}
        self.calls.append(record)
        for _ in range(self.fusion_calls):
            record["fused_used"] = self.hidden_norm(self.fc(target_hidden)).clone()
        if self.fail_forward:
            raise RuntimeError("injected draft failure")
        self.cache.entries.extend(("context", p) for p in range(cache_length, start))
        self.cache.entries.extend(("noise", p) for p in range(start, start + 16))
        candidates = torch.full((1, 16), 2, dtype=torch.long)
        for position in step.eos_positions:
            candidates[0, position] = step.eos_token
        logits = torch.zeros((1, 16, 16))
        logits.scatter_(-1, candidates.unsqueeze(-1), 1)
        return logits


class PrefusionCollectionAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = _load_functions(
            ROOT / "scripts/collect_prefusion_acceptance.py",
            {"observable_label", "collect_prompt"},
            {"torch": _CpuTorch(), "np": np, "hashlib": hashlib})
        cls.model_functions = _load_functions(
            ROOT / "dflash/model.py", {"extract_context_feature"}, {"torch": torch})

    def collect(self, steps, *, target=None, draft=None, **limits):
        target = target if target is not None else _Target(steps)
        draft = draft if draft is not None else _Draft(steps)
        args = SimpleNamespace(**{"max_prompt_tokens": 32, "max_new_tokens": 100,
                                  "max_cycles": len(steps), **limits})
        transformers = ModuleType("transformers")
        transformers.DynamicCache = _Cache
        model = ModuleType("dflash.model")
        model.extract_context_feature = self.model_functions["extract_context_feature"]
        with patch.dict(sys.modules, {"transformers": transformers, "dflash.model": model}):
            data, info = self.collector["collect_prompt"](
                {"messages": []}, target, draft, _Tokenizer(), args)
        self.assertFalse(draft.hidden_norm._forward_hooks)
        return data, info, target, draft

    def test_multicycle_alignment_and_paired_fusion(self):
        data, _, target, draft = self.collect([_Step(0), _Step(15), _Step(7)])
        np.testing.assert_array_equal(data["accepted_len"], [0, 15, 7])
        np.testing.assert_array_equal(data["cycle_id"], [0, 1, 2])
        np.testing.assert_array_equal(data["prefix_length"], [3, 4, 20])
        np.testing.assert_array_equal(data["anchor_id"], [2, 3, 3])
        positions = np.array([2, 3, 19])[:, None]
        expected_raw = np.stack([positions + offset for offset in (0, 100, 200, 300)], axis=-1)
        np.testing.assert_array_equal(data["raw_features"], expected_raw)
        np.testing.assert_array_equal(data["features"],
                                     np.stack([c["fused_used"][0, -1:].numpy() for c in draft.calls]))
        self.assertEqual([(len(c["cache_before"]), c["pending"].shape[1]) for c in draft.calls],
                         [(0, 3), (3, 1), (4, 16)])
        self.assertEqual([length for length, _ in draft.cache.crops], [3, 4, 20])
        self.assertEqual([length for length, _ in target.cache.crops], [4, 20, 28])
        for start, anchor, digest in zip(data["prefix_length"], data["anchor_id"], data["prefix_sha256"]):
            prefix = data["trajectory_token_ids"][:start + 1]
            self.assertEqual(len(prefix), start + 1)
            self.assertEqual(prefix[-1], anchor)
            self.assertEqual(hashlib.sha256(prefix.astype(np.int64).tobytes()).hexdigest(), digest)

    def test_rejected_states_and_speculative_cache_entries_are_isolated(self):
        data, _, target, draft = self.collect([_Step(0), _Step(2), _Step(7)])
        np.testing.assert_array_equal(data["raw_features"][:, 0, 0], [2, 3, 6])
        for call in draft.calls:
            self.assertFalse((call["pending"] == REJECTED).any())
            self.assertTrue(all(kind == "context" for kind, _ in call["cache_before"]))
            self.assertTrue((call["block_before"][:, 1:] == draft.mask_token_id).all())
        for call in target.verifications:
            self.assertTrue(all(kind == "target" for kind, _ in call["cache_before"]))
        self.assertTrue(all(kind == "target" for kind, _ in target.cache.entries))
        self.assertTrue(all(kind == "context" for kind, _ in draft.cache.entries))

    def test_hook_cleanup_on_forward_exceptions(self):
        steps = [_Step(0)]
        for failure in ("draft", "verifier"):
            with self.subTest(failure=failure):
                draft = _Draft(steps, fail_forward=failure == "draft")
                target = _Target(steps, fail_verify=failure == "verifier")
                with self.assertRaisesRegex(RuntimeError, f"injected {failure} failure"):
                    self.collect(steps, target=target, draft=draft)
                self.assertFalse(draft.hidden_norm._forward_hooks)

    def test_hook_requires_exactly_one_fusion_call(self):
        for count in (0, 2):
            with self.subTest(count=count):
                steps = [_Step(0)]
                draft = _Draft(steps, fusion_calls=count)
                with self.assertRaisesRegex(RuntimeError, "Fusion hook"):
                    self.collect(steps, draft=draft)
                self.assertFalse(draft.hidden_norm._forward_hooks)

    def test_known_eos_anchor_stops_before_draft(self):
        for token in (9, 10):
            with self.subTest(token=token):
                steps = [_Step(0)]
                data, info, target, draft = self.collect(
                    steps, target=_Target(steps, first_anchor=token, eos=[9, 10]))
                self.assertIsNone(data)
                self.assertEqual(info["skipped"], "no_observable_cycles")
                self.assertFalse(draft.calls)
                self.assertFalse(target.verifications)

    def test_accepted_eos_excludes_row_and_stops(self):
        for position in (1, 15):
            with self.subTest(position=position):
                steps = [_Step(position, eos_positions=(position,)), _Step(0)]
                data, _, target, draft = self.collect(steps)
                self.assertIsNone(data)
                self.assertEqual(len(draft.calls), 1)
                self.assertEqual(len(target.verifications), 1)

    def test_rejected_eos_does_not_stop_next_cycle(self):
        for position in (3, 8):
            with self.subTest(position=position):
                data, _, _, draft = self.collect([_Step(2, eos_positions=(position,)), _Step(0)])
                np.testing.assert_array_equal(data["accepted_len"], [2, 0])
                np.testing.assert_array_equal(data["prefix_length"], [3, 6])
                self.assertEqual(len(draft.calls), 2)
                self.assertNotIn(9, data["trajectory_token_ids"])

    def test_correction_eos_keeps_label_but_stops_next_cycle(self):
        for accepted in (0, 7, 15):
            with self.subTest(accepted=accepted):
                steps = [_Step(accepted, correction=10), _Step(0)]
                data, _, _, draft = self.collect(steps, target=_Target(steps, eos=[9, 10]))
                np.testing.assert_array_equal(data["accepted_len"], [accepted])
                self.assertEqual(len(draft.calls), 1)
                self.assertEqual(data["trajectory_token_ids"][-1], 10)

    def test_full_block_budget_boundary(self):
        for remaining in (15, 16):
            for accepted in (0, 15):
                with self.subTest(remaining=remaining, accepted=accepted):
                    data, _, target, draft = self.collect(
                        [_Step(accepted), _Step(0)], max_new_tokens=remaining)
                    if remaining == 15:
                        self.assertIsNone(data)
                        self.assertFalse(draft.calls)
                        self.assertFalse(target.verifications)
                    else:
                        np.testing.assert_array_equal(data["accepted_len"], [accepted])
                        self.assertEqual(len(draft.calls), 1)


if __name__ == "__main__":
    unittest.main()
