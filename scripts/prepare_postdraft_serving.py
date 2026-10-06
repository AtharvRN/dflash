"""Verified historical restore + narrowly scoped, auditable serving extension.

Historical archive/overlay are immutable. Every replacement must match exactly
once; the result records before/after hashes and the extension's own source hash.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from recover_legacy_ragged import restore, digest

ROOT = Path(__file__).resolve().parents[1]


def extend(destination):
    changes = {}
    def edit(rel, old, new):
        path = destination / "python/sglang/srt" / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f"Extension anchor not unique: {rel}: {old[:90]!r}")
        entry = changes.setdefault(rel, {"before_sha256": digest(path)})
        path.write_text(text.replace(old, new))
        entry["after_sha256"] = digest(path)

    edit("server_args.py",
         '    speculative_dflash_postdraft_alpha: float = 0.84',
         '    speculative_dflash_postdraft_logprob_threshold: Optional[float] = None\n'
         '    speculative_dflash_postdraft_alpha: float = 0.84')
    edit("server_args.py", '            "--speculative-dflash-postdraft-alpha",',
         '            "--speculative-dflash-postdraft-logprob-threshold",\n'
         '            type=float, default=None,\n'
         '            help="Greedy post-draft consecutive logprob threshold; anchor always retained.",\n'
         '        )\n        parser.add_argument(\n'
         '            "--speculative-dflash-postdraft-alpha",')
    edit("speculative/dflash_worker_v2.py", '        self.postdraft_policy: Optional[DFlashPostdraftPolicy] = None',
         '        self.postdraft_policy: Optional[DFlashPostdraftPolicy] = None\n'
         '        raw_threshold = server_args.speculative_dflash_postdraft_logprob_threshold\n'
         '        if raw_threshold is not None:\n'
         '            from sglang.srt.speculative.dflash_raw_confidence import DFlashRawConfidencePolicy\n'
         '            if (server_args.speculative_dflash_postdraft_policy_path\n'
         '                or server_args.speculative_dflash_predraft_entropy_policy_path\n'
         '                or server_args.speculative_dflash_dynamic_block_size):\n'
         '                raise ValueError("Raw post-draft policy cannot be combined with another policy")\n'
         '            if server_args.tp_size != 1 or server_args.attention_backend != "flashinfer":\n'
         '                raise ValueError("Raw post-draft serving requires TP1 / FlashInfer")\n'
         '            self.postdraft_policy = DFlashRawConfidencePolicy(\n'
         '                raw_threshold, int(server_args.speculative_num_draft_tokens))\n')
    edit("speculative/dflash_worker_v2.py", '            return_confidence=need_postdraft_scalar,',
         '            return_confidence=(getattr(self.postdraft_policy, "confidence_mode", True)\n'
         '                               if need_postdraft_scalar else False),')
    edit("speculative/dflash_worker.py", '            probs = log_probs.exp()\n            entropy = -(probs * log_probs).sum(dim=-1)',
         '            if return_confidence == "logprob_only":\n'
         '                token_logprob = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)\n'
         '                stats = torch.zeros((logits.shape[0], 4), device=logits.device, dtype=torch.float32)\n'
         '                stats[:, 3] = token_logprob\n'
         '                return stats\n'
         '            probs = log_probs.exp()\n            entropy = -(probs * log_probs).sum(dim=-1)')
    edit("model_executor/runner/decode_cuda_graph_runner.py",
         '                    or self.model_runner.server_args.speculative_dflash_postdraft_policy_path',
         '                    or self.model_runner.server_args.speculative_dflash_postdraft_policy_path\n'
         '                    or self.model_runner.server_args.speculative_dflash_postdraft_logprob_threshold is not None')
    edit("model_executor/runner/decode_cuda_graph_runner.py",
         '                if max_tokens_per_bs not in dynamic_buckets:',
         '                if self.model_runner.server_args.speculative_dflash_postdraft_logprob_threshold is not None:\n'
         '                    dynamic_buckets.extend(range(1, max_tokens_per_bs + 1))\n\n'
         '                if max_tokens_per_bs not in dynamic_buckets:')
    target = destination / "python/sglang/srt/speculative/dflash_raw_confidence.py"
    shutil.copy2(ROOT / "dflash/serving_raw_confidence.py", target)
    changes["speculative/dflash_raw_confidence.py"] = {"after_sha256": digest(target)}
    manifest = {"format": "postdraft_serving_extension_v1", "changes": changes,
                "extension_sha256": digest(Path(__file__)), "policy_sha256": digest(target)}
    (destination / "POSTDRAFT_EXTENSION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(destination):
    restore(ROOT / "vendor/sglang_ragged_20260723", destination)
    return extend(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
