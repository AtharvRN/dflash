#!/usr/bin/env python3
"""Locked benchmark entrypoint for sample_multi + tree verification.

This wrapper forwards a minimal, stable flag surface to
benchmark_candidate_solutions.py while hard-locking:
  --candidate-mode sample_multi
  --candidate-verify-mode tree
"""

from __future__ import annotations

import argparse
import sys

import benchmark_candidate_solutions as base


def _build_forward_argv(args: argparse.Namespace) -> list[str]:
    out = [
        "--model-name-or-path",
        str(args.model_name_or_path),
        "--draft-name-or-path",
        str(args.draft_name_or_path),
        "--dataset",
        str(args.dataset),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--max-candidates",
        str(int(args.max_candidates)),
        "--candidate-sample-temperature",
        str(float(args.candidate_sample_temperature)),
        "--candidate-mode",
        "sample_multi",
        "--candidate-verify-mode",
        "tree",
        "--verify-cache-clone-mode",
        str(args.verify_cache_clone_mode),
    ]

    if args.block_size is not None:
        out.extend(["--block-size", str(int(args.block_size))])
    if args.max_samples is not None:
        out.extend(["--max-samples", str(int(args.max_samples))])
    if args.skip_baseline:
        out.append("--skip-baseline")
    if args.local_files_only:
        out.append("--local-files-only")
    if args.log_all_ranks:
        out.append("--log-all-ranks")
    if args.collect_profile:
        out.append("--collect-profile")
    if args.candidate_verify_static_shape:
        out.append("--candidate-verify-static-shape")
    if args.detailed_cycle_metadata:
        out.append("--detailed-cycle-metadata")
    if args.save_outputs_path:
        out.extend(["--save-outputs-path", str(args.save_outputs_path)])
    if args.save_cycle_trace_path:
        out.extend(["--save-cycle-trace-path", str(args.save_cycle_trace_path)])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark_candidate_solutions.py with a locked path: "
            "candidate_mode=sample_multi and candidate_verify_mode=tree."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--candidate-sample-temperature", type=float, default=1.0)
    parser.add_argument(
        "--verify-cache-clone-mode",
        type=str,
        choices=["inplace", "shallow", "deep"],
        default="inplace",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--collect-profile", action="store_true")
    parser.add_argument("--candidate-verify-static-shape", action="store_true")
    parser.add_argument("--detailed-cycle-metadata", action="store_true")
    parser.add_argument("--save-outputs-path", type=str, default=None)
    parser.add_argument("--save-cycle-trace-path", type=str, default=None)
    args = parser.parse_args()

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be >= 1")
    if args.candidate_sample_temperature <= 0.0:
        raise ValueError("--candidate-sample-temperature must be > 0")
    if args.temperature >= 1e-5:
        raise ValueError("This benchmark path currently supports only --temperature 0.0")

    forward_argv = _build_forward_argv(args)
    old_argv = sys.argv
    try:
        sys.argv = ["benchmark_candidate_solutions.py", *forward_argv]
        base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
