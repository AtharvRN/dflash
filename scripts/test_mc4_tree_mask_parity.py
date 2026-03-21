#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_function(*, source_path: Path, function_name: str):
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    fn_node = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            fn_node = node
            break
    if fn_node is None:
        raise RuntimeError(f"Could not locate {function_name} in {source_path}")

    isolated = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"torch": torch}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace[function_name]


build_block_diagonal_tree_attention_mask = _load_function(
    source_path=REPO_ROOT / "benchmark_candidate_solutions.py",
    function_name="build_block_diagonal_tree_attention_mask",
)
build_dflash_verify_allow_mask = _load_function(
    source_path=REPO_ROOT / "third_party" / "sglang" / "python" / "sglang" / "srt" / "speculative" / "dflash_info.py",
    function_name="build_dflash_verify_allow_mask",
)


def _allow_to_additive_mask(allow: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    neg_inf = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, allow.shape[0], allow.shape[1]),
        neg_inf,
        dtype=dtype,
        device=allow.device,
    )
    mask.masked_fill_(allow.unsqueeze(0).unsqueeze(0), 0.0)
    return mask


def _run_single_case(
    *,
    prefix_len: int,
    num_candidates: int,
    block_len: int,
    device: torch.device,
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
) -> None:
    allow = build_dflash_verify_allow_mask(
        prefix_len=prefix_len,
        draft_token_num=block_len,
        num_candidates=num_candidates,
        candidate_block_size=block_len,
        device=device,
    )
    ref_mask = build_block_diagonal_tree_attention_mask(
        num_candidates=num_candidates,
        block_len=block_len,
        past_len=prefix_len,
        device=device,
        dtype=dtype,
    )
    test_mask = _allow_to_additive_mask(allow, dtype=dtype)

    if not torch.equal(ref_mask, test_mask):
        raise AssertionError(
            f"Mask mismatch for prefix_len={prefix_len}, "
            f"num_candidates={num_candidates}, block_len={block_len}"
        )

    q_len = num_candidates * block_len
    kv_len = prefix_len + q_len
    q = torch.randn((1, heads, q_len, head_dim), device=device, dtype=dtype)
    k = torch.randn((1, heads, kv_len, head_dim), device=device, dtype=dtype)
    v = torch.randn((1, heads, kv_len, head_dim), device=device, dtype=dtype)

    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask)
    out_test = F.scaled_dot_product_attention(q, k, v, attn_mask=test_mask)
    if not torch.allclose(out_ref, out_test, atol=1e-5, rtol=1e-5):
        max_diff = float((out_ref - out_test).abs().max().item())
        raise AssertionError(
            f"SDPA output mismatch for prefix_len={prefix_len}, "
            f"num_candidates={num_candidates}, block_len={block_len}, "
            f"max_diff={max_diff:.3e}"
        )


def _run_batch_flatten_case(
    *,
    prefix_lens: list[int],
    num_candidates: int,
    block_len: int,
    device: torch.device,
) -> None:
    chunks = []
    for prefix_len in prefix_lens:
        allow = build_dflash_verify_allow_mask(
            prefix_len=prefix_len,
            draft_token_num=block_len,
            num_candidates=num_candidates,
            candidate_block_size=block_len,
            device=device,
        )
        chunks.append(allow.flatten())

    flat = torch.cat(chunks, dim=0)
    expected_numel = sum(
        (prefix_len + num_candidates * block_len) * (num_candidates * block_len)
        for prefix_len in prefix_lens
    )
    if flat.numel() != expected_numel:
        raise AssertionError(
            f"Flattened custom mask size mismatch: got {flat.numel()}, "
            f"expected {expected_numel}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity test for SGLang DFLASH mc4 packed-tree mask against the Transformers SDPA tree reference."
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--block-len", type=int, default=4)
    parser.add_argument(
        "--prefix-lens",
        default="0,1,3,7",
        help="Comma-separated prefix lengths for per-request parity checks.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    prefix_lens = [int(x) for x in args.prefix_lens.split(",") if x.strip()]
    if not prefix_lens:
        raise ValueError("Need at least one prefix length.")
    if args.num_candidates <= 1:
        raise ValueError("This parity test is for packed multi-candidate verify only.")
    if args.block_len <= 0:
        raise ValueError("block_len must be positive.")

    torch.manual_seed(0)

    for prefix_len in prefix_lens:
        _run_single_case(
            prefix_len=prefix_len,
            num_candidates=args.num_candidates,
            block_len=args.block_len,
            device=device,
            dtype=dtype,
            heads=args.heads,
            head_dim=args.head_dim,
        )

    _run_batch_flatten_case(
        prefix_lens=prefix_lens,
        num_candidates=args.num_candidates,
        block_len=args.block_len,
        device=device,
    )

    print(
        "PASS: DFLASH mc4 packed-tree mask matches the Transformers SDPA tree reference "
        f"for prefix_lens={prefix_lens}, num_candidates={args.num_candidates}, block_len={args.block_len}."
    )


if __name__ == "__main__":
    main()
