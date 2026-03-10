#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _make_model(*, vocab_size: int, dtype: torch.dtype, device: torch.device) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_bias=False,
    )
    model = LlamaForCausalLM(config)
    model.config._attn_implementation = "sdpa"
    model.eval().to(device=device, dtype=dtype)
    return model


@torch.inference_mode()
def _run_parity_case(
    *,
    model: LlamaForCausalLM,
    prefix_len: int,
    num_candidates: int,
    block_len: int,
    vocab_size: int,
    device: torch.device,
    atol: float,
    rtol: float,
) -> None:
    prefix_ids = torch.randint(3, vocab_size, (1, prefix_len), device=device)
    candidate_blocks = torch.randint(
        3, vocab_size, (num_candidates, block_len), device=device
    )
    position_ids = torch.arange(
        prefix_len, prefix_len + block_len, device=device
    ).unsqueeze(0)

    prefix_out = model(
        prefix_ids,
        use_cache=True,
        output_hidden_states=False,
    )
    prefix_cache = prefix_out.past_key_values

    packed_ids = candidate_blocks.reshape(1, num_candidates * block_len)
    packed_position_ids = position_ids.repeat(1, num_candidates)
    tree_mask = build_block_diagonal_tree_attention_mask(
        num_candidates=num_candidates,
        block_len=block_len,
        past_len=prefix_len,
        device=device,
        dtype=model.dtype,
    )
    packed_out = model(
        packed_ids,
        position_ids=packed_position_ids,
        attention_mask=tree_mask,
        past_key_values=prefix_cache,
        use_cache=True,
        output_hidden_states=False,
    )
    packed_logits = packed_out.logits.view(1, num_candidates, block_len, -1)[0]

    branch_logits = []
    for cand_idx in range(num_candidates):
        branch_prefix_out = model(
            prefix_ids,
            use_cache=True,
            output_hidden_states=False,
        )
        branch_out = model(
            candidate_blocks[cand_idx : cand_idx + 1],
            position_ids=position_ids,
            past_key_values=branch_prefix_out.past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )
        branch_logits.append(branch_out.logits[0])
    branch_logits = torch.stack(branch_logits, dim=0)

    if not torch.allclose(packed_logits, branch_logits, atol=atol, rtol=rtol):
        max_diff = float((packed_logits - branch_logits).abs().max().item())
        raise AssertionError(
            f"Packed-tree logits mismatch for prefix_len={prefix_len}, "
            f"num_candidates={num_candidates}, block_len={block_len}, max_diff={max_diff:.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny end-to-end logits parity check for packed mc4 SDPA tree verify."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--block-len", type=int, default=4)
    parser.add_argument("--prefix-lens", default="1,5,9")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    torch.manual_seed(args.seed)
    model = _make_model(vocab_size=args.vocab_size, dtype=dtype, device=device)
    prefix_lens = [int(x) for x in args.prefix_lens.split(",") if x.strip()]
    atol = 1e-4 if dtype != torch.float32 else 1e-5
    rtol = 1e-4 if dtype != torch.float32 else 1e-5
    for prefix_len in prefix_lens:
        _run_parity_case(
            model=model,
            prefix_len=prefix_len,
            num_candidates=args.num_candidates,
            block_len=args.block_len,
            vocab_size=args.vocab_size,
            device=device,
            atol=atol,
            rtol=rtol,
        )

    print(
        "PASS: packed-tree logits match per-branch SDPA logits "
        f"for prefix_lens={prefix_lens}, num_candidates={args.num_candidates}, block_len={args.block_len}."
    )


if __name__ == "__main__":
    main()
