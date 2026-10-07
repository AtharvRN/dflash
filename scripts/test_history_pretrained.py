"""Bounded pretrained greedy correctness check; writes evidence even on mismatch."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refusing existing evidence')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel
    from dflash.history_generate import generate_history
    torch.manual_seed(1007)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = json.loads(args.models.read_text())
    kw = dict(local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    target = AutoModelForCausalLM.from_pretrained(models['target']['path'], **kw).cuda().eval()
    draft = DFlashDraftModel.from_pretrained(models['draft']['path'], **kw).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(models['target']['path'], local_files_only=True)
    stops = target.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [stops] if stops is not None else []
    evidence = {'scope': 'BF16 SDPA pretrained greedy correctness, not throughput', 'tests': []}
    for prompt in ('What is 17 times 23? Explain briefly.',
                   'Write a Python function that reverses a list without modifying the input.'):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            enable_thinking=False, add_generation_prompt=True, return_tensors='pt').cuda()
        with torch.inference_mode():
            ar = target.generate(ids, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        blocks = tuple(range(2, 17))
        def choose(h, cycle):
            return blocks[cycle % len(blocks)]
        common = dict(blocks=blocks, choose=choose, max_new_tokens=96, stop_token_ids=stops)
        plain = generate_history(draft, target, ids, **common)
        paired = generate_history(draft, target, ids, collect=True, **common)
        def suffix(x):
            return x[0, ids.shape[1]:].tolist()
        evidence['tests'].append(dict(prompt=prompt, ar=suffix(ar),
            plain=suffix(plain['output_ids']), paired=suffix(paired['output_ids']),
            exact_ar=torch.equal(ar, plain['output_ids']),
            exact_probe_output=torch.equal(plain['output_ids'], paired['output_ids']),
            exact_probe_history=[r['history'] for r in plain['cycles']] ==
                                [r['history'] for r in paired['cycles']],
            blocks_observed=[r['selected_block'] for r in plain['cycles']]))
        print(evidence['tests'][-1]['prompt'], evidence['tests'][-1]['exact_ar'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    if not all(t[k] for t in evidence['tests'] for k in
               ('exact_ar', 'exact_probe_output', 'exact_probe_history')):
        raise SystemExit('Pretrained mismatch: inspect saved evidence before proceeding')


if __name__ == '__main__':
    main()
