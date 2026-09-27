"""Paired post-hoc draft-mask intervention on saved training-development states."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def draft_mask(context, block, device, dtype):
    import torch
    allowed = torch.cat([torch.ones(block, context, dtype=torch.bool, device=device),
                         torch.ones(block, block, dtype=torch.bool, device=device).tril()], dim=1)
    return torch.zeros((1, 1, block, context+block), device=device, dtype=dtype).masked_fill(
        ~allowed[None, None], float('-inf'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompts-per-source', type=int, default=4)
    p.add_argument('--states-per-prompt', type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing an existing output directory')
    used = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError('GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache
    from dflash.model import DFlashDraftModel, extract_context_feature
    from scripts.diagnose_dflash_paired_lengths import atomic_json, prefix_matches

    torch.set_num_threads(4)
    torch.manual_seed(927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Explicitly test rectangular-mask alignment: every query sees all context.
    m = draft_mask(7, 4, 'cpu', torch.float32)[0, 0]
    assert (m[:, :7] == 0).all()
    assert (m[:, 7:] == 0).equal(torch.ones(4, 4, dtype=torch.bool).tril())
    pools = {}
    for path in sorted(args.input.glob('prompt_*.json'), key=lambda x: int(x.stem.split('_')[1])):
        states = [r for r in json.loads(path.read_text())['states'] if r['eligible']]
        if states:
            pools.setdefault(states[0]['source'], []).append(states)
    selected = []
    for source, prompts in sorted(pools.items()):
        random.Random(927).shuffle(prompts)
        for states in prompts[:args.prompts_per_source]:
            idx = np.linspace(0, len(states)-1, min(len(states), args.states_per_prompt), dtype=int)
            selected.extend(states[i] for i in idx)
    args.output.mkdir(parents=True)
    models = json.loads((Path(os.environ['DFLASH_ROOT'])/'models.json').read_text())
    blocks = [4, 8, 12, 16, 20]
    atomic_json(args.output/'config.json', {'models': models, 'blocks': blocks,
        'selection': [[s['prompt_id'], s['cycle']] for s in selected],
        'input': str(args.input), 'seed': 927, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'protocol': 'Full-prefix replay, same states and target features for both masks; greedy BF16 SDPA; no training or throughput claim.'})
    target = AutoModelForCausalLM.from_pretrained(models['target']['path'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).cuda().eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models['draft']['path'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).cuda().eval().requires_grad_(False)
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) - {None}
    results = []
    with torch.inference_mode():
        for index, state in enumerate(selected):
            prefix = state['prefix_token_ids']
            assert hashlib.sha256(np.asarray(prefix, dtype=np.int64).tobytes()).hexdigest() == state['prefix_sha256']
            ids = torch.tensor([prefix], device='cuda')
            n = len(prefix)-1
            assert n == state['prefix_length']
            tc = DynamicCache()
            prefill = target(ids[:, :-1], past_key_values=tc, use_cache=True,
                             output_hidden_states=True, logits_to_keep=1)
            hidden = extract_context_feature(prefill.hidden_states, draft.target_layer_ids)
            del prefill
            # Independent greedy continuation is the common acceptance reference.
            canonical, cc, token = [], copy.deepcopy(tc), ids[:, -1:]
            for _ in range(19):
                out = target(token, past_key_values=cc, use_cache=True, logits_to_keep=1)
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                canonical.append(int(token))
                if int(token) in eos:
                    break
            del out, cc
            row = {k: state[k] for k in ('prompt_id', 'cycle', 'source', 'prefix_sha256')}
            row.update({'outcomes': {}, 'canonical': canonical, 'verification_disagreements': 0})
            jobs = [(mode, b) for mode in ('bidirectional', 'causal') for b in blocks]
            random.Random(927+index).shuffle(jobs)
            for mode, b in jobs:
                tokens = torch.full((1, b), draft.mask_token_id, device='cuda', dtype=torch.long)
                tokens[:, 0] = ids[:, -1]
                mask = draft_mask(n, b, 'cuda', hidden.dtype) if mode == 'causal' else None
                output = draft(target_hidden=hidden, noise_embedding=target.model.embed_tokens(tokens),
                    position_ids=torch.arange(n+b, device='cuda')[None], attention_mask=mask,
                    past_key_values=DynamicCache(), use_cache=True, is_causal=False)
                tokens[:, 1:] = target.lm_head(output[:, 1:]).argmax(-1)
                candidate = tokens[0, 1:].tolist()
                verified = target(tokens, past_key_values=copy.deepcopy(tc), use_cache=True)
                a = int((tokens[:, 1:] == verified.logits[:, :-1].argmax(-1)).cumprod(-1).sum())
                ca = prefix_matches(candidate, canonical)
                row['verification_disagreements'] += int(min(a, len(canonical)) != ca)
                row['outcomes'].setdefault(mode, {})[str(b)] = {'accepted': a, 'canonical_accepted': ca,
                    'draft_ids': candidate}
                del output, verified
            row['eligible'] = not any(t in eos for mode in row['outcomes'].values()
                for o in mode.values() for t in o['draft_ids'][:o['accepted']])
            row['saved_b16_replay_equal'] = row['outcomes']['bidirectional']['16'] == {
                **state['outcomes']['16'], 'canonical_accepted': state['outcomes']['16']['accepted']}
            atomic_json(args.output/f'state_{index:03d}.json', row)
            results.append(row)
            del hidden, tc
            print(f'Completed {index+1}/{len(selected)} states', flush=True)
    eligible = [r for r in results if r['eligible']]
    summary = {'states': len(results), 'eligible': len(eligible),
        'prompts': len({r['prompt_id'] for r in results}),
        'verification_disagreements': sum(r['verification_disagreements'] for r in results),
        'saved_b16_replay_matches': sum(r['saved_b16_replay_equal'] for r in results), 'modes': {}}
    for mode in ('bidirectional', 'causal'):
        summary['modes'][mode] = {'mean_accepted': {str(b): float(np.mean([
            r['outcomes'][mode][str(b)]['accepted'] for r in eligible])) for b in blocks}, 'adjacent_blocks': {}}
        for small, large in zip(blocks, blocks[1:]):
            pairs = [(r['outcomes'][mode][str(small)], r['outcomes'][mode][str(large)]) for r in eligible]
            full = [(s,l) for s,l in pairs if s['accepted'] == small-1]
            summary['modes'][mode]['adjacent_blocks'][f'{small}->{large}'] = {
                'pairs': len(pairs), 'prefix_changes': sum(s['draft_ids'] != l['draft_ids'][:small-1] for s,l in pairs),
                'acceptance_regressions': sum(l['accepted'] < s['accepted'] for s,l in pairs),
                'fully_accepted_small': len(full),
                'fully_accepted_small_regressions': sum(l['accepted'] < small-1 for s,l in full)}
    atomic_json(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
