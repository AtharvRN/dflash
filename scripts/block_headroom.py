"""Exact, descriptive acceptance/budget frontiers on paired common states.

No trained policy, timing reward, or claim about closed-loop speedup is involved.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def select_training_prompts(manifest, split_dir, per_source, seed):
    train = set(map(int, json.loads((split_dir / 'train_prompt_ids.json').read_text())['train_prompt_ids']))
    val = set(map(int, json.loads((split_dir / 'val_prompt_ids.json').read_text())['val_prompt_ids']))
    if train & val:
        raise ValueError('Canonical train/validation overlap')
    rows, seen, content_groups = [], set(), defaultdict(set)
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = int(row['manifest_index'])
            if pid in seen:
                raise ValueError('Duplicate prompt ID')
            seen.add(pid)
            digest = hashlib.sha256(json.dumps(row['messages'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            group = 'train' if pid in train else 'val' if pid in val else 'other'
            content_groups[digest].add(group)
            if group == 'train':
                rows.append({**row, 'content_sha256': digest})
    pools = defaultdict(list)
    unique = set()
    for row in rows:
        digest = row['content_sha256']
        if content_groups[digest] == {'train'} and digest not in unique:
            pools[str(row['source'])].append(row)
            unique.add(digest)
    expected = {'nemotron', 'opencodeinstruct', 'openr1_math', 'evol_codealpaca'}
    if set(pools) != expected:
        raise ValueError(f'Unexpected source labels: {sorted(pools)}')
    rng = random.Random(seed)
    sources = sorted(pools)
    for source in sources:
        rng.shuffle(pools[source])
        if len(pools[source]) < per_source:
            raise ValueError(f'Insufficient prompts for {source}')
    # Interleave sources so partial runs do not disproportionately cover one source.
    return [pools[source][i] for i in range(per_source) for source in sources]


def exact_frontier(acceptance, costs, targets):
    """Multiple-choice integer DP; one action per group, maximum A at each budget.

    Prompt groups aggregate all their states. Cycle groups contain one state.
    Targets are absolute accepted-token counts, not a mean of per-cycle ratios.
    """
    acceptance = np.asarray(acceptance, dtype=np.int32)
    costs = np.asarray(costs, dtype=np.int32)
    if acceptance.shape != costs.shape or acceptance.ndim != 2 or not len(acceptance):
        raise ValueError('Expected nonempty, matching group/action matrices')
    if (acceptance < 0).any() or (costs < 1).any() or acceptance.shape[1] > 127:
        raise ValueError('Invalid acceptance/cost/action count')
    limit = int(costs.max(axis=1).sum())
    dp = np.full(limit + 1, -1, dtype=np.int32)
    dp[0] = 0
    pointers = []
    reachable = 0
    for accepted, budgets in zip(acceptance, costs):
        nxt = np.full(limit + 1, -1, dtype=np.int32)
        choice = np.full(limit + 1, -1, dtype=np.int8)
        prior = dp[:reachable + 1]
        for action, (a, c) in enumerate(zip(accepted, budgets)):
            candidate = np.where(prior >= 0, prior + a, -1)
            dest = nxt[c:c + reachable + 1]
            better = candidate > dest
            dest[better] = candidate[better]
            choice[c:c + reachable + 1][better] = action
        reachable += int(budgets.max())
        dp = nxt
        pointers.append(choice)
    result = []
    for target in targets:
        feasible = np.flatnonzero(dp >= target)
        if not len(feasible):
            result.append({'target_accepted': int(target), 'feasible': False})
            continue
        budget = int(feasible[0])
        cursor, selected = budget, []
        for group in range(len(acceptance)-1, -1, -1):
            action = int(pointers[group][cursor])
            if action < 0:
                raise AssertionError('Broken DP backpointer')
            selected.append(action)
            cursor -= int(costs[group, action])
        selected.reverse()
        idx = np.arange(len(selected))
        assert cursor == 0 and int(costs[idx, selected].sum()) == budget
        assert int(acceptance[idx, selected].sum()) == int(dp[budget])
        result.append({'target_accepted': int(target), 'feasible': True,
                       'total_budget': budget, 'total_accepted': int(dp[budget]),
                       'selected_actions': selected})
    best_budget = int(np.argmax(dp))
    return result, {'total_budget': best_budget, 'total_accepted': int(dp[best_budget])}


def summarize(rows, by_source=True):
    eligible = [r for r in rows if r['eligible']]
    result = {'states': len(rows), 'eligible_states': len(eligible),
              'scope': 'source-balanced TRAINING development diagnostic, common B16 states; hindsight oracles, not deployed policies or throughput',
              'exclusions': dict(Counter(r.get('exclusion') for r in rows if not r['eligible']))}
    if not eligible:
        return result
    blocks = sorted(map(int, eligible[0]['outcomes']))
    if any(sorted(map(int, r['outcomes'])) != blocks for r in eligible):
        raise ValueError('Incomplete paired outcome matrix')
    a = np.array([[r['outcomes'][str(b)]['accepted'] for b in blocks] for r in eligible], dtype=np.int32)
    budgets = np.array(blocks, dtype=np.int32) - 1
    if (a < 0).any() or (a > budgets).any():
        raise ValueError('Acceptance outside proposed budget')
    n = len(a)
    baseline = int(a[:, blocks.index(16)].sum())
    if baseline == 0:
        raise ValueError('Zero B16 acceptance; retention undefined')
    prompt_ids = sorted({r['prompt_id'] for r in eligible})
    prompt_rows = [[i for i, r in enumerate(eligible) if r['prompt_id'] == p] for p in prompt_ids]
    pa = np.array([a[idx].sum(axis=0) for idx in prompt_rows])
    pc = np.array([budgets * len(idx) for idx in prompt_rows])
    result.update({'blocks': blocks, 'prompts': len(prompt_ids), 'b16_total_accepted': baseline,
                   'mean_b16_accepted': baseline / n, 'source_states': dict(Counter(r['source'] for r in eligible)),
                   'fixed': {str(b): {'mean_accepted': float(a[:, j].mean()), 'mean_budget': int(b-1),
                                       'retention': float(a[:, j].sum()/baseline),
                                       'aggregate_ratio': float(a[:, j].sum()/(n*(b-1)))} for j, b in enumerate(blocks)}})
    retentions = [.90, .95, .96, .98, 1.0]
    targets = [int(np.ceil(r * baseline - 1e-9)) for r in retentions]
    frontiers = {}
    for name, ga, gc in [('global', a.sum(axis=0)[None], (budgets*n)[None]),
                          ('per_prompt', pa, pc), ('per_cycle', a, np.broadcast_to(budgets, a.shape))]:
        points, maximum = exact_frontier(ga, gc, targets)
        for target_retention, point in zip(retentions, points):
            point['target_retention'] = target_retention
            if point['feasible']:
                point['mean_budget'] = point['total_budget']/n
                point['mean_accepted'] = point['total_accepted']/n
                point['retention'] = point['total_accepted']/baseline
                point['aggregate_ratio'] = point['total_accepted']/point['total_budget']
                actions = point.pop('selected_actions')
                point['selected_blocks'] = [blocks[j] for j in actions]
        maximum.update({'mean_accepted': maximum['total_accepted']/n, 'mean_budget': maximum['total_budget']/n})
        frontiers[name] = {'points': points, 'maximum_acceptance': maximum}
    result['frontiers'] = frontiers
    result['prompt_order'] = prompt_ids
    result['cycle_order'] = [[r['prompt_id'], r['cycle']] for r in eligible]
    means = np.array([a[idx, blocks.index(16)].mean() for idx in prompt_rows])
    weights = np.array([len(idx) for idx in prompt_rows]) / n
    overall = baseline/n
    between = float(np.sum(weights * (means-overall)**2))
    within = float(sum(np.sum((a[idx, blocks.index(16)]-m)**2) for idx,m in zip(prompt_rows, means))/n)
    result['b16_variance'] = {'between_prompt': between, 'within_prompt': within,
                              'within_fraction': within/(within+between) if within+between else None}
    # Conservative switching diagnostic: adjacent states with disjoint argmax sets.
    pairs = switches = 0
    for idx in prompt_rows:
        idx = sorted(idx, key=lambda i: eligible[i]['cycle'])
        for left, right in zip(idx, idx[1:]):
            pairs += 1
            switches += not bool(set(np.flatnonzero(a[left] == a[left].max())) & set(np.flatnonzero(a[right] == a[right].max())))
    result['argmax_set_switching'] = {'sampled_adjacent_pairs': pairs, 'disjoint_best_block_sets': switches,
                                     'rate': switches/pairs if pairs else None,
                                     'note': 'sampled states need not be consecutive decoding cycles; ties retained'}
    result['checks'] = {k: sum(r.get(k, 0) for r in rows) for k in
                        ('reverse_order_checked', 'canonical_checked', 'canonical_disagreements', 'truncated_verify_checks')}
    result['source_results'] = {s: {'states': sum(r['source'] == s for r in eligible),
        'b16_mean_accepted': float(a[[i for i,r in enumerate(eligible) if r['source']==s], blocks.index(16)].mean())}
        for s in sorted({r['source'] for r in eligible})}
    if by_source:
        for source in result['source_results']:
            subset = summarize([r for r in rows if r['source'] == source], by_source=False)
            result['source_results'][source].update({k:subset[k] for k in ('prompts','frontiers','b16_variance','argmax_set_switching')})
    return result


def markdown_report(result):
    lines=['# Actual block-size headroom diagnostic', '', result['scope'], '',
           f"States: {result['states']}; eligible: {result['eligible_states']}.", '']
    if not result['eligible_states']:
        return '\n'.join(lines)+'\n'
    lines += [f"Prompts with eligible states: {result['prompts']}. Mean B16 accepted drafts: {result['mean_b16_accepted']:.4f}.", '',
              '| Retention target | Global budget / retention | Prompt-oracle budget / retention | Cycle-oracle budget / retention | Cycle budget reduction vs prompt |',
              '| --- | --- | --- | --- | --- |']
    for i in range(len(result['frontiers']['global']['points'])):
        points=[result['frontiers'][name]['points'][i] for name in ('global','per_prompt','per_cycle')]
        display=[f"{p['mean_budget']:.3f} / {100*p['retention']:.2f}%" if p['feasible'] else 'infeasible' for p in points]
        reduction=1-points[2]['mean_budget']/points[1]['mean_budget']
        lines.append(f"| {100*points[0]['target_retention']:.0f}% | {' | '.join(display)} | {100*reduction:.2f}% |")
    lines += ['', 'Budgets count proposed draft tokens (B−1). Ratios use summed accepted tokens and budgets. Exact integer choices may overshoot the retention target.', '',
              '| Source | Eligible states | Mean B16 A | Prompt budget at 96% | Cycle budget at 96% |',
              '| --- | --- | --- | --- | --- |']
    for source, sub in result['source_results'].items():
        if 'frontiers' in sub:
            pp=sub['frontiers']['per_prompt']['points'][2]
            cp=sub['frontiers']['per_cycle']['points'][2]
            lines.append(f"| {source} | {sub['states']} | {sub['b16_mean_accepted']:.3f} | {pp['mean_budget']:.3f} | {cp['mean_budget']:.3f} |")
    lines += ['', '## Checks and limits', '',
              '```json', json.dumps(result['checks'],indent=2), '```', '',
              'These oracles see true outcomes from the analyzed sample. They do not demonstrate that causal inputs can predict the choices. Prompt-level results use sampled shared B16 states, not independent full-response rollouts. Timing and throughput were not measured. Source-balanced training diagnostics are not held-out generalization results. Canonical disagreements, if nonzero, require numerical investigation.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    rows = [r for path in sorted(args.run.glob('prompt_*.json')) for r in json.loads(path.read_text())['states']]
    result = summarize(rows)
    (args.run / 'headroom_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.run / 'headroom_report.md').write_text(markdown_report(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cycle_order','prompt_order','frontiers')}, indent=2))


if __name__ == '__main__':
    main()
