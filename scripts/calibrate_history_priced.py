"""Freeze a retention-free priced policy on calibration; assess it separately.

calibrate --table TABLE --cost-profile PROFILE --concurrency C --runs CAL_RUNS
          --expected-prompt-ids JSON_LIST --output FROZEN.json
assess --frozen FROZEN.json --runs TEST_RUNS --expected-prompt-ids JSON_LIST
       --output REPORT.json

Both commands also require --table and --cost-profile. Profile schema is checked
by dflash.history_priced.validate_profile. Source paths are relative to profile.
Assessment cannot change rho; outputs never overwrite prior evidence.

Cost profile required fields:
  schema_version=1, units="ms", scope="whole_cycle",
  cost_basis="uniform_batch_cycle", model_identity=<same as value table>,
  costs_ms={"64": {"4": <measured ms>, "8": ..., "12": ..., "16": ...}},
  source_hashes={<measurement-file path>: <SHA256>},
  engine, engine_revision, gpu, dtype, attention_backend, graph_mode,
  context_workload, timing_boundaries (nonempty strings describing the run).
Use isolated fixed-width measurements, not counterfactual collection timings
or serving tokens/s substituted for cycle costs. Every arm must be measured.

For greedy rollout, pass dflash_history.py rollout --mode priced --priced-policy
FROZEN.json --cost-profile PROFILE --concurrency C plus the usual model/data
arguments. This is reference Transformers execution, not SGLang integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.history_policy import HistoryValueTable
from dflash.history_priced import calibrate_priced, priced_report, validate_profile
from scripts.dflash_history import digest, load_runs, save


def check_coverage(paths, expected, group):
    if not isinstance(expected, list) or not expected or len(set(map(str, expected))) != len(expected):
        raise ValueError("expected prompt IDs must be a nonempty unique JSON list")
    observed = []
    for path in paths:
        config = json.loads((path/'config.json').read_text())
        if config.get('group') != group:
            raise ValueError("wrong collection group")
        summaries = json.loads((path/'summary.json').read_text())['prompts']
        if any('skipped' in r or r.get('cycle_limited') for r in summaries):
            raise ValueError("skipped or cycle-capped question")
        observed.extend(str(r['prompt_id']) for r in summaries)
    if len(observed) != len(set(observed)) or set(observed) != set(map(str, expected)):
        raise ValueError("incomplete or duplicate question coverage")
    return sorted(observed)


def run(args):
    if args.output.exists():
        raise ValueError("refusing existing output")
    table = HistoryValueTable(json.loads(args.table.read_text()))
    profile = json.loads(args.cost_profile.read_text())
    frozen = json.loads(args.frozen.read_text()) if args.command == 'assess' else None
    bindings = dict(table_sha256=digest(args.table), cost_profile_sha256=digest(args.cost_profile))
    if frozen is not None and (frozen.get('schema_version') != 1 or frozen['bindings'] != bindings):
        raise ValueError("frozen table/profile binding mismatch")
    concurrency = frozen['concurrency'] if frozen else args.concurrency
    costs = validate_profile(profile, table, concurrency)
    for path, expected_hash in profile['source_hashes'].items():
        if digest(args.cost_profile.parent / path) != expected_hash:
            raise ValueError("cost measurement source hash mismatch")
    group = 'calibration' if frozen is None else 'assessment'
    expected = json.loads(args.expected_prompt_ids.read_text())
    ids = check_coverage(args.runs, expected, group)
    if set(ids) & set(table.payload['fit_prompt_ids']):
        raise ValueError("question coverage overlaps fitting")
    rows, provenance = load_runs(args.runs)
    if (provenance['model_identity'] != table.payload['provenance']['model_identity']
            or any(r['group'] != group or str(r['prompt_id']) not in set(ids) for r in rows)):
        raise ValueError("model or group mismatch")
    result = dict(schema_version=1, bindings=bindings, concurrency=concurrency, provenance=provenance,
                  expected_prompt_ids_sha256=digest(args.expected_prompt_ids), prompt_ids=ids)
    if frozen is None:
        result['policies'] = {name: calibrate_priced(table, rows, costs, history_free=free)
                              for name, free in [('history', False), ('history_free', True)]}
        result['calibration_choice_note'] = ('RecGuide score; our rho calibration objective and histogram estimator. '
            'No retention floor. No assessment data used. Offline cost proxy only.')
    else:
        if set(ids) & set(frozen['prompt_ids']):
            raise ValueError("assessment overlaps calibration")
        result['frozen_sha256'] = digest(args.frozen)
        result['policies'] = {}
        for name, policy in frozen['policies'].items():
            report = priced_report(table, rows, costs, policy['rho'], history_free=policy['history_free'])
            best_fixed = policy['best_fixed_on_calibration']
            report.update(rho=policy['rho'], best_fixed_on_calibration=best_fixed,
                modeled_rate_ratio_vs_calibration_best_fixed=report['adaptive']['progress_per_modeled_ms'] /
                    report['fixed'][str(best_fixed)]['progress_per_modeled_ms'])
            result['policies'][name] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('calibrate', 'assess'):
        c = sub.add_parser(name)
        for flag in ('table', 'cost-profile', 'expected-prompt-ids', 'output'):
            c.add_argument('--'+flag, type=Path, required=True)
        c.add_argument('--runs', type=Path, nargs='+', required=True)
        if name == 'calibrate':
            c.add_argument('--concurrency', type=int, required=True)
        else:
            c.add_argument('--frozen', type=Path, required=True)
    args = p.parse_args()
    run(args)
    print(args.output)


if __name__ == '__main__':
    main()
