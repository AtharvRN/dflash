"""Frozen post-draft policy input contract shared by replay and tests."""
def features(candidate_vectors, stats, mean, std):
    import torch
    # Match training's cached FP16 LM-head vectors and FP64 statistics scaling.
    e = candidate_vectors.half().float()
    e = e / torch.sqrt(e.square().mean(-1, keepdim=True) + 1e-6)
    conf = ((stats.double() - mean.double()) / std.double()).float()
    return torch.cat((e, conf), -1)


def settings(summary, target=.96):
    key = str(float(target))
    result = {}
    for arm in ('hard', 'hard_tv', 'hard_margin'):
        name = arm + '_seed913'
        entry = summary['models'][name]
        point = entry['operating_points'][key]['calibration']
        if point['retention'] < target - 1e-12:
            raise ValueError('Checkpoint fails calibration retention')
        result['postdraft_' + arm] = dict(kind='postdraft_candidate_confidence',
            checkpoint=name+'.pt', arm=arm, seed=913,
            selected_update=entry['selected_update'], threshold=point['threshold'])
    raw = summary['raw_confidence'][key]['calibration']
    if raw['retention'] < target - 1e-12:
        raise ValueError('Raw control fails calibration retention')
    result['raw_confidence'] = dict(kind='candidate_logprob_threshold', threshold=raw['threshold'],
        calibration=raw, input='Current full-B16 draft logits only; no target verification input')
    return result
