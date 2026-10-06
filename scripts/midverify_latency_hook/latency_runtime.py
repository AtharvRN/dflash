"""Native-layer, paged-FlashInfer replay with explicitly timed split/packing.

No scheduler changes. Prefixes and B16 candidates are fixed snapshots. Every
policy computes confidence/probe scores at runtime. Target graphs capture only
model segments, not decisions or metadata, and are exact-shape replay graphs.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from dflash.midverify_latency import packed_indices, compact_indices, accepted_from_top1
from dflash.predraft_latency import fixed_width, frozen_lengths, replay_offsets
from scripts.audit_block_headroom import sha256
from scripts.profile_sglang_latency import atomic_json, distribution
from scripts.train_midverify_cascade import build_probe
from scripts.train_midverify_probe import make_features
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.dflash_info import DFlashRaggedVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

CAPTURE_DIAGNOSTIC_LOGITS = False


def cuda(x, dtype=None):
    return torch.as_tensor(x, dtype=dtype, device='cuda')


def difference(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError('Invalid numerical comparison')
    return {'bitwise': torch.equal(a, b), 'max_abs': float((a-b).abs().max()),
            'relative_l2': float((a-b).norm() / b.norm().clamp_min(1e-12))}


class Timer:
    def __init__(self, enabled=True):
        self.enabled, self.rows = enabled, []

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        yield
        end.record()
        self.rows.append((name, start, end, (time.perf_counter()-begin)*1000))

    def finish(self):
        torch.cuda.synchronize()
        return {name: {'stream_ms': start.elapsed_time(end), 'host_ms': host}
                for name, start, end, host in self.rows}


class Snapshot:
    def __init__(self, worker, rows):
        self.worker, self.rows = worker, rows
        self.runner = worker.target_worker.model_runner
        self.draft = worker.draft_model_runner
        self.n = len(rows)
        self.lengths = [r['prefix_length'] for r in rows]
        self.lengths_gpu = cuda(self.lengths, torch.int32)
        self.lengths_cpu = torch.tensor(self.lengths, dtype=torch.int32)
        self.blocks = cuda([[r['prefix_token_ids'][-1], *r['draft_ids']] for r in rows], torch.long)
        self.pool, self.alloc = self.runner.req_to_token_pool, self.runner.token_to_kv_pool_allocator
        if self.alloc.page_size != 1 or worker.use_compact_draft_cache:
            raise ValueError('This test requires existing non-windowed page-size1 path')
        self.before = (self.pool.available_size(), self.alloc.available_size())
        self.requests = [SimpleNamespace(req_pool_idx=None) for _ in rows]
        reqs = self.pool.alloc(self.requests)
        self.slots = self.alloc.alloc(sum(self.lengths) + self.n*16)
        if reqs is None or self.slots is None:
            raise RuntimeError('Insufficient snapshot cache capacity')
        self.reqs = cuda(reqs, torch.long)
        self.prefix_slots, cursor = [], 0
        self.saved_tables = []
        for j, length in enumerate(self.lengths):
            slot = self.slots[cursor:cursor+length]
            self.prefix_slots.append(slot)
            cursor += length
            # Native draft and target usually share the request mapping; handle either case.
            self.saved_tables.append((self.pool.req_to_token[reqs[j], :length+16].clone(),
                                      self.draft.req_to_token_pool.req_to_token[reqs[j], :length+16].clone()))
            self.pool.req_to_token[reqs[j], :length] = slot.int()
            self.draft.req_to_token_pool.req_to_token[reqs[j], :length] = slot.int()
        self.suffix = self.slots[cursor:].reshape(self.n, 16)
        for j, length in enumerate(self.lengths):
            self.pool.req_to_token[reqs[j], length:length+16] = self.suffix[j].int()
            self.draft.req_to_token_pool.req_to_token[reqs[j], length:length+16] = self.suffix[j].int()
        self.positions = self.lengths_gpu.long()[:, None] + torch.arange(16, device='cuda')[None]
        self.anchor_top1 = []
        self.raw_latest = []
        self.prefill()
        self.raw_latest = torch.cat(self.raw_latest)
        if self.raw_latest.shape != (self.n, 12800):
            raise AssertionError('Latest committed-token capture has wrong shape')
        self.fingerprint_locs = torch.stack([s[-1] for s in self.prefix_slots])
        self.prefix_fingerprint = [self.runner.token_to_kv_pool.get_key_buffer(i)[self.fingerprint_locs].clone()
                                   for i in range(36)]

    def prefill(self):
        # Bounded chunks avoid full-prefix vocabulary/activation peaks at C128.
        for lo in range(0, self.n, 4):
            hi = min(lo+4, self.n)
            lengths = self.lengths[lo:hi]
            ids = cuda([t for r in self.rows[lo:hi] for t in r['prefix_token_ids'][:-1]], torch.long)
            pos = torch.cat([torch.arange(n, device='cuda') for n in lengths])
            slots = torch.cat(self.prefix_slots[lo:hi])
            lens = cuda(lengths, torch.int32)
            fb = ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=hi-lo,
                input_ids=ids, positions=pos, req_pool_indices=self.reqs[lo:hi],
                seq_lens=lens, seq_lens_cpu=self.lengths_cpu[lo:hi], seq_lens_sum=sum(lengths),
                out_cache_loc=slots, extend_num_tokens=len(ids), extend_seq_lens=lens,
                extend_prefix_lens=torch.zeros_like(lens), extend_start_loc=lens.cumsum(0).int()-lens,
                extend_prefix_lens_cpu=[0]*len(lengths), extend_seq_lens_cpu=lengths,
                extend_logprob_start_lens_cpu=[0]*len(lengths),
                capture_hidden_mode=CaptureHiddenMode.FULL,
                num_token_non_padded_cpu=len(ids))
            out = self.runner.forward(fb).logits_output
            self.anchor_top1.extend(out.next_token_logits.argmax(-1).tolist())
            self.raw_latest.append(out.hidden_states[lens.long().cumsum(0)-1].clone())
            self.worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=out.hidden_states, cache_loc=slots, positions=pos)
        torch.cuda.synchronize()

    def fb(self, lens, noise=False, blocks=None):
        lens = np.asarray(lens, dtype=np.int32)
        idx = cuda(packed_indices(lens), torch.long)
        tokens = self.blocks if blocks is None else blocks
        if noise:
            tokens = torch.full_like(tokens, int(self.worker._mask_token_id))
            tokens[:, 0] = self.blocks[:, 0]
        ids, pos = tokens.flatten()[idx], self.positions.flatten()[idx]
        spec = DFlashRaggedVerifyInput(draft_token=ids, positions=pos,
            draft_token_lens=cuda(lens, torch.int32), draft_token_num=int(lens.max()),
            total_draft_tokens=len(idx), num_tokens_per_batch=int(lens.max()), disable_cuda_graph=True,
            capture_hidden_mode=CaptureHiddenMode.NULL if noise else CaptureHiddenMode.FULL)
        fb = ForwardBatch(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=self.n,
            input_ids=ids, positions=pos, req_pool_indices=self.reqs,
            seq_lens=self.lengths_gpu, seq_lens_cpu=self.lengths_cpu,
            seq_lens_sum=sum(self.lengths), out_cache_loc=self.suffix.flatten()[idx],
            spec_info=spec, spec_algorithm=SpeculativeAlgorithm.DFLASH,
            capture_hidden_mode=spec.capture_hidden_mode, num_token_non_padded_cpu=len(idx))
        if noise:
            fb.input_embeds = self.runner.model.model.embed_tokens(ids)
        return fb, idx

    def close(self):
        for i in range(36):
            if not torch.equal(self.prefix_fingerprint[i], self.runner.token_to_kv_pool.get_key_buffer(i)[self.fingerprint_locs]):
                raise AssertionError('A replay overwrote a committed target prefix')
        for j, length in enumerate(self.lengths):
            self.pool.req_to_token[self.reqs[j], :length+16] = self.saved_tables[j][0]
            self.draft.req_to_token_pool.req_to_token[self.reqs[j], :length+16] = self.saved_tables[j][1]
        self.alloc.free(self.slots)
        for request in self.requests:
            self.pool.free(request)
        if (self.pool.available_size(), self.alloc.available_size()) != self.before:
            raise AssertionError('Snapshot allocation leak')


def target_range(runner, fb, begin, end, state=None):
    model = runner.model.model
    if begin == 0:
        h, residual, aux = model.embed_tokens(fb.input_ids), None, []
    else:
        h, residual, aux = state[0], state[1], list(state[2:])
    with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        for i in range(begin, end):
            if i in model.layers_to_capture:
                aux.append(h+residual if residual is not None else h)
            h, residual = model.layers[i](fb.positions, h, fb, residual)
        if end < 36:
            return (h, residual, *aux)
        hidden, _ = model.norm(h, residual)
        out = runner.model.logits_processor(fb.input_ids, hidden, runner.model.lm_head, fb, aux)
        result = (hidden, out.hidden_states, out.next_token_logits.argmax(-1))
        return (*result, out.next_token_logits) if CAPTURE_DIAGNOSTIC_LOGITS else result


class Segment:
    serial = 100

    def __init__(self, runner, fb, begin, end, mode, state=None):
        self.runner, self.fb, self.begin, self.end = runner, fb, begin, end
        self.state = tuple(x.clone() for x in state) if state is not None else None
        self.mode, self.graph = mode, None
        if mode == 'graph':
            Segment.serial += 1
            fb.spec_info.num_tokens_per_batch = Segment.serial
            backend = runner.attn_backend
            backend.init_forward_metadata_out_graph(fb, in_capture=True)
            # No decisions, planning, or hidden D2H transfer is captured.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if state is not None:
                        for dst, src in zip(self.state, state):
                            dst.copy_(src)
                    self.result = target_range(runner, fb, begin, end, self.state)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if state is not None:
                for dst, src in zip(self.state, state):
                    dst.copy_(src)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.result = target_range(runner, fb, begin, end, self.state)

    def prepare(self, fb, state=None):
        if fb.input_ids.shape != self.fb.input_ids.shape:
            raise AssertionError('Exact-shape graph cache miss; benchmark cannot silently pad')
        for key in ('input_ids', 'positions', 'out_cache_loc'):
            getattr(self.fb, key).copy_(getattr(fb, key))
        self.fb.spec_info.draft_token_lens.copy_(fb.spec_info.draft_token_lens)
        if state is not None:
            for dst, src in zip(self.state, state):
                dst.copy_(src)
        if self.mode == 'graph':
            self.runner.attn_backend.init_forward_metadata_out_graph(self.fb)
        else:
            self.runner.attn_backend.init_forward_metadata(self.fb)

    def forward(self):
        if self.graph:
            self.graph.replay()
        else:
            self.result = target_range(self.runner, self.fb, self.begin, self.end, self.state)
        return self.result

    def close(self):
        # Snapshot-specific graph wrappers must not accumulate across hundreds
        # of full-cohort batches. Every replay is synchronized before cleanup.
        if self.graph is not None:
            self.graph = None
            backend = self.runner.attn_backend
            key = backend._prefill_cuda_graph_metadata_key(self.fb.batch_size, self.fb.spec_info)
            backend.prefill_cuda_graph_metadata.pop(key)


class Policy:
    def __init__(self, directory, name):
        payload = torch.load(directory / (name+'.pt'), map_location='cpu', weights_only=False)
        self.name, self.setting = name, payload['setting']
        self.model = build_probe(payload['input_width']).cuda().eval()
        self.model.load_state_dict(payload['model'])
        self.mean = cuda(payload['config']['confidence_mean'], torch.float64)
        self.std = cuda(payload['config']['confidence_std'], torch.float64)

    def confidence(self, stats):
        return ((stats.double()-self.mean)/self.std).float()

    def front(self, stats):
        threshold = self.setting['stage0_threshold']
        if threshold is None:
            return torch.full((len(stats),), 16, device='cuda', dtype=torch.int32)
        return (stats[:, :, 0].double() >= threshold).int().cumprod(1).sum(1).int()+1

    def end(self, features, front):
        q = self.model(features).squeeze(-1).sigmoid()
        threshold = self.setting['stage1_threshold']
        k = torch.full_like(front, 16) if threshold is None else (q.double() >= threshold).int().cumprod(1).sum(1).int()+1
        return torch.minimum(front, k), q


class PredraftPolicy:
    """Only latest committed-token features, before any current draft forward."""
    def __init__(self, directory, setting):
        from scripts.train_predraft_soft_supervision import build_model
        self.setting = setting
        payload = torch.load(directory / setting['checkpoint'], map_location='cpu', weights_only=False)
        if (payload['seed'] != setting['seed'] or payload['arm'] != setting['arm']
                or payload['selected_update'] != setting['selected_update']):
            raise ValueError('Checkpoint identity differs from frozen bundle')
        self.model = build_model().cuda().eval()
        self.model.load_state_dict(payload['model'])

    def lengths(self, fused):
        from scripts.train_predraft_soft_supervision import policy_scores
        scores = policy_scores(self.model(fused), self.setting['arm'])
        return frozen_lengths(scores, self.setting['threshold'])


class RawConfidencePolicy:
    def __init__(self, setting):
        if setting['kind'] != 'candidate_logprob_threshold':
            raise ValueError('Unknown confidence rule')
        self.setting = setting

    def lengths(self, candidate_logprobs):
        # No survival multiplication: stop at the first low-confidence token.
        return frozen_lengths(candidate_logprobs, self.setting['threshold'])


def predraft_fused(snapshot):
    # Charge recomputing fc/norm and the FP16 cache quantization used in training.
    return snapshot.draft.model.project_target_hidden(snapshot.raw_latest).half().float()


def draft_forward(snapshot, timer, confidence=True, block_size=16):
    with timer.phase('draft_setup_and_metadata'):
        fb, _ = snapshot.fb(np.full(snapshot.n, block_size, dtype=np.int32), noise=True)
        snapshot.draft.attn_backend.init_forward_metadata(fb)
    with timer.phase('draft_transformer'):
        with forward_context(ForwardContext(attn_backend=snapshot.draft.attn_backend)):
            output = snapshot.draft.model(fb.input_ids, fb.positions, fb, input_embeds=fb.input_embeds)
    with timer.phase('draft_projection_and_confidence'):
        h = output.hidden_states.reshape(snapshot.n, block_size, -1)[:, 1:].reshape(-1, 2560)
        weight = snapshot.runner.model.lm_head.weight[:snapshot.runner.model.config.vocab_size]
        logits = (h @ weight.T).float()
        predictions = logits.argmax(-1).reshape(snapshot.n, block_size-1)
        stats = None
        if confidence:
            logprob = logits.log_softmax(-1)
            candidate_logprob = logprob.gather(1, snapshot.blocks[:, 1:].reshape(-1, 1)).squeeze(1)
            if confidence == 'logprob_only':
                stats = candidate_logprob.reshape(snapshot.n, 15)
            else:
                chosen = logits.gather(1, snapshot.blocks[:, 1:].reshape(-1, 1)).squeeze(1)
                stats = torch.stack([candidate_logprob,
                    -(logprob.exp()*logprob).sum(-1), chosen-logits.amax(-1)], -1).reshape(snapshot.n, 15, 3)
    return stats, predictions


class Case:
    def __init__(self, snapshot, case, mode, policies):
        self.s, self.case, self.mode, self.policies = snapshot, case, mode, policies
        self.segments, self.preparing = [], True
        self.run(False)
        self.preparing = False

    def close(self):
        for segment in self.segments:
            segment.close()

    def segment(self, index, fb, begin, end, timer, state=None):
        if self.preparing:
            self.segments.append(Segment(self.s.runner, fb, begin, end, self.mode, state))
        seg = self.segments[index]
        with timer.phase(f'target_metadata_{begin}_{end}'):
            seg.prepare(fb, state)
        with timer.phase(f'target_layers_{begin}_{end}'):
            result = seg.forward()
        return result

    def run(self, timed=True):
        s, timer, case = self.s, Timer(timed), self.case
        torch.cuda.synchronize()
        begin = time.perf_counter()
        predraft = case.startswith('predraft_')
        if predraft:
            with timer.phase('predraft_fusion_and_predictor'):
                k_gpu = self.policies[case].lengths(predraft_fused(s))
            with timer.phase('predraft_lengths_device_to_host'):
                front = k_gpu.cpu().numpy()
        draft_size = fixed_width(case) if case.endswith('_redraft') else 16
        confidence = 'logprob_only' if case == 'raw_confidence' else not (case.startswith('fixed') or predraft)
        stats, draft_top = draft_forward(s, timer, confidence=confidence, block_size=draft_size)
        with timer.phase('stage0_probe_and_pack'):
            blocks = s.blocks
            if case.endswith('_redraft'):
                blocks = blocks.clone()
                blocks[:, 1:draft_size] = draft_top
            if case.startswith('fixed'):
                front = np.full(s.n, fixed_width(case), dtype=np.int32)
                policy, k_gpu = None, cuda(front, torch.int32)
            elif predraft:
                policy = None
            elif case == 'raw_confidence':
                policy = None
                k_gpu = self.policies[case].lengths(stats)
                front = k_gpu.cpu().numpy()
            else:
                policy = self.policies['cascade' if case == 'cascade' else 'target_free']
                k_gpu = policy.front(stats)
                e = s.runner.model.lm_head.weight[s.blocks[:, 1:]].float()
                e = e/e.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                conf = policy.confidence(stats)
                if case != 'cascade':
                    k_gpu, _ = policy.end(torch.cat([e, conf], -1), k_gpu)
                front = k_gpu.cpu().numpy()
            fb, front_idx = s.fb(front, blocks=blocks)
        if case in ('cascade', 'target_free_split'):
            state = self.segment(0, fb, 0, 6, timer)
            with timer.phase('stage1_probe'):
                if case == 'cascade':
                    # Only h_(j-1) inside the already-admitted prefix is available.
                    hidden = state[0]+state[1]
                    starts = cuda(np.r_[0, np.cumsum(front)[:-1]], torch.long)
                    rel = torch.arange(15, device='cuda')[None]
                    at = (starts[:, None]+rel).minimum(starts[:, None]+k_gpu[:, None]-1)
                    h = hidden[at]
                    features = make_features(h, s.runner.model.lm_head.weight[s.blocks[:, 1:]])
                    end_gpu, _ = policy.end(torch.cat([features, conf], -1), k_gpu)
                else:
                    end_gpu = k_gpu
            with timer.phase('suffix_compaction_and_pack'):
                end = end_gpu.cpu().numpy()
                gather = cuda(compact_indices(front, end), torch.long)
                compact = tuple(x.index_select(0, gather) for x in state)
                final_fb, final_idx = s.fb(end, blocks=blocks)
            result = self.segment(1, final_fb, 6, 36, timer, compact)
        else:
            end, end_gpu, final_fb, final_idx = front, k_gpu, fb, front_idx
            result = self.segment(0, fb, 0, 36, timer)
        with timer.phase('acceptance_and_commit_indices'):
            top1 = torch.full((s.n*16,), -1, device='cuda', dtype=torch.long)
            top1[final_idx] = result[2]
            top1 = top1.reshape(s.n, 16)
            match = (top1[:, :15] == blocks[:, 1:]) & (torch.arange(15, device='cuda')[None] < end_gpu[:, None]-1)
            accepted = match.int().cumprod(1).sum(1).int()
            committed = accepted+1
            bonus = top1.gather(1, accepted.long()[:, None]).squeeze(1)
            dense_valid = (torch.arange(16, device='cuda')[None] < committed[:, None]).reshape(-1)
            live = dense_valid[final_idx]
            locations = final_fb.out_cache_loc[live]
            hidden = result[1][live]
            positions = final_fb.positions[live]
            new_seq_lens = s.lengths_gpu+committed
        with timer.phase('draft_kv_upkeep'):
            s.worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=hidden, cache_loc=locations, positions=positions)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter()-begin)*1000
        phases = timer.finish() if timed else {}
        # These copies/diagnostics are OUTSIDE cycle timing.
        self.last = {'front': front.tolist(), 'end': end.tolist(), 'accepted': accepted.tolist(),
            'bonus': bonus.tolist(), 'top1': top1.tolist(), 'new_seq_lens': new_seq_lens.tolist(),
            'candidate_blocks': blocks.tolist(),
            'draft_top1_changed_positions': int((draft_top != s.blocks[:, 1:draft_size]).sum()),
            'mean_front': float(np.mean(front)), 'mean_end': float(np.mean(end)),
            'committed_tokens': int(committed.sum()), 'wall_ms': wall_ms, 'phases': phases,
            'row_layer_proxy': float(np.mean((6*np.asarray(front)+30*np.asarray(end))/36)) if case == 'cascade' else float(np.mean(end))}
        if getattr(s, 'native_candidate_replay', False) and not case.endswith('_redraft') and self.last['draft_top1_changed_positions']:
            raise AssertionError('Fresh B16 candidates changed within a frozen native replay snapshot')
        self.last_hidden, self.last_aux, self.last_fb = result[0], result[1], final_fb
        self.last_logits = result[3] if CAPTURE_DIAGNOSTIC_LOGITS else None
        return self.last


def detailed_correctness(case, original, graph_h, graph_aux, graph_logits, ref_h, ref_aux, ref_logits, fb, idx):
    """Diagnostic-only controls; no guard changes and no performance claims."""
    runner, backend = case.s.runner, case.s.runner.attn_backend
    backend.init_forward_metadata(fb)
    repeated = target_range(runner, fb, 0, 36)
    report = {'eager_repeat_hidden': difference(repeated[0], ref_h),
              'eager_repeat_features': difference(repeated[1], ref_aux),
              'eager_repeat_top1_differences': int((repeated[2] != ref_logits.argmax(-1)).sum()),
              'captured_feature_layers': {str(layer): difference(graph_aux[:,j*2560:(j+1)*2560], ref_aux[:,j*2560:(j+1)*2560])
                                         for j,layer in enumerate([2,10,18,26,34])}}
    changed = (graph_logits.argmax(-1) != ref_logits.argmax(-1)).nonzero().flatten()
    tokens = []
    for packed in changed[:64].tolist():
        dense = int(idx[packed]); request, position = divmod(dense,16)
        g, e = int(graph_logits[packed].argmax()), int(ref_logits[packed].argmax())
        tokens.append({'request': request, 'source_row': case.s.rows[request]['row'],
            'prompt_id': case.s.rows[request]['prompt_id'], 'cycle': case.s.rows[request]['cycle'],
            'query_position': position, 'graph_token': g, 'eager_token': e,
            'graph_logit_pair': graph_logits[packed, [g,e]].float().tolist(),
            'eager_logit_pair': ref_logits[packed, [g,e]].float().tolist()})
    report['changed_tokens'] = tokens
    report['changed_tokens_total'] = len(changed)
    delta = (graph_h.float()-ref_h.float()).norm(dim=-1)/ref_h.float().norm(dim=-1).clamp_min(1e-12)
    report['row_hidden_relative_l2_quantiles'] = torch.quantile(delta, cuda([0,.5,.9,.99,1.])).tolist()
    case.run(False)
    report['same_mode_repeat_hidden'] = difference(case.last_hidden, graph_h)
    report['same_mode_repeat_features'] = difference(case.last_aux, graph_aux)
    report['same_mode_repeat_top1_differences'] = int((np.array(case.last['top1']) != np.array(original['top1'])).sum())
    report['same_mode_repeat_acceptance_differences'] = int((np.array(case.last['accepted']) != np.array(original['accepted'])).sum())
    if case.mode == 'graph' and len(case.segments) == 1:
        from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata
        segment = case.segments[0]
        segment.prepare(fb)
        saved_metadata = backend.forward_metadata
        key = backend._prefill_cuda_graph_metadata_key(segment.fb.batch_size, segment.fb.spec_info)
        backend.forward_metadata = PrefillMetadata(backend.prefill_cuda_graph_metadata[key], False, False)
        try:
            uncaptured = target_range(runner, segment.fb, 0, 36)
            report['uncaptured_graph_wrapper_vs_graph_hidden'] = difference(uncaptured[0], graph_h)
            report['uncaptured_graph_wrapper_vs_eager_hidden'] = difference(uncaptured[0], ref_h)
            report['uncaptured_graph_wrapper_vs_graph_top1_differences'] = int((uncaptured[2] != graph_logits.argmax(-1)).sum())
        finally:
            backend.forward_metadata = saved_metadata
    report['prefix_last_key_fingerprints_unchanged'] = all(torch.equal(case.s.prefix_fingerprint[i],
        runner.token_to_kv_pool.get_key_buffer(i)[case.s.fingerprint_locs]) for i in range(36))
    report['prefix_slot_mapping_unchanged'] = all(torch.equal(case.s.pool.req_to_token[case.s.reqs[j], :length],
        case.s.prefix_slots[j].int()) for j,length in enumerate(case.s.lengths))
    return report


def correctness(case, audit_path=None, detailed=False):
    """Independently run all 36 layers on the actual final prefix, same KV state."""
    case.run(False)
    final_h, final_aux = case.last_hidden.clone(), case.last_aux.clone()
    original = case.last
    final_logits = case.last_logits.clone() if detailed else None
    final_top = np.array(case.last['top1'])
    locations = case.last_fb.out_cache_loc
    pool = case.s.runner.token_to_kv_pool
    kv = [(i, pool.get_key_buffer(i)[locations].clone(), pool.get_value_buffer(i)[locations].clone()) for i in (0, 5, 35)]
    fb, idx = case.s.fb(np.asarray(case.last['end'], dtype=np.int32), blocks=cuda(case.last['candidate_blocks'], torch.long))
    case.s.runner.attn_backend.init_forward_metadata(fb)
    reference = target_range(case.s.runner, fb, 0, 36)
    ref_h, ref_aux, ref_top = reference[:3]
    dense = np.full((case.s.n, 16), -1, dtype=np.int64)
    dense.reshape(-1)[idx.cpu().numpy()] = ref_top.cpu().numpy()
    ref_a, ref_b = accepted_from_top1(case.last['candidate_blocks'], dense, case.last['end'])
    report = {'hidden': difference(final_h, ref_h), 'draft_features': difference(final_aux, ref_aux),
        'top1_differences': int((final_top != dense).sum()),
        'acceptance_differences': int((np.array(case.last['accepted']) != ref_a).sum()),
        'bonus_differences': int((np.array(case.last['bonus']) != ref_b).sum()),
        'kv': {str(i): {'k': difference(k, pool.get_key_buffer(i)[locations]),
                        'v': difference(v, pool.get_value_buffer(i)[locations])} for i, k, v in kv}}
    # Verify custom layer traversal agrees with native model dispatch.
    native = case.s.runner.forward(fb).logits_output
    report['manual_vs_native_aux'] = difference(ref_aux, native.hidden_states)
    report['manual_vs_native_top1'] = int((ref_top != native.next_token_logits.argmax(-1)).sum())
    report['numerical_guard_exceeded'] = max(report['hidden']['relative_l2'], report['draft_features']['relative_l2']) > .02
    report['case'], report['mode'] = case.case, case.mode
    report['accepted'], report['reference_accepted'] = original['accepted'], ref_a.tolist()
    report['bonus'], report['reference_bonus'] = original['bonus'], ref_b.tolist()
    if audit_path is not None and (detailed or report['numerical_guard_exceeded']):
        atomic_json(audit_path, report)
    if detailed:
        report['details'] = detailed_correctness(case, original, final_h, final_aux, final_logits,
            ref_h, ref_aux, reference[3], fb, idx)
    if audit_path is not None and (detailed or report['numerical_guard_exceeded']):
        atomic_json(audit_path, report)
    if report['numerical_guard_exceeded']:
        raise AssertionError('Segmented forward exceeds 2% numerical diagnostic guard')
    if case.case == 'target_free_split' and case.mode == 'eager' and not report['hidden']['bitwise']:
        raise AssertionError('No-prune same-shape eager split must be bitwise identical')
    if not report['manual_vs_native_aux']['bitwise'] or report['manual_vs_native_top1']:
        raise AssertionError('Manual traversal differs from native model')
    return report


@torch.no_grad()
def run(worker, config_path):
    global CAPTURE_DIAGNOSTIC_LOGITS
    config = json.loads(Path(config_path).read_text())
    CAPTURE_DIAGNOSTIC_LOGITS = config.get('diagnostic_offset') is not None
    output = Path(config['output'])
    started = time.monotonic()
    snapshot = None
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = json.loads((output/'states.json').read_text())
        directory = Path(config['policies'])
        predraft_suite = config.get('suite') == 'predraft_verification_trim'
        if predraft_suite:
            from scripts.export_predraft_latency_bundle import verify
            verify(directory)
            bundle = json.loads((directory/'bundle.json').read_text())
            policies = {name: (RawConfidencePolicy(setting) if name == 'raw_confidence'
                              else PredraftPolicy(directory, setting))
                        for name, setting in bundle['policies'].items()}
        else:
            policies = {'target_free': Policy(directory, 'candidate_confidence_seed913_r0.99'),
                        'cascade': Policy(directory, 'target_candidate_confidence_seed913_r0.99')}
        runner = worker.target_worker.model_runner
        if list(runner.model.model.layers_to_capture) != [2, 10, 18, 26, 34]:
            raise ValueError('DFlash hidden-layer capture mapping differs')
        if len(runner.model.model.layers) != 36:
            raise ValueError('This bounded implementation requires Qwen3-4B')
        runner.attn_backend.init_cuda_graph_state(128, 2048)
        results = []
        for concurrency in config['concurrencies']:
            for offset in replay_offsets(len(rows), concurrency, config.get('diagnostic_offset')):
                batch = rows[offset:offset+concurrency]
                snapshot = Snapshot(worker, batch)
                if predraft_suite:
                    # Generate once using this engine and batch shape. All trim
                    # cases subsequently pay a full draft and preserve these IDs.
                    _, native_ids = draft_forward(snapshot, Timer(False), confidence=False)
                    cached = snapshot.blocks[:, 1:].clone()
                    snapshot.blocks[:, 1:] = native_ids
                    snapshot.native_candidate_replay = True
                    fused = predraft_fused(snapshot)
                    cached_fused = cuda(np.load(directory/'cached_fused.npy', allow_pickle=False)[offset:offset+len(batch)]).float()
                    native_audit = {'cached_vs_native_candidate_differences': int((native_ids != cached).sum()),
                        'cached_vs_native_fused': difference(fused, cached_fused),
                        'policies': {name: {'native_lengths': p.lengths(fused).tolist(),
                                           'cached_lengths': p.lengths(cached_fused).tolist()}
                                     for name, p in policies.items() if name.startswith('predraft_')}}
                    atomic_json(output/f'predraft_feature_audit_c{concurrency}_offset{offset}.json', native_audit)
                for mode in config['modes']:
                    cases = {}
                    for name in config['cases']:
                        current = Case(snapshot, name, mode, policies)
                        audit = correctness(current, output/f'correctness_c{concurrency}_offset{offset}_{mode}_{name}.json',
                                            detailed=CAPTURE_DIAGNOSTIC_LOGITS)
                        if predraft_suite and config['smoke'] and any(audit[k] for k in ('top1_differences', 'acceptance_differences', 'bonus_differences')):
                            raise AssertionError('Small-C correctness smoke failed; full replay forbidden')
                        for _ in range(config['warmups']):
                            current.run(False)
                        cases[name] = (current, audit)
                        print(json.dumps({'stage': 'ready', 'C': concurrency, 'offset': offset,
                            'mode': mode, 'case': name, 'audit': audit, 'lengths': current.last['end']}), flush=True)
                    if not predraft_suite:
                        unsplit, split = cases['target_free'][0], cases['target_free_split'][0]
                        split_control = difference(split.last_hidden, unsplit.last_hidden)
                        atomic_json(output/f'split_control_c{concurrency}_offset{offset}_{mode}.json', split_control)
                        if not split_control['bitwise']:
                            raise AssertionError('Matched no-prune split differs from unsplit in the same execution mode')
                    observations = {name: [] for name in cases}
                    clean = {name: [] for name in cases}
                    rng = random.Random(929+concurrency+offset)
                    for repeat in range(config['repeats']):
                        order = list(cases)
                        rng.shuffle(order)
                        for name in order:
                            current = cases[name][0]
                            observations[name].append(current.run(True))
                            clean[name].append(current.run(False)['wall_ms'])
                        if time.monotonic()-started > config['max_seconds']-30:
                            raise TimeoutError('Worker bounded deadline')
                    base = observations['fixed16'][0]
                    base_sum = sum(base['accepted'])
                    for name, (current, audit) in cases.items():
                        obs = observations[name]
                        if predraft_suite and not name.endswith('_redraft'):
                            # Prefix invariance against FULL verification, not
                            # merely against another forward at the same width.
                            end = np.asarray(obs[0]['end'])
                            expected_a = np.minimum(np.asarray(base['accepted']), end-1)
                            expected_bonus = np.asarray(base['top1'])[np.arange(len(batch)), expected_a]
                            prefix_mask = np.arange(16)[None] < end[:, None]
                            audit['same_candidates_vs_b16'] = {
                                'kept_top1_differences': int(((np.asarray(obs[0]['top1']) != np.asarray(base['top1'])) & prefix_mask).sum()),
                                'acceptance_differences': int((np.asarray(obs[0]['accepted']) != expected_a).sum()),
                                'bonus_differences': int((np.asarray(obs[0]['bonus']) != expected_bonus).sum())}
                            if config['smoke'] and any(audit['same_candidates_vs_b16'].values()):
                                raise AssertionError('Small-C prefix invariance against full B16 failed')
                        # Repeated snapshots must not silently change policy decisions.
                        if any(r['end'] != obs[0]['end'] or r['accepted'] != obs[0]['accepted'] for r in obs):
                            raise AssertionError('Nondeterministic decisions across replay repeats')
                        values = {'C': concurrency, 'offset': offset, 'mode': mode, 'case': name,
                            'audit': audit, 'rows': len(batch), 'prompt_ids': [r['prompt_id'] for r in batch],
                            'prefix_lengths': snapshot.lengths, 'anchor_mismatches': sum(x != int(y) for x, y in zip(snapshot.anchor_top1, snapshot.blocks[:, 0].tolist())),
                            'observations': obs, 'wall_ms': distribution([r['wall_ms'] for r in obs]),
                            'uninstrumented_cycle_ms': distribution(clean[name]),
                            'uninstrumented_cycle_samples_ms': clean[name],
                            'uninstrumented_ms_per_committed_token': float(np.mean(clean[name])/obs[0]['committed_tokens']),
                            'phase_stream_ms': {key: distribution([r['phases'][key]['stream_ms'] for r in obs]) for key in obs[0]['phases']},
                            'assessment_retention_vs_same_engine_b16': sum(obs[0]['accepted'])/base_sum if base_sum else None,
                            'mean_front': obs[0]['mean_front'], 'mean_end': obs[0]['mean_end'],
                            'wall_ms_per_committed_token': np.mean([r['wall_ms']/r['committed_tokens'] for r in obs])}
                        results.append(values)
                    atomic_json(output/'partial.json', results)
                    for current, _ in cases.values():
                        current.close()
                    del current, cases, observations
                    gc.collect()
                    torch.cuda.empty_cache()
                # Native slot reservation/free cost is separately bounded, not hidden.
                allocation = []
                for _ in range(20):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    slots = snapshot.alloc.alloc(len(batch)*16)
                    if slots is None:
                        raise RuntimeError('Allocation microcheck failed')
                    snapshot.alloc.free(slots)
                    torch.cuda.synchronize()
                    allocation.append((time.perf_counter()-begin)*1000)
                atomic_json(output/f'allocation_c{concurrency}_offset{offset}.json', distribution(allocation))
                snapshot.close()
                snapshot = None
                atomic_json(output/'progress.json', {'completed_C': concurrency, 'offset': offset,
                    'elapsed_s': time.monotonic()-started, 'result_cells': len(results)})
        atomic_json(output/'summary.json', {'config': config, 'results': results,
            'elapsed_s': time.monotonic()-started, 'torch': torch.__version__,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'interpretation': 'Replay cycle timing, not serving throughput. Models, native layers/backend and graph mode matched within comparisons. Graphs are exact-shape segment captures, not SGLang scheduler bucket integration. Fresh same-engine labels; frozen thresholds not retuned to assessment.'})
        names = ['config.json', 'states.json', 'summary.json', 'partial.json']
        if predraft_suite:
            names += sorted(p.name for p in output.glob('predraft_feature_audit_*.json'))
        names += sorted(p.name for p in output.glob('correctness_*.json'))
        atomic_json(output/'COMPLETE.json', {'passed': True,
            'binding': {name: sha256(output/name) for name in names}})
        print('MIDVERIFY_LATENCY_COMPLETE', flush=True)
    except BaseException as error:
        atomic_json(output/'FAILED.json', {'error': repr(error), 'traceback': traceback.format_exc(),
            'elapsed_s': time.monotonic()-started})
        traceback.print_exc()
        raise
    finally:
        if snapshot is not None:
            snapshot.close()
