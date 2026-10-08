"""Layer a frozen pre-draft history controller onto the recovered ragged engine."""
import json
from pathlib import Path
import shutil

from prepare_postdraft_serving import prepare as prepare_base
from recover_legacy_ragged import digest

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    manifest = prepare_base(destination)
    changes = {}
    def edit(rel, old, new):
        path = destination/'python/sglang/srt'/rel
        source = path.read_text()
        if source.count(old) != 1:
            raise ValueError(f'History extension anchor count: {rel}: {old[:90]} = {source.count(old)}')
        entry = changes.setdefault(rel, {'before_sha256': digest(path)})
        path.write_text(source.replace(old, new))
        entry['after_sha256'] = digest(path)

    worker = 'speculative/dflash_worker_v2.py'
    edit(worker, '        self.postdraft_policy: Optional[DFlashPostdraftPolicy] = None',
         '''        self.history_policy = None
        if os.getenv("SGLANG_DFLASH_HISTORY_FROZEN"):
            from sglang.srt.speculative.dflash_history_policy import DFlashHistoryPolicy, load_artifacts
            if (server_args.tp_size != 1 or server_args.attention_backend != "flashinfer"
                or not server_args.disable_radix_cache
                or server_args.speculative_dflash_dynamic_block_size
                or server_args.speculative_dflash_predraft_entropy_policy_path
                or server_args.speculative_dflash_postdraft_policy_path
                or server_args.speculative_dflash_postdraft_logprob_threshold is not None):
                raise ValueError("History policy requires standalone greedy TP1 FlashInfer DFlash")
            table, costs, rho, concurrency = load_artifacts(
                os.environ["SGLANG_DFLASH_HISTORY_TABLE"], os.environ["SGLANG_DFLASH_HISTORY_PROFILE"],
                os.environ["SGLANG_DFLASH_HISTORY_FROZEN"])
            if max(table.blocks) != self.block_size:
                raise ValueError("History policy block capacity mismatch")
            self.history_policy = DFlashHistoryPolicy(table, costs, rho,
                capacity=self.model_runner.req_to_token_pool.req_to_token.shape[0], device=self.device)
            logger.info("Frozen history-priced policy loaded C%s rho=%s arms=%s", concurrency, rho, table.blocks)
        self.postdraft_policy: Optional[DFlashPostdraftPolicy] = None''')
    edit(worker, '        self._validate_phase1_sampling_support(model_worker_batch)',
         '''        self._validate_phase1_sampling_support(model_worker_batch)
        if self.history_policy is not None:
            si = model_worker_batch.sampling_info
            if si is not None and not si.is_all_greedy:
                raise ValueError("History-priced serving currently supports greedy only")''')
    edit(worker, '            # Target prefill: capture DFlash aux hidden states for prompt tokens.',
         '''            if self.history_policy is not None:
                fresh = torch.tensor([n == 0 for n in model_worker_batch.prefix_lens],
                                     device=self.device, dtype=torch.bool)
                self.history_policy.reset(model_worker_batch.req_pool_indices[fresh])
            # Target prefill: capture DFlash aux hidden states for prompt tokens.''')
    edit(worker, '        if (\n            forced_block_lens := self._forced_ragged_block_sizes(bs)',
         '''        if self.history_policy is not None:
            history_lens = self.history_policy.select(model_worker_batch.req_pool_indices)
            if not self._is_uniform_block_size(history_lens):
                return self._forward_batch_generation_ragged_decode(
                    model_worker_batch=model_worker_batch, draft_input=draft_input,
                    block_lens=history_lens, on_publish=on_publish)
            self._predraft_entropy_rectangular_override = int(history_lens[0].item())
        if (
            forced_block_lens := self._forced_ragged_block_sizes(bs)''')
    edit(worker, '        buckets: List[int] = []',
         '        buckets: List[int] = list(self.history_policy.arms) if self.history_policy is not None else []')
    edit(worker, '        if need_mamba_verify_commit:\n            assert seq_lens_pre_verify is not None\n            self._update_target_mamba_state_after_verify(\n                batch=model_worker_batch,\n                seq_lens_pre_verify=seq_lens_pre_verify,\n                commit_lens=commit_lens,\n            )\n        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)',
         '''        if self.history_policy is not None:
            self.history_policy.observe(model_worker_batch.req_pool_indices, block_lens, accept_len,
                                        next_token_logits, offsets, graph_real_indices,
                                        candidates=draft_tokens, output_tokens=out_tokens, packed=True)
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)''')
    edit(worker, '        if new_seq_lens is None:\n            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)',
         '''        if self.history_policy is not None:
            hist_blocks = torch.full((bs,), int(block_size), device=device, dtype=torch.int64)
            hist_offsets = torch.arange(bs+1, device=device, dtype=torch.int64)*int(block_size)
            self.history_policy.observe(model_worker_batch.req_pool_indices, hist_blocks, accept_len,
                                        logits_output.next_token_logits, hist_offsets,
                                        candidates=candidates, output_tokens=out_tokens)
        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)''')
    graph = 'model_executor/runner/decode_cuda_graph_runner.py'
    edit(graph, '                    self.model_runner.server_args.speculative_dflash_dynamic_block_size',
         '                    bool(__import__("os").getenv("SGLANG_DFLASH_HISTORY_FROZEN"))\n'
         '                    or self.model_runner.server_args.speculative_dflash_dynamic_block_size')
    edit(graph, '                dynamic_buckets: List[int] = []',
         '                dynamic_buckets: List[int] = ([4, 8, 12, 16] if __import__("os").getenv("SGLANG_DFLASH_HISTORY_FROZEN") else [])')
    for src, dst in [('serving_history_policy.py', 'dflash_history_policy.py'),
                     ('serving_history_entropy.py', 'dflash_history_entropy.py'),
                     ('history_policy.py', 'dflash_history_table.py')]:
        path = destination/'python/sglang/srt/speculative'/dst
        shutil.copy2(ROOT/'dflash'/src, path)
        changes['speculative/'+dst] = {'after_sha256': digest(path)}
    manifest['history_extension'] = {'changes': changes, 'source_sha256': digest(Path(__file__))}
    (destination/'HISTORY_EXTENSION.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
