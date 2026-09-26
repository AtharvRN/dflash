import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.block_headroom import exact_frontier, select_training_prompts, summarize
from scripts.audit_block_headroom import validate_states


class HeadroomTests(unittest.TestCase):
    def test_audit_detects_prefix_tampering_and_reference_drift(self):
        def state(prefix,cycle):
            return {'prefix_token_ids':prefix,'prefix_length':len(prefix)-1,'cycle':cycle,
                    'prefix_sha256':hashlib.sha256(np.array(prefix,dtype=np.int64).tobytes()).hexdigest(),
                    'outcomes':{'16':{'draft_ids':[7]*15,'accepted':2}}}
        a=state([1,2,3],0)
        b=state([1,2,3,7,7,8],1)
        validate_states([a,b],[16])
        with self.assertRaisesRegex(ValueError,'committed prefix'):
            validate_states([a,state([1,2,3,9,7,8],1)],[16])
        b['prefix_token_ids'][0]=99
        with self.assertRaisesRegex(ValueError,'hash'):
            validate_states([b],[16])

    def test_dp_matches_exhaustive_and_reconstructs(self):
        rng=np.random.default_rng(926)
        for _ in range(20):
            costs=rng.integers(1,8,size=(4,3))
            a=rng.integers(0,8,size=(4,3))
            targets=range(30)
            points, maximum=exact_frontier(a,costs,targets)
            brute=[(sum(costs[i,j] for i,j in enumerate(choice)),sum(a[i,j] for i,j in enumerate(choice)))
                   for choice in itertools.product(range(3),repeat=4)]
            self.assertEqual(maximum['total_accepted'],max(v for _,v in brute))
            for target,point in zip(targets,points):
                feasible=[c for c,v in brute if v>=target]
                self.assertEqual(point['feasible'],bool(feasible))
                if feasible:
                    self.assertEqual(point['total_budget'],min(feasible))

    def test_state_oracle_dominates_prompt_and_global(self):
        rows=[]
        for p in ('a','b'):
            for cycle,vals in enumerate(([1,0,0],[0,2,1])):
                rows.append({'prompt_id':p,'cycle':cycle,'source':'test','eligible':True,
                             'outcomes':{str(b):{'accepted':v} for b,v in zip([2,3,16],vals)}})
        result=summarize(rows)
        for i in range(5):
            f=result['frontiers']
            self.assertLessEqual(f['per_cycle']['points'][i]['total_budget'],f['per_prompt']['points'][i]['total_budget'])
            self.assertLessEqual(f['per_prompt']['points'][i]['total_budget'],f['global']['points'][i]['total_budget'])
        self.assertEqual(result['argmax_set_switching']['rate'],1)

    def test_incomplete_matrix_rejected(self):
        with self.assertRaises(ValueError):
            summarize([{'prompt_id':'a','cycle':i,'source':'test','eligible':True,
                        'outcomes':o} for i,o in enumerate([{'16':{'accepted':1}}, {'2':{'accepted':0},'16':{'accepted':1}}])])

    def test_selection_excludes_validation_content_and_interleaves(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sources=['nemotron','opencodeinstruct','openr1_math','evol_codealpaca']
            rows=[{'manifest_index':i,'source':s,'messages':[{'role':'user','content':str(i)}]}
                  for i,s in enumerate(sources*3)]
            rows.append({**rows[0],'manifest_index':99})
            (root/'train_prompt_ids.json').write_text(json.dumps({'train_prompt_ids':list(range(12))}))
            (root/'val_prompt_ids.json').write_text(json.dumps({'val_prompt_ids':[99]}))
            manifest=root/'manifest.jsonl'
            manifest.write_text('\n'.join(map(json.dumps,rows)))
            selected=select_training_prompts(manifest,root,2,926)
            self.assertEqual(len(selected),8)
            self.assertNotIn(0,[r['manifest_index'] for r in selected])
            self.assertEqual(len({r['source'] for r in selected[:4]}),4)


if __name__=='__main__':
    unittest.main()
