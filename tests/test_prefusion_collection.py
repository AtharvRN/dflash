import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.collect_prefusion_acceptance import select_prompts, observable_label, materialize


class PrefusionCollectionTests(unittest.TestCase):
    def test_terminal_and_cap(self):
        block = torch.arange(16)[None]
        self.assertTrue(observable_label(block, 0, {9}, 16))
        self.assertTrue(observable_label(block, 15, {99}, 16))
        self.assertFalse(observable_label(block, 9, {9}, 16))
        self.assertFalse(observable_label(block, 0, {99}, 15))
        self.assertFalse(observable_label(block, 0, {0}, 16))

    def test_fixed_prompt_groups_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train_prompt_ids.json').write_text(json.dumps({'train_prompt_ids': list(range(10))}))
            (root/'val_prompt_ids.json').write_text(json.dumps({'val_prompt_ids': list(range(10, 20))}))
            rows = [{'manifest_index': i, 'messages': [{'role': 'user', 'content': str(i)}]} for i in range(20)]
            rows[10]['messages'] = rows[0]['messages']
            (root/'messages.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            selected = select_prompts(root/'messages.jsonl', root, {'calibration_prompt_ids': [10,11,12,13]}, 8, 6, 913)
            self.assertEqual(len(selected), 14)
            self.assertNotIn(0, [r['manifest_index'] for _,r in selected])
            self.assertNotIn(10, [r['manifest_index'] for _,r in selected])
            for group, row in selected:
                self.assertEqual(group, 'train' if row['manifest_index'] < 10 else
                                 'calibration' if row['manifest_index'] < 14 else 'assessment')

    def test_materialize_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = []
            for pid, group in [(15, 'calibration'), (16, 'assessment')]:
                path = root/f'{pid}.npz'
                np.savez(path, raw_features=np.full((2,1,6), pid, np.float16),
                         features=np.full((2,1,2), pid, np.float16),
                         accepted_len=np.array([0,15]), cycle_id=np.array([0,1]))
                shards.append({'path': path, 'rows': 2, 'prompt_id': pid, 'group': group})
            self.assertEqual(materialize(shards, root/'val', 6, 2), 4)
            np.testing.assert_array_equal(np.load(root/'val'/'accepted_len.npy'), [0,15,0,15])
            np.testing.assert_array_equal(np.load(root/'val'/'calibration.npy'), [True,True,False,False])
            np.testing.assert_array_equal(np.load(root/'val'/'row_index.npy')[:,2], [15,15,16,16])
            np.testing.assert_array_equal(np.load(root/'val'/'raw_features.npy')[:,0,0], [15,15,16,16])


if __name__ == '__main__':
    unittest.main()
