import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.grow_context_attention_cache import grow_cache


class GrowCacheTest(unittest.TestCase):
    def test_training_growth_preserves_validation_bytes_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, shard, split = root/"parent", root/"shard", root/"split"
            for path in (parent/"val", shard, split):
                path.mkdir(parents=True)
            x = np.arange(5*3*8, dtype=np.float16).reshape(5,3,8)
            y = np.array([0,15,4,8,99])
            rows = np.array([[0,0,10,0,0], [0,1,20,0,15], [0,2,10,1,4], [0,3,30,0,8]])
            arrays = {"features":x, "mask":np.ones((5,3),np.uint8), "accepted_len":y,
                      "survival":y[:,None]>=np.arange(1,16)}
            for name,array in arrays.items():
                np.save(shard/(name+".npy"),array)
            for name,array in {"features":x[[1,3]], "mask":np.ones((2,3),np.uint8),
                    "accepted_len":y[[1,3]], "row_index":rows[[1,3]], "calibration":np.array([True,False])}.items():
                np.save(parent/"val"/(name+".npy"),array)
            np.save(parent/"source_rows.npy",rows)
            source_path = root/"source.json"
            source_path.write_text(json.dumps({"source_trace_dir":str(shard),"predraft_feature_kind":"fused"}))
            (split/"manifest.json").write_text(json.dumps({"trace_dir":str(shard),"num_rows_total":4,"val":{"rows":2}}))
            (split/"train_prompt_ids.json").write_text(json.dumps({"train_prompt_ids":[10]}))
            (split/"val_prompt_ids.json").write_text(json.dumps({"val_prompt_ids":[20,30]}))
            info = {"format":"dflash_context_attention_cache_v1","input_kind":"predraft_fused",
                    "input_dim":8,"context_window":3,"num_slots":15,"train_rows":1,"val_rows":2,"val_prompts":2,
                    "seed":913,"calibration_prompt_ids":[20],"source_shards":[str(shard)],
                    "trace_manifest":str(source_path),"split_dir":str(split),
                    "hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (source_path,split/"train_prompt_ids.json",split/"val_prompt_ids.json")}}
            (parent/"manifest.json").write_text(json.dumps(info))
            grow_cache(parent, root/"grown", 2, 1, 914)
            np.testing.assert_array_equal(np.load(root/"grown/train/features.npy"),x[[0,2]])
            for path in (parent/"val").glob("*.npy"):
                self.assertEqual(path.read_bytes(),(root/"grown/val"/path.name).read_bytes())
                self.assertEqual(path.stat().st_ino,(root/"grown/val"/path.name).stat().st_ino)
            info = json.loads((root/"grown/manifest.json").read_text())
            self.assertEqual(info["validation_calibration_seed"],914)
            np.save(parent/"val/row_index.npy",rows[[3,1]])
            with self.assertRaisesRegex(ValueError,"ordering"):
                grow_cache(parent, root/"invalid",2,1,913)
            self.assertFalse((root/"invalid/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
