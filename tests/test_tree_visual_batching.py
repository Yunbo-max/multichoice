"""CPU torch regression tests for mixed-size image prompts and cached rows."""
import json
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None,"torch unavailable")
class VisualBatchingTests(unittest.TestCase):
    def test_padded_image_batch_answer_cue_position(self):
        from treejev.visual_batching import append_answer_cue,last_valid_positions
        features={"input_ids":torch.tensor([[11,12,0,0,0],[21,22,23,24,25]]),
                  "attention_mask":torch.tensor([[1,1,0,0,0],[1,1,1,1,1]]),
                  "mm_token_type_ids":torch.tensor([[1,0,0,0,0],[1,1,0,0,0]]),
                  "image_grid_thw":torch.tensor([[1,1,1],[1,1,1]])}
        fixed=append_answer_cue(features,torch.tensor([[40,41]]),pad_token_id=99)
        self.assertEqual(fixed["input_ids"].tolist(),
                         [[11,12,40,41,99,99,99],[21,22,23,24,25,40,41]])
        self.assertEqual(fixed["attention_mask"].tolist(),
                         [[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]])
        self.assertEqual(last_valid_positions(fixed["attention_mask"]).tolist(),[3,6])
        self.assertTrue(torch.equal(features["image_grid_thw"],fixed["image_grid_thw"]))
        self.assertEqual(fixed["mm_token_type_ids"][0].tolist(),[1,0,0,0,0,0,0])

    def test_left_padding_and_bad_padding_guard(self):
        from treejev.visual_batching import append_answer_cue,last_valid_positions
        features={"input_ids":torch.tensor([[0,0,5,6],[9,8,7,6]]),
                  "attention_mask":torch.tensor([[0,0,1,1],[1,1,1,1]])}
        with self.assertRaisesRegex(ValueError,"right-packed"):
            last_valid_positions(features["attention_mask"])
        fixed=append_answer_cue(features,[50])
        self.assertEqual(fixed["input_ids"].tolist(),[[5,6,50,0,0],[9,8,7,6,50]])
        self.assertEqual(last_valid_positions(fixed["attention_mask"]).tolist(),[2,4])
        with self.assertRaisesRegex(ValueError,"right-packed"):
            last_valid_positions(torch.tensor([[1,0,1]]))

    def test_cached_rows_random_order_and_bounded_lru(self):
        from treejev.train_visual_action import FeatureCache
        from treejev.visual_batching import last_valid_positions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=[]
            for ci in range(4):
                name=f"chunk_{ci:04d}.pt"
                torch.save({
                    "ids":[f"row-{ci}"],
                    "pixel_values":[torch.full((1,1536),float(ci))],
                    "image_grid_thw":torch.tensor([[1,1,1]]),
                    "input_ids":torch.tensor([[20+ci,41,0,0]]),
                    "attention_mask":torch.tensor([[1,1,0,0]]),
                    "mm_token_type_ids":torch.tensor([[1,0,0,0]])},
                    root/name)
                manifest.append({"chunk":name,"n":1,"start":ci})
            (root/"manifest.json").write_text(json.dumps(manifest))
            cache=FeatureCache(root,max_loaded_chunks=2)
            order=[3,0,2,1]
            got=cache.get(order,expected_ids=[f"row-{i}" for i in order])
            self.assertEqual(got["input_ids"].tolist(),[[23,41],[20,41],[22,41],[21,41]])
            self.assertEqual(last_valid_positions(got["attention_mask"]).tolist(),[1]*4)
            self.assertEqual([float(v[0,0]) for v in got["pixel_values"].split(1)],list(map(float,order)))
            self.assertLessEqual(len(cache.loaded),2)
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                cache.get([1],expected_ids=["bad"])

if __name__=="__main__":
    unittest.main()
