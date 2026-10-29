"""Stage0 command and dataset safety smoke without GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from treejev.prepare import main as prepare
from treejev.data import read_jsonl, save_jsonl
from treejev.sokoban import DEMO_MAP

class PrepareTests(unittest.TestCase):
    def test_demo_exports_two_shortest_trajectories(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/"demo"
            prepare(["oracle","--demo","--output-dir",str(out)])
            train=list(read_jsonl(out/"train_multi.jsonl"))
            routes=list(read_jsonl(out/"trajectories_train.jsonl"))
            dev=list(read_jsonl(out/"dev_multi.jsonl"))
            self.assertTrue(train)
            self.assertEqual(len(routes),2)
            self.assertEqual(dev,[])
            self.assertTrue(all(r["split"]=="train" for r in train))
    def test_two_maps_have_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)
            source=home/"levels.jsonl"
            # Different board geometries are needed for independent groups.
            second=list(DEMO_MAP)
            second[3]="#    ##"
            save_jsonl(source,[{"level_id":"a","ascii":DEMO_MAP},
                               {"level_id":"b","ascii":second}])
            out=home/"split"
            prepare(["oracle","--maps",str(source),"--output-dir",str(out),
                     "--dev-fraction","0.5"])
            tr=list(read_jsonl(out/"train_multi.jsonl"))
            dv=list(read_jsonl(out/"dev_multi.jsonl"))
            self.assertTrue(tr and dv)
            self.assertFalse({r["group_id"] for r in tr}&{r["group_id"] for r in dv})

if __name__=="__main__":
    unittest.main()
