"""Proof/contract checks: exact small-K oracle, block-DP and disjoint splits."""
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static import pairwise_exact as dense
from paper_static import block_pair_dp as block
from paper_static import controlled_regimes as regimes
try:
    import torch
except ImportError:
    torch=None


class BlockInference(unittest.TestCase):
    def test_exact_partition_map_and_nll_small_k(self):
        s=[.6,-.8,.2,.3,-1.1,1.2]
        pairs=[(1,4),(2,5)]
        weights=[2.,-1.5]
        a=[-.1,.2,-.3,.1,.4,-.7,0.]
        J=[[0.]*6 for _ in range(6)]
        for (i,j),w in zip(pairs,weights):
            J[i][j]=J[j][i]=w
        truth=dense.exact_distribution(s,a,J,allowed=[1,2,3,4,5,6])
        got=block.exact_block_readout(s,pairs,weights,a,allowed=[1,2,3,4,5,6])
        self.assertAlmostEqual(sum(got["count_probs"]),1.0)
        self.assertAlmostEqual(math.exp(got["log_partition"]),
                               sum(math.exp(sum(s[i] for i in range(6) if mask>>i&1)
                                    +a[mask.bit_count()]+sum(weights[n] for n,(i,j)
                                    in enumerate(pairs) if mask>>i&1 and mask>>j&1))
                                   for mask in truth), places=9)
        for m in range(7):
            self.assertAlmostEqual(got["count_probs"][m],
                sum(p for mask,p in truth.items() if mask.bit_count()==m),places=10)
        best=max(truth.values())
        selected=sum(1<<i for i in got["selected"])
        self.assertAlmostEqual(truth[selected],best,places=10)
        for mask,prob in truth.items():
            gold=[int(bool(mask>>i&1)) for i in range(6)]
            self.assertAlmostEqual(block.exact_subset_nll(s,gold,pairs,weights,a,
                allowed=[1,2,3,4,5,6]), -math.log(prob), places=10)

    def test_reductions_bce_and_count_anchor(self):
        scores=[.2,-.4,1.3,.5]
        pairs=[(0,2),(1,3)]
        got=block.exact_block_readout(scores,pairs,[0.,0.])
        bce=dense.exact_distribution(scores)
        self.assertAlmostEqual(got["log_partition"],
                               sum(math.log1p(math.exp(s)) for s in scores),places=10)
        for m in range(5):
            self.assertAlmostEqual(got["count_probs"][m],
                sum(v for mask,v in bce.items() if mask.bit_count()==m),places=11)
        a=[0.,.7,-1.2,.1,.4]
        got2=block.exact_block_readout(scores,pairs,[0.,0.],a)
        expected=dense.exact_distribution(scores,a)
        self.assertAlmostEqual(got2["log_partition"],
            math.log(sum(math.exp(sum(scores[i] for i in range(4) if mask>>i&1)
                +a[mask.bit_count()]) for mask in expected)), places=10)

    def test_255_option_dp_and_invalid_matching(self):
        k=255
        scores=[.03*((i%11)-5) for i in range(k)]
        pairs=[(i,i+1) for i in range(0,k-1,2)]
        weights=[0.5]*len(pairs)
        result=block.exact_block_readout(scores,pairs,weights,allowed=range(1,k+1))
        self.assertEqual(len(result["count_probs"]),k+1)
        self.assertEqual(len(result["count_log_probs"]),k+1)
        self.assertAlmostEqual(sum(result["count_probs"]),1.0,places=10)
        self.assertNotEqual(result["map_count"],0)
        with self.assertRaises(ValueError):
            block.exact_block_readout([0.,0.,0.],[(0,1),(1,2)],[.1,.2])

    @unittest.skipIf(torch is None,"PyTorch not installed")
    def test_differentiable_logz_gradient(self):
        s=torch.tensor([.2,-.8,1.1,.4],dtype=torch.double,requires_grad=True)
        w=torch.tensor([.9,-.5],dtype=torch.double,requires_grad=True)
        a=torch.tensor([0.,.2,-.1,.4,0.],dtype=torch.double,requires_grad=True)
        z=block.torch_block_log_partition(s,[(0,2),(1,3)],w,a)
        expected=block.exact_block_readout(s.detach().tolist(),
               [(0,2),(1,3)],w.detach().tolist(),a.detach().tolist())
        self.assertAlmostEqual(z.item(),expected["log_partition"],places=11)
        z.backward()
        for grad in (s.grad,w.grad,a.grad):
            self.assertTrue(torch.isfinite(grad).all().item())
        eps=1e-5
        wa=w.detach().tolist()
        ap=a.detach().tolist()
        def get(w0):
            return block.exact_block_readout(s.detach().tolist(),
                     [(0,2),(1,3)],[w0,wa[1]],ap)["log_partition"]
        finite=(get(wa[0]+eps)-get(wa[0]-eps))/(2*eps)
        self.assertAlmostEqual(w.grad[0].item(),finite,places=7)


class RegimeGeneration(unittest.TestCase):
    def test_matching_moments_identical_but_edges_differ(self):
        k=8
        scores=[0.0]*k
        a=[-.9*(m-2)**2 for m in range(k+1)]
        graphs=[((0,1),(2,3),(4,5),(6,7)),
                ((0,2),(1,3),(4,6),(5,7))]
        stats=[]
        for graph in graphs:
            J=[[0.]*k for _ in range(k)]
            for i,j in graph:
                J[i][j]=J[j][i]=3.0
            stats.append(dense.moments(dense.exact_distribution(scores,a,J),k))
        for i in range(k):
            self.assertAlmostEqual(stats[0]["marginals"][i],stats[1]["marginals"][i])
        for i in range(k+1):
            self.assertAlmostEqual(stats[0]["count_probs"][i],stats[1]["count_probs"][i])
        self.assertGreater(abs(stats[0]["pairwise_marginals"][0][1]-
                               stats[1]["pairwise_marginals"][0][1]),0.01)

    def test_generator_split_and_no_teacher_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)/"controlled"
            manifest=regimes.generate(root,seed=17,ntrain=3,ndev=2,ntest=2,repeats=3,k=6)
            self.assertEqual(manifest["status"],"SYNTHETIC_ORACLE_DATA_ONLY_NOT_TRAINED_MODEL")
            seen=set()
            matching={s:set() for s in ("train","dev","test")}
            for split in matching:
                data=regimes.read_rows(root/"student"/f"{split}.jsonl") if hasattr(regimes,"read_rows") else [
                    json.loads(line) for line in (root/"student"/f"{split}.jsonl").read_text().splitlines()]
                oracle=[json.loads(line) for line in (
                    root/"oracle_DO_NOT_USE_AS_MODEL_INPUT"/f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(data),len(oracle)*3)
                self.assertTrue(all(row["split"]==split and row["allow_empty"] for row in data))
                self.assertTrue(all("teacher_scores" not in row for row in data))
                contexts={r["group_id"] for r in data}
                self.assertFalse(contexts & seen)
                seen|=contexts
                for row in oracle:
                    self.assertAlmostEqual(sum(row["true_set_probs"].values()),1.0,places=10)
                    if row["regime"]=="R":
                        matching[split].add(tuple(tuple(p) for p in row["teacher_pairs"]))
            self.assertFalse(matching["train"] & matching["dev"])
            self.assertFalse(matching["train"] & matching["test"])
            self.assertFalse(matching["dev"] & matching["test"])
            with self.assertRaisesRegex(ValueError,"already exists"):
                regimes.generate(root,seed=17,ntrain=3,ndev=2,ntest=2,repeats=3,k=6)


if __name__=="__main__":
    unittest.main()
