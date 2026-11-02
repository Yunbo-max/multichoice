"""Offline P0 matched decoding and P1 frozen-BCE head control contracts."""
from __future__ import annotations
from argparse import Namespace
import itertools
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from harness.set_distillation import structured_prediction, input_digest
from paper_static.eurlex_p0_diagnostics import analyze
from paper_static.eurlex_frozen_count import (
    centered_count_logz, count_log_probabilities, verify_cache, validate_pair
)


def row(idx, split="dev", target_indices=(1,2)):
    options=[{"key":str(100200+i),"desc":f"eurovoc {i}"} for i in range(100)]
    return {"id":f"eurlex:{split}:{idx:06d}", "source":"lexglue_eurlex",
            "split":split, "lang":"en","group_id":f"{split}-passage-{idx}",
            "state":f"Legal text document {split}-{idx}","question":"Select concepts",
            "options":options,"n_options":100,
            "selection_mode":"multi","allow_empty":True,
            "target":[int(i in target_indices) for i in range(100)]}


def output(row,scores, *, anchor=False, independent=False):
    if anchor or independent:
        count=[-1.0]*101
        count[2]=1.0
        pred=structured_prediction(scores,count,row,count_anchor=anchor)
        pred["candidate_logits"]=scores
        pred["count_residual_logits"]=count
    else:
        pred={"p":[1/(1+math.exp(-s)) for s in scores],"logits":scores}
    pred.update(id=row["id"],option_keys=[z["key"] for z in row["options"]],
                input_digest=input_digest(row),
                checkpoint_id=("fixture|anchor" if anchor else
                               "fixture|independent" if independent else
                               "fixture|bce_continue"))
    return pred


class P0(unittest.TestCase):
    def test_matched_DEV_all_decoders_f1_and_nll(self):
        rows=[row(i,target_indices=(1,2) if i%2 else (3,4,5)) for i in range(8)]
        bce,anchor,independent=[],[],[]
        for r in rows:
            scores=[-3.]*100
            for i,y in enumerate(r["target"]):
                if y:
                    scores[i]=2.0
            bce.append(output(r,scores))
            anchor.append(output(r,scores,anchor=True))
            independent.append(output(r,scores,independent=True))
        result=analyze(rows,{"bce_continue":bce,"anchor":anchor,
                             "learned_count":independent},bootstrap=12)
        self.assertEqual(result["status"],"DEV_OFFLINE_COMPLETE_NO_GPU")
        self.assertEqual(len(result["arms"]),3)
        self.assertIn("anchor",result["paired_vs_bce_continue"])
        self.assertIn("paired_cluster_bootstrap_95_ci",
                      result["paired_vs_bce_continue"]["anchor"])
        self.assertGreaterEqual(
            result["arms"]["bce_continue"]["decoders"]["oracle_gold_count_topk_NON_DEPLOYABLE"]["micro_f1"],
            result["arms"]["bce_continue"]["decoders"]["set_map_or_fixed_threshold_05"]["micro_f1"])
        self.assertGreater(
            result["arms"]["anchor"]["proper_probability"]["proper_joint_set_nll"],0)
        self.assertAlmostEqual(
            result["arms"]["bce_continue"]["decoders"]["dev_tuned_marginal_threshold"]["micro_f1"],1.)
        self.assertEqual(
            result["arms"]["anchor"]["decoders"]["dev_tuned_marginal_threshold"]["threshold"],
            .05)
        with self.assertRaisesRegex(ValueError,"DEV only"):
            analyze([dict(rows[0],split="test")],{"bce_continue":[bce[0]]})
        with self.assertRaisesRegex(ValueError,"IDs mismatch"):
            analyze(rows,{"bce_continue":bce[:-1]})

    def test_reject_missing_model_provenance(self):
        r=row(1)
        s=[-2.]*100
        p=output(r,s)
        p.pop("checkpoint_id")
        with self.assertRaisesRegex(ValueError,"checkpoint identity"):
            analyze([r],{"bce_continue":[p]})


class FrozenPureMathematics(unittest.TestCase):
    def test_stable_elementary_symmetric_count_partition(self):
        scores=[-1.2,.1,.8,1.4]
        z=centered_count_logz(scores)
        for m in range(5):
            exact=math.log(sum(math.exp(sum(scores[i] for i in subset))
                               for subset in itertools.combinations(range(4),m)))
            self.assertAlmostEqual(z[m],exact,places=10)

    def test_anchor_reparameterizes_independent_count_distribution(self):
        s=[.3,-1.1,2.0,-.4]
        z=centered_count_logz(s)
        a=[-.2,1.4,-.1,.2,.8]
        independent=[x+y for x,y in zip(a,z)]
        pa=count_log_probabilities(a,z,"anchor")
        pc=count_log_probabilities(independent,z,"independent")
        for x,y in zip(pa,pc):
            self.assertAlmostEqual(x,y,places=12)
        self.assertGreater(math.exp(sum(pa)/len(pa)),0)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"PyTorch is required for frozen-feature training contracts")
class FrozenActualTorch(unittest.TestCase):
    def test_exact_feature_alignment_detects_row_and_checkpoint_corruption(self):
        rows=[row(i,"train") for i in range(3)]
        cache={
            "schema":"eurlex_frozen_bce_features_v1","split":"train",
            "hidden":torch.zeros(3,64),
            "scores":torch.zeros(3,100),
            "logz":torch.zeros(3,101),
            "ids":[r["id"] for r in rows],
            "input_digests":[input_digest(r) for r in rows],
            "label_counts":[2]*3,
            "checkpoint_identity":"/frozen/base",
            "label_token_ids":list(range(100)),
            "model_config_sha256":"a"*64,
            "data_sha256":"b"*64}
        self.assertTrue(verify_cache(cache,rows,split="train",data_sha="b"*64))
        bad=dict(cache,ids=[*cache["ids"][:2],"wrong"])
        with self.assertRaisesRegex(ValueError,"different"):
            verify_cache(bad,rows,split="train",data_sha="b"*64)

    def test_count_heads_share_frozen_logits_and_reduce_dev_count_nll(self):
        from paper_static.eurlex_frozen_count import train,predict,digest
        rng=random.Random(5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={}
            for split,n in (("train",48),("dev",18)):
                rows=[]
                feats=[]
                scores=[]
                logs=[]
                for i in range(n):
                    m=1 if i%2 else 3
                    r=row(i,split,target_indices=(1,) if m==1 else (1,2,3))
                    rows.append(r)
                    vec=[.0]*64
                    vec[0]=0.5 if m==1 else -0.5
                    vec[1]=rng.uniform(-.02,.02)
                    feats.append(vec)
                    ss=[-2.]*100
                    for j in range(m):
                        ss[j]=.5
                    scores.append(ss)
                    logs.append(centered_count_logz(ss))
                path=root/f"{split}.jsonl"
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
                cache_path=root/f"{split}.pt"
                payload={
                    "schema":"eurlex_frozen_bce_features_v1",
                    "split":split,"data_sha256":digest(path),
                    "checkpoint_identity":"/fake/frozen/qwen4b",
                    "model_config_sha256":"a"*64,
                    "label_token_ids":list(range(100)),
                    "ids":[r["id"] for r in rows],
                    "input_digests":[input_digest(r) for r in rows],
                    "label_counts":[sum(r["target"]) for r in rows],
                    "hidden":torch.tensor(feats,dtype=torch.float32),
                    "scores":torch.tensor(scores,dtype=torch.float32),
                    "logz":torch.tensor(logs,dtype=torch.float32),
                }
                torch.save(payload,cache_path)
                paths[split]=(path,cache_path)
            for mode in ("anchor","independent"):
                head_path=root/f"head_{mode}"
                args=Namespace(train_gold=paths["train"][0],
                    dev_gold=paths["dev"][0],
                    train_cache=paths["train"][1],
                    dev_cache=paths["dev"][1],
                    mode=mode,init="zero",seed=43,steps=35,batch=8,
                    eval_every=5,lr=.006,device="cpu",output=head_path)
                train(args)
                summary=json.loads((head_path/"training_summary.json").read_text())
                self.assertTrue(summary["frozen_backbone_and_candidate_scores"])
                self.assertTrue(math.isfinite(summary["best_dev_count_nll"]))
                self.assertLess(summary["best_dev_count_nll"],summary["initial_dev_count_nll"])
                predict(Namespace(gold=paths["dev"][0],cache=paths["dev"][1],
                        mode=mode,head=head_path,output=root/f"pred_{mode}"))
                preds=[json.loads(s) for s in
                    (root/f"pred_{mode}"/"predictions.jsonl").read_text().splitlines()]
                self.assertEqual(len(preds),18)
                self.assertEqual(preds[0]["candidate_logits"],
                                 [float(v) for v in
                                    torch.load(paths["dev"][1],weights_only=True)["scores"][0].tolist()])
                self.assertIn("count_probs",preds[0])


if __name__=="__main__":
    unittest.main()
