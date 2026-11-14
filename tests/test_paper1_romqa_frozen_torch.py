"""CPU Torch INTEGRATION fixture only: NO Qwen, no RoMQA dataset, no GPU run."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch=None

from harness.set_distillation import input_digest
from paper_static.romqa_frozen import (
    SCHEMA, train, predict, check_cache, check_pair, digest
)
from paper_static.eurlex_frozen_count import centered_count_logz
from paper_static.romqa_metrics import score


def rows(split, n):
    out=[]
    for i in range(n):
        options=[{"key":s,"desc":f"Entity {s} evidence on {i}"}
                 for s in ("Alice","Bob","Carol","David")]
        target=([1,1,0,0] if i%3==0 else [1,0,0,0] if i%3==1 else [0,0,0,0])
        out.append({
            "id":f"{split}_{i}",
            "group_id":f"{split}_cluster_{i//2}",
            "source":"romqa_closed","lang":"en","split":split,
            "state":"Read the given evidence","question":f"Which satisfy task {split} {i}?",
            "selection_mode":"multi","allow_empty":True,
            "options":options,"n_options":4,"target":target
        })
    return out


def write_jsonl(path,values):
    with Path(path).open("x",encoding="utf-8") as file:
        for item in values:file.write(json.dumps(item)+"\n")


def make_cache(split,values,sha):
    xs=[]
    scores=[]
    logz=[]
    for i,row in enumerate(values):
        v=torch.linspace(-.2,.2,64)+(i+1)*.005
        xs.append(v)
        s=torch.tensor([.2+(i%2)*.1,-.1,.3,-.25])
        scores.append(s)
        logz.append(torch.tensor(centered_count_logz(s.tolist())))
    return {
        "schema":SCHEMA,"split":split,"k":4,"data_sha256":sha,
        "checkpoint_identity":"/frozen/common/romqa/bce",
        "model_config_sha256":"d"*64,
        "label_token_ids":[1,2,3,4],
        "hidden":torch.stack(xs),"scores":torch.stack(scores),
        "logz":torch.stack(logz),
        "ids":[r["id"] for r in values],
        "input_digests":[input_digest(r) for r in values],
        "label_counts":[sum(r["target"]) for r in values],
    }


@unittest.skipUnless(torch is not None,"PyTorch CPU test requires torch")
class FrozenCountWorkflow(unittest.TestCase):
    def test_train_anchor_and_prior_reload_proper_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            t,dv=rows("train",8),rows("dev",4)
            train_file=work/"train.jsonl"
            dev_file=work/"dev.jsonl"
            write_jsonl(train_file,t)
            write_jsonl(dev_file,dv)
            tcache=make_cache("train",t,digest(train_file))
            dcache=make_cache("dev",dv,digest(dev_file))
            train_pt=work/"train.pt"
            dev_pt=work/"dev.pt"
            torch.save(tcache,train_pt)
            torch.save(dcache,dev_pt)
            self.assertTrue(check_cache(tcache,t,"train",digest(train_file)))
            self.assertTrue(check_pair(tcache,dcache,t,dv,digest(train_file),digest(dev_file)))
            sidecar=[{
                "id":r["id"],"cluster_id":r["group_id"],
                "candidate_keys":[o["key"] for o in r["options"]],
                "candidate_gold":[o["key"] for o,y in zip(r["options"],r["target"]) if y],
                "complete_answer":[o["key"] for o,y in zip(r["options"],r["target"]) if y],
            } for r in dv]
            for mode,init in (("anchor","zero"),("independent","prior")):
                head_dir=work/f"{mode}_{init}"
                train(SimpleNamespace(
                    train_gold=train_file,dev_gold=dev_file,
                    train_cache=train_pt,dev_cache=dev_pt,
                    mode=mode,init=init,output=head_dir,
                    steps=6,batch=4,seed=43,eval_every=2,lr=.001,device="cpu"
                ))
                self.assertTrue((head_dir/"head.pt").exists())
                meta=json.loads((head_dir/"training_summary.json").read_text())
                self.assertTrue(meta["frozen_backbone_and_candidate_scores"])
                self.assertFalse(meta["test_opened"])
                self.assertEqual(meta["mode"],mode)
                dest=work/f"pred_{mode}_{init}"
                predict(SimpleNamespace(
                    gold=dev_file,cache=dev_pt,mode=mode,head=head_dir,
                    output=dest
                ))
                records=[json.loads(s) for s in
                         (dest/"predictions.jsonl").read_text().splitlines()]
                report,per=score(dv,records,full_answers=sidecar,bootstrap=0)
                self.assertEqual(report["n"],len(dv))
                self.assertGreaterEqual(report["metrics_all_questions"]["count_nll"],0.)
                self.assertTrue(all(p["joint_set_nll"]>=0 for p in per))
            bce_dest=work/"pred_bce"
            predict(SimpleNamespace(gold=dev_file,cache=dev_pt,
                                    mode="bce",head=None,output=bce_dest))
            bce=[json.loads(s) for s in
                 (bce_dest/"predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(bce),len(dv))
            self.assertTrue(all(len(p["logits"])==4 for p in bce))

    def test_reject_different_data_or_model_cache(self):
        t,dv=rows("train",3),rows("dev",3)
        tc=make_cache("train",t,"a"*64)
        dc=make_cache("dev",dv,"b"*64)
        dc["checkpoint_identity"]="OTHER-BCE-CHECKPOINT"
        with self.assertRaisesRegex(ValueError,"unpaired"):
            check_pair(tc,dc,t,dv,"a"*64,"b"*64)
        dc["checkpoint_identity"]=tc["checkpoint_identity"]
        dc["scores"][0][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            check_pair(tc,dc,t,dv,"a"*64,"b"*64)


if __name__=="__main__":
    unittest.main()
