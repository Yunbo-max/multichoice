"""End-to-end OFFLINE RoMQA Closed converter/scorer contracts; NO real Test/GPU."""
from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
import unittest

from harness.multiselect_metrics import read_dataset
from harness.set_distillation import input_digest, structured_prediction
from paper_static.romqa_data import (_as_record,_dpr_first,fit_prompt,convert,
                                     _cluster_sample,prepare)
from paper_static.romqa_metrics import score, verify_source
from paper_static.romqa_compare import paired


class ToyTokenizer:
    name_or_path="fixture-Qwen-simulator"
    chat_template="toy-chat-template"
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,enable_thinking=False):
        return "<user>"+messages[0]["content"]+"</user><assistant>"
    def encode(self,text,add_special_tokens=False):
        return [ord(c)+1 for c in text]


LABELS=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def raw(ident,cluster,question,labels,complete=None,missing_dpr=False):
    names=["Alice","Bob","Carol","David"]
    candidates=[{
        "text":name,"is_answer":bool(y),
        "dpr":[] if missing_dpr else [
            f"Public biographical passage about {name}: {'x'*500}"],
        "dpr_score":[] if missing_dpr else [.92],
    } for name,y in zip(names,labels)]
    if complete is None:complete=[names[i] for i,b in enumerate(labels) if b]
    return {"id":ident,"cluster_id":cluster,"question":question,
            "constraints":[{"prop_dir":"subj"}],
            "candidates":candidates,
            "complete_answer":[{"text":s} for s in complete]}


def fixture():
    return {
        "train":[raw("a","c1","Who were born in Paris?",[1,0,1,0]),
                 raw("b","c1","Who were pianists born in Paris?",[1,0,0,0]),
                 raw("c","c2","Which are notable figures?",[0,0,0,0])],
        "dev":[raw("d","c3","Who studied piano?",[1,0,1,0],
                   complete=["Alice","Carol","External"]),
               raw("e","c3","Who studied violin?",[0,0,0,0],complete=[]),
               raw("f","c4","Who was an artist?",[0,1,0,0])]
    }


def predict_bce(rows):
    out=[]
    for r in rows:
        logits=[1 if y else -1 for y in r["target"]]
        out.append({
            "id":r["id"],
            "option_keys":[o["key"] for o in r["options"]],
            "input_digest":input_digest(r),
            "logits":logits,
            "p":[1/(1+math.exp(-v)) for v in logits],
            "checkpoint_id":"fixture-bce",
        })
    return out


class Converter(unittest.TestCase):
    def test_same_visible_prompt_when_labels_change(self):
        r=raw("x","c","What is the evidence?",[1,0,0,1])
        new=raw("x","c","What is the evidence?",[0,1,1,0])
        row,e,_=_as_record(r,"train",expected_k=4)
        other,ee,_=_as_record(new,"train",expected_k=4)
        self.assertEqual(e,ee)
        self.assertEqual([o["key"] for o in row["options"]],
                         [o["key"] for o in other["options"]])
        tok=ToyTokenizer()
        rendered,_=fit_prompt(row,e,tok,LABELS,max_input_tokens=4000,evidence_chars=60)
        other_rendered,_=fit_prompt(other,ee,tok,LABELS,max_input_tokens=4000,evidence_chars=60)
        self.assertEqual(rendered["options"],other_rendered["options"])
        self.assertEqual(rendered["state"],other_rendered["state"])
        self.assertNotEqual(row["target"],other["target"])

    def test_gold_never_fills_candidate_set(self):
        row,e,sidecar=_as_record(
            raw("x","cx","Which entities?",[1,0,0,0],
                complete=["Alice","External"]), "train",expected_k=4)
        self.assertEqual(len(row["options"]),4)
        self.assertEqual(sum(row["target"]),1)
        self.assertEqual(sidecar["complete_answer"],["Alice","External"])
        self.assertNotIn("External",[c["key"] for c in row["options"]])

    def test_evidence_budget_keeps_every_entity(self):
        tok=ToyTokenizer()
        row,e,_=_as_record(raw("x","c1","Which?",[1,0,1,0]),"train",expected_k=4)
        baseline,b=fit_prompt(row,e,tok,LABELS,max_input_tokens=10000,evidence_chars=0)
        self.assertEqual(b["evidence_chars_used"],0)
        desired=b["prompt_tokens"]+120
        row2,meta=fit_prompt(row,e,tok,LABELS,max_input_tokens=desired,evidence_chars=500)
        self.assertEqual(len(row2["options"]),4)
        self.assertLessEqual(meta["prompt_tokens"],desired)
        self.assertLess(meta["evidence_chars_used"],500)
        self.assertEqual([o["key"] for o in row2["options"]],
                         [o["key"] for o in baseline["options"]])
        with self.assertRaisesRegex(ValueError,"exceed budget"):
            fit_prompt(row,e,tok,LABELS,max_input_tokens=512,evidence_chars=0)

    def test_dpr_is_deterministic_not_gold_based(self):
        c={"text":"x","is_answer":True,"dpr":["a","b"],"dpr_score":[.4,.9]}
        self.assertEqual(_dpr_first(c,"x"),"b")
        c["is_answer"]=False
        self.assertEqual(_dpr_first(c,"x"),"b")
        c["dpr_score"]=[.2,.3]
        self.assertEqual(_dpr_first(c,"x"),"a")
        c["dpr_score"]=[.2]
        with self.assertRaisesRegex(ValueError,"length mismatch"):
            _dpr_first(c,"x")

    def test_cross_split_leakage_guard_and_expected_k(self):
        data=fixture()
        data["dev"][0]["cluster_id"]="c1"
        with self.assertRaisesRegex(ValueError,"leak"):
            convert(data,ToyTokenizer(),LABELS,expected_k=4)
        data=fixture()
        with self.assertRaisesRegex(ValueError,"K=4 differs"):
            convert(data,ToyTokenizer(),LABELS,expected_k=100)

    def test_cluster_preserved_and_complete_answer_heldout(self):
        tok=ToyTokenizer()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"real_like"
            m=prepare(root,fixture(),tok,LABELS,expected_k=4,
                      evidence_chars=60,max_input_tokens=5000,
                      pilot_train_clusters=1,pilot_dev_clusters=1)
            self.assertFalse(m["test_opened"])
            self.assertFalse(m["candidate_injection"])
            self.assertFalse((root/"test.jsonl").exists())
            rows=read_dataset(root/"dev.jsonl",expected_split="dev")
            sidecar=[json.loads(x) for x in
                     (root/"evaluation_gold_only"/"dev.jsonl").read_text().splitlines()]
            verify_source(rows,sidecar)
            pilot=read_dataset(root/"pilot"/"dev.jsonl",expected_split="dev")
            self.assertEqual(len({r["group_id"] for r in pilot}),1)
            self.assertEqual(m["pilot"]["dev"]["n"],len(pilot))
            eval_pilot=[json.loads(x) for x in
                        (root/"pilot"/"dev_complete_answers.jsonl").read_text().splitlines()]
            verify_source(pilot,eval_pilot)
            self.assertTrue(all("complete_answer" not in r for r in rows))
            self.assertTrue(all("candidate_gold" not in r for r in rows))


class Scorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows,gold,stats=convert(fixture(),ToyTokenizer(),LABELS,
                                expected_k=4,max_input_tokens=5000,evidence_chars=48)
        cls.rows,cls.full=rows["dev"],gold["dev"]

    def test_bce_proper_joint_and_cluster_metrics(self):
        predictions=predict_bce(self.rows)
        report,per=score(self.rows,predictions,full_answers=self.full,bootstrap=15)
        self.assertEqual(report["n"],3)
        self.assertEqual(report["n_clusters"],2)
        self.assertEqual(report["n_official_complete_answer_nonempty"],2)
        self.assertAlmostEqual(report["metrics_all_questions"]["exact_match"],1.)
        self.assertLess(report["metrics_official_closed_candidate"]["complete_set_f1"],1.)
        self.assertEqual(report["test_read"],False)
        self.assertEqual(len(per),3)
        self.assertTrue(all(p["count_nll"]>=0 and p["joint_set_nll"]>=0 for p in per))
        self.assertIsNotNone(report["bootstrap_cluster_95ci_all_questions"])

    def test_zero_anchor_exactly_matches_independent_bce_probabilities(self):
        bce=predict_bce(self.rows)
        predictions=[]
        for row,item in zip(self.rows,bce):
            pred=structured_prediction(item["logits"],[0.]*(len(item["logits"])+1),
                                       row,count_anchor=True)
            pred.update({"id":row["id"],"option_keys":item["option_keys"],
                         "input_digest":item["input_digest"],
                         "candidate_logits":item["logits"]})
            predictions.append(pred)
        a,ap=score(self.rows,predictions,full_answers=self.full,bootstrap=0)
        b,bp=score(self.rows,bce,full_answers=self.full,bootstrap=0)
        for j in range(len(ap)):
            self.assertAlmostEqual(ap[j]["count_nll"],bp[j]["count_nll"],places=8)
            self.assertAlmostEqual(ap[j]["joint_set_nll"],bp[j]["joint_set_nll"],places=8)

    def test_no_silent_prediction_drop_or_sidecar_swap(self):
        predictions=predict_bce(self.rows)
        with self.assertRaisesRegex(ValueError,"prediction IDs mismatch"):
            score(self.rows,predictions[:-1],full_answers=self.full,bootstrap=0)
        bad=[dict(x) for x in self.full]
        bad[0]["candidate_gold"]=["Unknown"]
        with self.assertRaisesRegex(ValueError,"candidate gold"):
            score(self.rows,predictions,full_answers=bad,bootstrap=0)

    def test_paired_cluster_bootstrap_requires_same_data_and_ids(self):
        prediction=predict_bce(self.rows)
        report,per=score(self.rows,prediction,full_answers=self.full,bootstrap=0)
        report.update(source_data_sha256="a"*64,sidecar_sha256="b"*64)
        as_rows={x["id"]:x for x in per}
        results={key:(dict(report),dict(as_rows)) for key in ("bce","anchor","prior")}
        combined=paired(results,bootstrap=20)
        for comparison in combined["comparisons"].values():
            self.assertAlmostEqual(comparison["count_nll"]["improvement"],0)
            self.assertEqual(comparison["count_nll"]["ci95"],[0.0,0.0])
        broken={key:(dict(v[0]),dict(v[1])) for key,v in results.items()}
        broken["prior"][0]["source_data_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"incompatible"):
            paired(broken)


if __name__=="__main__":
    unittest.main()
