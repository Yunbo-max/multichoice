"""Dependency-light EUR-LEX converter / no-leakage / K-aware count baseline tests."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_static import eurlex_data as d
from qz import make_eurlex_payloads as qz


class FakeTokenizer:
    """Stateful reversible word tokenizer, supporting the Qwen template interface."""
    def __init__(self):
        self.to_id = {}
        self.from_id = {}

    def encode(self, text, add_special_tokens=False):
        parts = re.findall(r"\S+", text)
        result = []
        for part in parts:
            if part not in self.to_id:
                idx = len(self.to_id) + 1
                self.to_id[part] = idx
                self.from_id[idx] = part
            result.append(self.to_id[part])
        return result

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(self.from_id[x] for x in tokens)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        assert not tokenize
        return "<user> " + messages[0]["content"] + " </user> <assistant>"


def fake_options():
    return d.label_catalog([str(100163+i) for i in range(100)],
                           {str(100163+i):f"legal topic {i}" for i in range(100)})


class DataConversion(unittest.TestCase):
    def test_catalog_exact_and_descriptions(self):
        c = fake_options()
        self.assertEqual(len(c),100)
        self.assertEqual(c[0]["key"],"100163")
        with self.assertRaisesRegex(ValueError,"missing verified"):
            d.label_catalog([str(100163+i) for i in range(100)],{"100163":"only one"})
        with self.assertRaisesRegex(ValueError,"100 unique"):
            d.label_catalog(["100163"]*100,{"100163":"topic"})

    def test_qwen_single_token_slots_and_only_document_truncation(self):
        tok=FakeTokenizer()
        labels, ids=d.readout_labels(tok)
        self.assertEqual(len(labels),255)
        self.assertEqual(len(set(ids)),255)
        doc=("Substantive legal article on trade measures. " * 850)
        records=[{"text":doc, "labels":[3,7,12]}]
        converted, audit = d.convert_split(records,"train",fake_options(),
                                             tokenizer=tok,labels=labels,
                                             max_prompt_tokens=1200)
        row=converted[0]
        self.assertEqual(len(row["options"]),100)
        self.assertEqual(row["n_options"],100)
        self.assertEqual(sum(row["target"]),3)
        self.assertEqual([i for i,y in enumerate(row["target"]) if y],[3,7,12])
        self.assertEqual(d.prompt_length(tok,row,labels)<=1200,True)
        self.assertIn(d.TRUNCATION_MARKER.strip(),row["state"])
        self.assertTrue(audit["truncation"]["n_truncated"]==1)
        self.assertEqual(d.stable_text_key(doc),row["group_id"])
        self.assertIn("legal topic 99",d.render_prompt(tok,row,labels))
        self.assertIn(d.ANSWER_CUE,d.render_prompt(tok,row,labels))
        self.assertNotIn("target",d.render_prompt(tok,row,labels))

    def test_reject_bad_labels_and_option_catalog(self):
        options=fake_options()
        with self.assertRaisesRegex(ValueError,"invalid ClassLabel"):
            d.convert_split([{"text":"x","labels":[1,100]}],"train",options)
        with self.assertRaisesRegex(ValueError,"invalid ClassLabel"):
            d.convert_split([{"text":"x","labels":[1,1]}],"train",options)

    def test_save_split_without_opening_test_and_detect_leak(self):
        rows={"train":[{"text":"training legal document A","labels":[1,4]},
                       {"text":"training legal document B","labels":[2]}],
              "validation":[{"text":"validation document A","labels":[3,7]}]}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"converted"
            m=d.save_package(output,rows,fake_options(),provenance={"fixture":True},
                             pilot_train=1,pilot_dev=1)
            self.assertFalse(m["test_loaded"])
            self.assertFalse((output/"test.jsonl").exists())
            self.assertEqual(m["pilot"]["train"],1)
            self.assertEqual(len(m["pilot_files"]["train"]["sha256"]),64)
            train=[json.loads(s) for s in (output/"train.jsonl").read_text().splitlines()]
            self.assertEqual(len(train),2)
            self.assertTrue(all(len(z["options"])==100 for z in train))
            with self.assertRaisesRegex(ValueError,"output exists"):
                d.save_package(output,rows,fake_options(),provenance={"fixture":True},
                               pilot_train=1,pilot_dev=1)
            rows["validation"][0]["text"]="training legal document A"
            with self.assertRaisesRegex(ValueError,"cross-split"):
                d.save_package(Path(tmp)/"leaked",rows,fake_options(),
                               provenance={"fixture":True},pilot_train=1,pilot_dev=1)


class QzPlanner(unittest.TestCase):
    def test_six_safe_jobs_and_full_protocol(self):
        manifest={
            "dataset":"coastalcph/lex_glue",
            "test_loaded":False,
            "label_ids":[str(100163+i) for i in range(100)],
            "splits":{"train":{"n":55000,"sha256":"a"*64},
                      "validation":{"n":5000,"sha256":"b"*64}},
            "pilot":{"train":5000,"dev":1000},
            "pilot_files":{"train":{"sha256":"c"*64}, "dev":{"sha256":"d"*64}},
            "max_input_tokens":8192,
        }
        jobs=qz.job_specs({"workspace_id":"mock","framework":"pytorch"},
                          manifest,repo_root="/repo",
                          data_root="/repo/data/eurlex_decision_v1",
                          output_root="/repo/runs/eurlex_pilot",
                          model_root="/models/qwen4b",seeds=(43,),
                          warm_steps=120,continuation_steps=80)
        self.assertEqual(len(jobs),6)
        self.assertIn("warm",jobs[0]["name"])
        self.assertTrue(all("test.jsonl" not in p["command"] for p in jobs))
        self.assertTrue(all("rm -rf" not in p["command"] for p in jobs))
        self.assertTrue(all("STAGED_NOT_SUBMITTED" in p["description"] for p in jobs))
        joint=next(j for j in jobs if "goldcount" in j["name"])
        self.assertIn("--gold-count-weight 100",joint["command"])
        self.assertIn("--count-target gold",joint["command"])
        self.assertTrue(all("--max-input-tokens 8192" in j["command"] for j in jobs))
        jobs3=qz.job_specs({},manifest,repo_root="/repo",data_root="/d",
                           output_root="/runs",model_root="/m",seeds=(43,44,45),
                           warm_steps=120,continuation_steps=80)
        self.assertEqual(len(jobs3),16)
        with self.assertRaisesRegex(ValueError,"TEST"):
            qz.job_specs({},dict(manifest,test_loaded=True),repo_root="/repo",
                         data_root="/d",output_root="/runs",model_root="/m")

    def test_count_budget_preflight(self):
        manifest={"dataset":"coastalcph/lex_glue",
                  "test_loaded":False, "label_ids":[str(i) for i in range(100)],
                  "splits":{"train":{"n":55000,"sha256":"a"*64},
                            "validation":{"n":5000,"sha256":"b"*64}},
                  "pilot":{"train":4,"dev":1},
                  "pilot_files":{"train":{"sha256":"c"*64},"dev":{"sha256":"d"*64}},
                  "max_input_tokens":8192}
        with self.assertRaisesRegex(ValueError,"optimizer"):
            qz.job_specs({},manifest,repo_root="/repo",data_root="/d",
                         output_root="/runs",model_root="/m",seeds=(43,))


class CountBaselineLoss(unittest.TestCase):
    def test_gold_count_scaling_matches_analytic(self):
        try:
            import torch
            from harness.set_distillation import set_loss_terms
        except ImportError:
            self.skipTest("torch unavailable on CPU fixture runner")
        scores=torch.tensor([[.5,-.2,.4,-.3]],requires_grad=True)
        count=torch.zeros((1,5),requires_grad=True)
        row={"options":[{"key":str(i),"desc":str(i)} for i in range(4)],
             "train_target":[1.,0.,1.,0.],
             "train_count_probs":[0.,0.,1.,0.,0.],
             "target":[1,0,1,0],"selection_mode":"multi","allow_empty":True}
        a=set_loss_terms(scores,count,[row],count_target="gold",
                         gold_count_weight=1.0)[0][0]
        b=set_loss_terms(scores,count,[row],count_target="gold",
                         gold_count_weight=4.0)[0][0]
        self.assertAlmostEqual((b-a).item(),3*__import__("math").log(5)/4,places=5)
        b.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(count.grad).all())


if __name__=="__main__":
    unittest.main()
