"""Stage-only RoMQA command matrix, content hashes and non-execution checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paper_static.romqa_data import prepare
from paper_static.romqa_plan import stage


class Tokenizer:
    name_or_path="offline-mock"
    chat_template="mock"
    def encode(self,text,add_special_tokens=False):return [ord(x)+1 for x in text]
    def apply_chat_template(self,messages,**kwargs):
        return "<user>"+messages[0]["content"]+"</user><assistant>"


def sample(ident,cluster):
    return dict(
        id=ident,cluster_id=cluster,question=f"Which answer to {ident}?",
        constraints=[],complete_answer=[{"text":"Alice"}],
        candidates=[
            dict(text=t,is_answer=i==0,dpr=[f"Evidence about {t}"],
                 dpr_score=[.9]) for i,t in enumerate(("Alice","Bob","Carol","David"))
        ]
    )


class StageOnly(unittest.TestCase):
    def test_generates_ordered_matrix_without_running_any_commands(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            dat=p/"data"
            manifest=prepare(
                dat,{"train":[sample("a","train1"),sample("b","train2")],
                     "dev":[sample("c","dev1"),sample("d","dev2")]},
                Tokenizer(),list("ABCD"),expected_k=4,
                evidence_chars=32,max_input_tokens=2048,
                pilot_train_clusters=1,pilot_dev_clusters=1,
                provenance={"kind":"fixture","sha256":"z"*64},
            )
            repo=p/"repo"
            (repo/"harness").mkdir(parents=True)
            (repo/"harness"/"train_multiselect.py").write_text("# mock only\n")
            model=p/"base_model"
            model.mkdir()
            modules=p/"modules.json"
            modules.write_text('{"mock":true}')
            root=p/"future_runs"
            output=p/"staged_commands"
            plan=stage(
                manifest=dat/"manifest.json",data=dat,repo=repo,
                base_model=model,modules=modules,
                run_root=root,output=output,pilot=True,
                seeds=(43,44,45),lrs=(.001,.0003),
                head_steps=5,head_batch=4,batch=2,
            )
            self.assertEqual(plan["status"],"ROMQA_PLAN_ONLY_NOT_SUBMITTED_NOT_TRAINED")
            self.assertEqual(plan["scripts"]["01_warm_bce.sh"],1)
            self.assertEqual(plan["scripts"]["02_matched_lora_continuations.sh"],12)
            self.assertEqual(plan["scripts"]["03_extract_same_bce_train_dev.sh"],2)
            self.assertEqual(plan["scripts"]["04_train_matched_frozen_heads.sh"],18)
            self.assertEqual(plan["scripts"]["05_evaluate_frozen_dev_only.sh"],38)
            self.assertFalse(root.exists())
            self.assertTrue((output/"plan.json").is_file())
            self.assertTrue(plan["test_opened"] is False)
            scripts="".join((output/name).read_text() for name in plan["scripts"])
            self.assertIn("romqa_frozen",scripts)
            self.assertIn("--count-anchor",scripts)
            self.assertIn("--count-anchor-detach",scripts)
            self.assertIn("--init prior",scripts)
            self.assertNotIn("test.jsonl",scripts)
            self.assertNotIn("rm -rf",scripts)
            with self.assertRaisesRegex(ValueError,"already"):
                stage(manifest=dat/"manifest.json",data=dat,repo=repo,
                      base_model=model,modules=modules,
                      run_root=root,output=output,pilot=True)

    def test_refuses_tampered_input_sha(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            data=p/"data"
            prepare(data,{"train":[sample("a","train")],
                          "dev":[sample("b","dev")]},
                    Tokenizer(),list("ABCD"),expected_k=4,
                    evidence_chars=0,max_input_tokens=2048,
                    provenance={"kind":"fixture"},
                    pilot_train_clusters=0,pilot_dev_clusters=0)
            with (data/"dev.jsonl").open("a") as f:
                f.write('{"id":"injected"}\n')
            repo=p/"repo"
            (repo/"harness").mkdir(parents=True)
            (repo/"harness"/"train_multiselect.py").touch()
            model=p/"model"
            model.mkdir()
            mods=p/"modules.json"
            mods.touch()
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                stage(manifest=data/"manifest.json",data=data,repo=repo,
                      base_model=model,modules=mods,
                      run_root=p/"future",output=p/"output")


if __name__=="__main__":
    unittest.main()
