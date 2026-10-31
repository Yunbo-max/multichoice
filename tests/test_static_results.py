"""Artificial-run auditor regression tests, no checkpoint or GPU needed."""
import json
import tempfile
import unittest
from pathlib import Path
from paper_static.collect_results import summary

class CollectTests(unittest.TestCase):
    def test_completed_and_pending_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/"one";run.mkdir()
            (run/"training_complete.json").write_text(json.dumps({
              "steps":79,"selected":{"step":79,"question_em":0.6,"checkpoint":"checkpoint-79"}}))
            (run/"training_config.json").write_text(json.dumps({
              "seed":43,"objective":"set","model":"BCE_S44",
              "count_anchor":True,"count_anchor_detach":False,
              "train_sha256":"a","dev_sha256":"b","modules_sha256":"c"}))
            (run/"validation_000079.json").write_text(json.dumps({
              "question_em":0.6,"f1":0.9,"count_nll":0.5}))
            spec={"id":"run1","arm":"set_ce_anchor","seed":43,"model_tag":"4b",
                  "output":str(run),"model":"BCE_S44","objective":"set",
                  "count_anchor":True,"count_anchor_detach":False,"update_budget":79,
                  "train_sha256":"a","dev_sha256":"b","module_sha256":"c"}
            pending=dict(spec,id="run2",output=str(root/"missing"),arm="bce_continue")
            report=summary([spec,pending])
            self.assertEqual(report["completed"],1)
            self.assertEqual(report["status"],"INCOMPLETE")
            with self.assertRaisesRegex(ValueError,"missing"):
                summary([spec,pending],require_all=True)
            wrong=dict(spec,train_sha256="WRONG")
            with self.assertRaisesRegex(ValueError,"SHA"):
                summary([wrong])

if __name__=="__main__":unittest.main()
