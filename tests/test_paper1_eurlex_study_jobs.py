"""Offline P1/P2 gates and real-QZ-profile staging safety tests."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from qz.make_eurlex_frozen_cache_payloads import payloads
from paper_static.eurlex_frozen_head_jobs import create_plan
from paper_static.eurlex_study_plan import stage, read_timing, pass_p2_gate


def manifest():
    return {
        "dataset":"coastalcph/lex_glue",
        "test_loaded":False,
        "label_ids":[str(i) for i in range(100)],
        "splits":{
            "train":{"n":55000,"sha256":"a"*64},
            "validation":{"n":5000,"sha256":"b"*64},
        },
        "pilot":{"train":5000,"dev":1000},
        "pilot_files":{"train":{"sha256":"c"*64},"dev":{"sha256":"d"*64}},
        "max_input_tokens":8192,
    }


def result(positive=True):
    count_anchor=0.5 if positive else 1.05
    set_anchor=1.2 if positive else 1.95
    return {
        "status":"DEV_OFFLINE_COMPLETE_NO_GPU",
        "gold_sha256":"a"*64,"n_questions":1000,
        "arms":{
            "frozen_anchor":{"proper_probability":{"count_nll":count_anchor,
                                                 "proper_joint_set_nll":set_anchor},
                             "decoders":{"dev_tuned_marginal_threshold":{"micro_f1":.502}}},
            "frozen_independent":{"proper_probability":{"count_nll":.8,
                                                      "proper_joint_set_nll":1.5},
                                  "decoders":{"dev_tuned_marginal_threshold":{"micro_f1":.49}}},
            "bce_continue":{"proper_probability":{"count_nll":1.,
                                                  "proper_joint_set_nll":1.8},
                            "decoders":{"dev_tuned_marginal_threshold":{"micro_f1":.5}}},
        }}


class Jobs(unittest.TestCase):
    def test_p2_gate_needs_three_matched_seed_reports(self):
        self.assertFalse(pass_p2_gate([result()] * 2)["passed"])
        self.assertTrue(pass_p2_gate([result()] * 3)["passed"])
        self.assertFalse(pass_p2_gate([result(False)] * 3)["passed"])
        mismatched=result()
        mismatched["gold_sha256"]="z"*64
        self.assertFalse(pass_p2_gate([result(),result(),mismatched])["passed"])

    def test_frozen_feature_qz_payloads_preserve_checkpoint_and_no_test(self):
        jobs=payloads({"max_running_time_ms":"10800000"},
                      manifest(),root="/repo",data="/repo/data/eurlex",
                      model="/repo/runs/shared_bce/final",
                      output="/repo/runs/frozen_cache",batch_size=1)
        self.assertEqual(len(jobs),2)
        for job in jobs:
            self.assertIn("eurlex_frozen_count extract",job["command"])
            self.assertNotIn("/test.jsonl",job["command"])
            self.assertNotIn("rm -rf",job["command"])
            self.assertIn("sha256",job["command"])
        with self.assertRaisesRegex(ValueError,"Pilot requires"):
            payloads({},dict(manifest(),test_loaded=True),root="/repo",data="/repo/data",
                     model="/model",output="/cache")

    def test_frozen_cpu_paired_head_job_grid(self):
        plan=create_plan(
            train_cache=Path("/repo/train.pt"),dev_cache=Path("/repo/dev.pt"),
            train_gold=Path("/repo/train.jsonl"),dev_gold=Path("/repo/dev.jsonl"),
            output_root=Path("/repo/runs/head"),seeds=(43,44,45),
            rates=(.0003,.001),steps=200,batch=32)
        self.assertEqual(plan["status"],"CPU_HEAD_PLAN_NOT_RUN")
        self.assertEqual(len(plan["jobs"]),18)
        self.assertEqual(
            {x["mode"] for x in plan["jobs"]},{"anchor","independent"})
        self.assertTrue(all("--train-cache" in x["train_argv"] for x in plan["jobs"]))

    def test_runtime_guard_blocks_unfeasible_full_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            log=root/"training.jsonl"
            log.write_text("".join(json.dumps({"seconds":3.})+"\n" for _ in range(20)))
            info=read_timing([log])
            self.assertAlmostEqual(info["seconds_p90"],3.)
            full=stage({"max_running_time_ms":"10800000"},
                manifest(),root="/repo",data="/repo/data/eurlex",run_root="/repo/runs/full",
                model="/model/qwen",seed_list=(43,44,45),pilot=False,
                timings=[log],output=root/"blocked_full",
                decision_reports=[result(),result(),result()])
            self.assertEqual(full["status"],"BLOCKED_ON_QZ_TIME_LIMIT")
            self.assertEqual(full["jobs"],[])
            self.assertEqual(len(list((root/"blocked_full").glob("*.json"))),1)
            denied=stage({"max_running_time_ms":"10800000"},
                manifest(),root="/repo",data="/repo/data/eurlex",run_root="/repo/runs/full",
                model="/model/qwen",seed_list=(43,44,45),pilot=False,
                timings=[log],output=root/"no_pilot",
                decision_reports=[])
            self.assertEqual(denied["status"],"BLOCKED_ON_P0_P1_EVIDENCE")

    def test_full_5k_pilot_matched_steps_are_not_120_plus_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            log=root/"training.jsonl"
            log.write_text("".join(json.dumps({"seconds":1.})+"\n" for _ in range(20)))
            p=stage({"max_running_time_ms":"10800000"},
                manifest(),root="/repo",data="/repo/data/eurlex",
                run_root="/repo/runs/pilot",model="/model/qwen",
                seed_list=(43,),pilot=True,timings=[log],output=root/"full5k")
            self.assertEqual(p["status"],"STAGED_NOT_SUBMITTED")
            self.assertEqual(p["steps_per_arm"],625)
            self.assertEqual(len(p["jobs"]),6)
            saved=list((root/"full5k").glob("*.json"))
            self.assertEqual(len(saved),7)
            train=json.loads(next(x for x in saved if "bce_continue-s43" in x.name).read_text())
            self.assertIn("--max-steps 625",train["command"])


if __name__=="__main__":
    unittest.main()
