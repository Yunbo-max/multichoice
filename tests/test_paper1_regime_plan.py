import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static import plan_regime_pilots as planner


class PilotPlanTests(unittest.TestCase):
    def test_plan_is_13_unsubmitted_matched_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            student=d/"student"
            student.mkdir()
            (student/"train.jsonl").write_text('{"id":"tr"}\n')
            (student/"dev.jsonl").write_text('{"id":"dv"}\n')
            plan=planner.plan(d,Path("/models/Qwen4B"),
                Path("innerjev/configs/4B/modules.json"),
                Path("/runs/case"),Path("/runs/case/warm_bce_shared/final"),
                seeds=(43,44,45),stage1_steps=1,stage2_steps=1)
            self.assertEqual(plan["status"],"PLAN_ONLY_NO_GPU_JOB")
            self.assertEqual(len(plan["rows"]),13)
            jobs=[r for r in plan["rows"] if r["phase"]=="paired-continuation"]
            self.assertEqual(len(jobs),12)
            self.assertEqual(len({(j["seed"],j["arm"]) for j in jobs}),12)
            self.assertTrue(all(j["status"]=="NOT_RUN" for j in plan["rows"]))
            self.assertEqual({j["requires_checkpoint"] for j in jobs},
                             {"/runs/case/warm_bce_shared/final"})
            self.assertTrue(all("--lora" in j["argv"] for j in jobs))
            for j in jobs:
                self.assertIn("--target",j["argv"])
                self.assertIn("gold",j["argv"])
                self.assertIn("--eval-every-frac",j["argv"])
            self.assertFalse(any("pairwise" in j["arm"] for j in jobs))

    def test_plan_refuses_absent_or_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            with self.assertRaisesRegex(ValueError,"generate"):
                planner.plan(d,Path("/m"),Path("/modules"),d,d,seeds=(43,44))
            (d/"student").mkdir()
            for n in ("train","dev"):
                (d/"student"/f"{n}.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"seed"):
                planner.plan(d,Path("/m"),Path("/modules"),d,d,seeds=(43,43))


if __name__=="__main__":
    unittest.main()
