"""No-GPU manifest validation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_static"))
from plan_experiments import ARMS, make_plan

class StaticManifestTests(unittest.TestCase):
    def test_all_arms_seeds_and_equal_budgets(self):
        rows = make_plan("MODEL", "modules.json", "train.jsonl", "dev.jsonl",
                         "runs/paper_static", seeds=(43,44,45))
        self.assertEqual(len(rows), len(ARMS)*3)
        self.assertEqual(len(set(x["output"] for x in rows)), len(rows))
        self.assertEqual({x["update_budget"] for x in rows}, {79})
        self.assertEqual({x["model"] for x in rows}, {"MODEL"})
        for row in rows:
            self.assertEqual(row["status"], "NOT_RUN")
            self.assertEqual(row["argv"][0], "torchrun")
            self.assertIn("--eval-every-frac", row["argv"])
            self.assertEqual(row["argv"][row["argv"].index("--eval-every-frac")+1], "1.0")
            self.assertEqual("--count-anchor" in row["argv"], row["count_anchor"])
            self.assertEqual("--count-anchor-detach" in row["argv"], row["count_anchor_detach"])
    def test_reject_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            make_plan("m","u","t","v","runs",seeds=(43,43))

if __name__ == "__main__":
    unittest.main()
