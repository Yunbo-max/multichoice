import json
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static.regime_oracle_check import example_regimes


class OracleMechanism(unittest.TestCase):
    def test_three_regimes_have_distinct_capacity_boundaries(self):
        report=example_regimes()
        self.assertEqual(report["status"],"SYNTHETIC_ORACLE_CAPACITY_ONLY_NOT_LLM_TRAINING")
        i=report["scenarios"]["I_independent"]
        c=report["scenarios"]["C_cardinality"]
        r=report["scenarios"]["R_pairwise"]
        self.assertAlmostEqual(i["best_bce_excess_joint_kl"],0,places=9)
        self.assertGreater(c["bce_excess_joint_kl"],0.1)
        self.assertAlmostEqual(c["oracle_count_anchor_excess_joint_kl"],0,places=9)
        self.assertGreater(r["anchor_excess_joint_kl"],0.5)
        self.assertAlmostEqual(r["oracle_pairwise_excess_joint_kl"],0,places=9)
        self.assertTrue(r["marginals_and_count_invariant_under_rewiring"])
        self.assertGreater(abs(r["cooccurrence_p_AB_original"]-
                               r["cooccurrence_p_AB_rewired"]),.05)


if __name__=="__main__":
    unittest.main()
