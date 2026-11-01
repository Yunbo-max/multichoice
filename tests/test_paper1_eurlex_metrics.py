"""EUR-LEX official metric contract, 100-label and full-set legal-NLL tests."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static.eurlex_metrics import score
from harness.set_distillation import input_digest, structured_prediction


def gold_row(k=100):
    options=[{"key":str(100163+i),"desc":f"topic {i}"} for i in range(k)]
    target=[int(i in (1,2)) for i in range(k)]
    return {"id":"eurlex:validation:000001","source":"lexglue_eurlex",
            "lang":"en","split":"dev","state":"Act on trade policy",
            "question":"Select all EuroVoc concepts applicable to the legal document.",
            "options":options,"n_options":k,
            "selection_mode":"multi","allow_empty":True,"target":target}


def candidate_scores(k):
    return [-9.0 if j not in (1,2) else 9.0 for j in range(k)]


class EurlexMetrics(unittest.TestCase):
    def test_bce_macro_micro_f1_and_joint_probability(self):
        row=gold_row()
        s=candidate_scores(100)
        sigmoid=lambda t:1/(1+math.exp(-t))
        pred={"id":row["id"],"logits":s,
              "p":[sigmoid(z) for z in s],
              "option_keys":[o["key"] for o in row["options"]],
              "input_digest":input_digest(row)}
        report=score([row],[pred])
        self.assertAlmostEqual(report["official_micro_f1"],1)
        self.assertAlmostEqual(report["official_macro_f1"],.02)
        self.assertAlmostEqual(report["exact_set_em"],1)
        self.assertLess(report["proper_joint_set_nll"],.05)
        self.assertLess(report["count_nll"],.05)
        self.assertEqual(report["n_options"],100)
        self.assertEqual(report["n_gold_positive"],2)
        self.assertEqual(report["n_predicted_positive"],2)
        with self.assertRaisesRegex(ValueError,"complete fixed 100"):
            score([row],[pred],expected_k=99)

    def test_anchor_same_legal_support_with_zero_residual(self):
        row=gold_row()
        s=candidate_scores(100)
        ind=structured_prediction(s,[0.0]*101,row,count_anchor=True)
        ind.update({"id":row["id"],
                    "candidate_logits":s,
                    "option_keys":[o["key"] for o in row["options"]],
                    "input_digest":input_digest(row)})
        report=score([row],[ind])
        self.assertEqual(report["structured_readout_count"],1)
        self.assertAlmostEqual(report["official_micro_f1"],1)
        self.assertAlmostEqual(report["exact_set_em"],1)
        self.assertLess(report["proper_joint_set_nll"],.05)
        self.assertLess(report["count_nll"],.05)
        self.assertAlmostEqual(report["count_nll"],
                               -math.log(ind["count_probs"][2]),places=5)

    def test_reject_missing_predictions_and_wrong_option_order(self):
        row=gold_row()
        with self.assertRaisesRegex(ValueError,"mismatch"):
            score([row],[])
        s=candidate_scores(100)
        pred={"id":row["id"],"logits":s,
              "p":[1/(1+math.exp(-z)) for z in s],
              "option_keys":[o["key"] for o in row["options"]][::-1]}
        with self.assertRaisesRegex(ValueError,"mismatch"):
            score([row],[pred])


if __name__=="__main__":
    unittest.main()
