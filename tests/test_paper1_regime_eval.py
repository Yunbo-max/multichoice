import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static import pairwise_exact as dense
from paper_static import regime_eval as evaluator


class RegimeScoreTests(unittest.TestCase):
    def setUp(self):
        self.k=4
        self.scores=[.3,-.5,.2,.4]
        self.residual=[-.2,.1,-.4,.5,0.]
        self.pairs=[(0,1),(2,3)]
        self.weights=[1.0,-.8]
        J=[[0.]*4 for _ in range(4)]
        for (i,j),value in zip(self.pairs,self.weights):
            J[i][j]=J[j][i]=value
        q=dense.exact_distribution(self.scores,self.residual,J)
        m=dense.moments(q,self.k)
        self.teacher={"context_id":"dev-R-0000","regime":"R","strength":1.0,
            "true_marginals":m["marginals"],"true_count_probs":m["count_probs"],
            "true_pairwise_marginals":m["pairwise_marginals"],
            "true_set_probs":{str(k):v for k,v in q.items()}}
        self.rows=[]
        self.preds=[]
        for repeat,mask in enumerate([0b0011,0b1100]):
            row={"id":f"dev-R-0000/draw-{repeat}","group_id":"dev-R-0000",
                 "split":"dev","state":"pair tokens","question":"Choose all",
                 "selection_mode":"multi","allow_empty":True,
                 "options":[{"key":x,"desc":"test"} for x in "ABCD"],
                 "n_options":4,"target":[int(bool(mask>>i&1)) for i in range(4)]}
            self.rows.append(row)
            self.preds.append({"id":row["id"],"option_keys":list("ABCD"),
                               "candidate_logits":list(self.scores),
                               "count_residual_logits":list(self.residual),
                               "predicted_pairs":[list(p) for p in self.pairs],
                               "predicted_pair_weights":self.weights,
                               "edge_source":"oracle_diagnostic"})

    def test_oracle_prediction_has_zero_joint_and_count_kl(self):
        result=evaluator.evaluate(self.rows,[self.teacher],self.preds,"pair","dev")
        metrics=result["metrics_by_regime"]["R"]
        self.assertAlmostEqual(metrics["mean_joint_kl"],0,places=9)
        self.assertAlmostEqual(metrics["mean_count_kl"],0,places=9)
        self.assertTrue(result["not_a_learned_pair_result"])
        self.assertAlmostEqual(metrics["mean_edge_f1"],1.0)

    def test_wrong_matching_changes_joint_quality(self):
        predictions=[dict(p,predicted_pairs=[[0,2],[1,3]],
                          edge_source="learned") for p in self.preds]
        result=evaluator.evaluate(self.rows,[self.teacher],predictions,"pair","dev")
        self.assertGreater(result["metrics_by_regime"]["R"]["mean_joint_kl"],.05)
        self.assertFalse(result["not_a_learned_pair_result"])
        self.assertAlmostEqual(result["metrics_by_regime"]["R"]["mean_edge_f1"],0.0)

    def test_no_predictions_missing_or_test_leak(self):
        with self.assertRaisesRegex(ValueError,"missing"):
            evaluator.evaluate(self.rows,[self.teacher],self.preds[:-1],"pair","dev")
        with self.assertRaisesRegex(ValueError,"split"):
            evaluator.evaluate(self.rows,[self.teacher],self.preds,"pair","test")

    def test_independent_bce_matches_true_law(self):
        scores=[.3,-.5,.2,.4]
        prob=dense.exact_distribution(scores)
        moment=dense.moments(prob,4)
        teacher=dict(self.teacher,regime="I",true_marginals=moment["marginals"],
           true_count_probs=moment["count_probs"],
           true_pairwise_marginals=moment["pairwise_marginals"],
           true_set_probs={str(k):v for k,v in prob.items()})
        base=[{"id":p["id"],"option_keys":p["option_keys"],"logits":scores} for p in self.preds]
        report=evaluator.evaluate(self.rows,[teacher],base,"bce","dev")
        self.assertAlmostEqual(report["metrics_by_regime"]["I"]["mean_joint_kl"],0,places=9)


if __name__=="__main__":
    unittest.main()
