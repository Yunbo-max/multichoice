import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_static import paired_count_audit as audit
from paper_static import pairwise_exact as pairwise
from paper_static import plan_dev_eval as planner


class CountAuditTests(unittest.TestCase):
    def test_poisson_binomial_matches_enumeration_and_protocol(self):
        for scores in ([1.,-1.,.3],[5.,0.,-8.,2.],[1000.,-1000.,8.,0.]):
            k=len(scores)
            for allowed in (range(k+1),range(1,k+1),[1]):
                got=audit.log_count_distribution(scores,allowed)
                denominator=0
                exact=[0.]*(k+1)
                for mask in range(1<<k):
                    m=mask.bit_count()
                    if m in allowed:
                        weight=math.exp(sum(scores[i] for i in range(k) if mask>>i&1)-max(0,sum(max(s,0) for s in scores)))
                        exact[m]+=weight
                        denominator+=weight
                for m in allowed:
                    if denominator:
                        self.assertAlmostEqual(math.exp(got[m]),exact[m]/denominator,places=10)

    def test_logit_common_shift_restored(self):
        a=audit.log_count_distribution([0.,0.],[0,1,2])
        b=audit.log_count_distribution([2.,2.],[0,1,2])
        self.assertNotAlmostEqual(a[2],b[2])
        self.assertAlmostEqual(math.exp(a[2]),0.25)

    def _row(self,idx,seed=43):
        row={"id":f"sample-{idx}","source":"MultiRC","group_id":f"passage-{idx//2}",
             "split":"dev","state":"state", "question":f"Q{idx}?",
             "options":[{"key":"A","text":"a"},{"key":"B","text":"b"},{"key":"C","text":"c"}],
             "selection_mode":"multi","allow_empty":False,
             "target":[1,int(idx%2==0),0]}
        scores=[0.8,0.6,-1.2] if idx%2==0 else [.8,-.8,-1.2]
        probs=[1/(1+math.exp(-x)) for x in scores]
        gold_indices={i for i,y in enumerate(row["target"]) if y}
        base={"id":row["id"],"option_keys":["A","B","C"],
              "input_digest":audit.input_digest(row),
              "p":probs,"logits":scores,"predicted_indices":[i for i,p in enumerate(probs) if p>=.5],
              "checkpoint_id":f"paper1|bce_continue|seed_{seed}"}
        count_log=audit.log_count_distribution(scores,[1,2,3])
        anch=dict(base,checkpoint_id=f"paper1|set_ce_anchor|seed_{seed}",
                  decoder="set_map",predicted_indices=sorted(gold_indices),
                  count_probs=[0 if t==-math.inf else math.exp(t) for t in count_log],
                  count_log_probs=[None if t==-math.inf else t for t in count_log],
                  candidate_logits=scores, count_residual_logits=[0.0]*4)
        return row,base,anch

    def test_pairing_and_quartiles_and_cluster_ci(self):
        data=[self._row(i) for i in range(8)]
        result=audit.paired_rows([z[0] for z in data],[z[1] for z in data],[z[2] for z in data],43)
        self.assertEqual(len(result),8)
        self.assertEqual(len(audit.quartile_buckets(result)),4)
        self.assertEqual(sum(len(g) for g in audit.quartile_buckets(result).values()),8)
        stat=audit.analyze_seed(result, 40, 42)
        self.assertEqual(stat["overall"]["groups"],4)
        self.assertAlmostEqual(stat["overall"]["anchor_residual_span"],0.0)
        self.assertIn("anchor_minus_zero_em",stat["overall"])
        self.assertEqual(stat["overall"]["delta_em_cluster_bootstrap_ci95"],
                         audit.bootstrap_ci(result,"delta_em",40,42))
        with self.assertRaisesRegex(ValueError,"ID sets differ"):
            audit.paired_rows([z[0] for z in data],[z[1] for z in data],
                              [z[2] for z in data[:-1]],43)

    def test_cli_three_seed_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            gold=[self._row(i)[0] for i in range(8)]
            gold_path=root/"dev.jsonl"
            gold_path.write_text("".join(json.dumps(x)+"\n" for x in gold),encoding="utf-8")
            for seed in (43,44,45):
                examples=[self._row(i,seed) for i in range(8)]
                for arm,index in (("bce_continue",1),("set_ce_anchor",2)):
                    dest=root/"pred"/f"seed_{seed}"/arm/"predictions.jsonl"
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_text("".join(json.dumps(r[index])+"\n" for r in examples),encoding="utf-8")
            summary=root/"count_report.json"
            audit.main(["--gold",str(gold_path),"--predictions-root",str(root/"pred"),
                       "--output",str(summary),"--bootstrap","10"])
            result=json.loads(summary.read_text())
            self.assertEqual(result["status"],"COMPLETE")
            self.assertEqual(len(result["seeds"]),3)
            self.assertTrue(all(v["overall"]["n"]==8 for v in result["seeds"].values()))
            with self.assertRaises(SystemExit):
                audit.main(["--gold",str(gold_path),"--predictions-root",str(root/"pred"),
                           "--output",str(summary)])

    def test_anchor_residual_is_input_dependent_not_only_bias(self):
        r,b,a=self._row(0)
        a["count_residual_logits"]=[0.0,0.3,-0.5,0.6]
        stat=audit.paired_rows([r],[b],[a],43)
        self.assertAlmostEqual(stat[0]["anchor_residual_span"],1.1)

    def test_reject_test_leak_and_checkpoint_mismatch(self):
        r,b,a=self._row(0)
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            audit.paired_rows([r],[b],[dict(a,checkpoint_id="wrong")],43)
        r["split"]="test"
        with self.assertRaisesRegex(ValueError,"DEV-only"):
            audit.paired_rows([r],[b],[a],43)


class PairwiseExactTests(unittest.TestCase):
    def test_exact_bce_reduction(self):
        scores=[1.1,-.7,.2]
        probabilities=pairwise.exact_distribution(scores)
        sigmoid=[1/(1+math.exp(-x)) for x in scores]
        for mask,p in probabilities.items():
            expected=math.prod(sigmoid[i] if mask>>i&1 else 1-sigmoid[i] for i in range(3))
            self.assertAlmostEqual(p,expected,places=12)

    def test_anchor_reduction_and_nonempty_conditional(self):
        scores=[.2,-.9,1.]
        residual=[-0.4,.6,-.2,.3]
        p=pairwise.exact_distribution(scores,residual=residual,counts=[1,2,3])
        self.assertAlmostEqual(sum(p.values()),1)
        self.assertNotIn(0,p)
        for mask, prob in p.items():
            for other,pr in p.items():
                same_count=mask.bit_count()==other.bit_count()
                if same_count:
                    lhs=math.log(prob/pr)
                    rhs=sum(scores[i]*(((mask>>i)&1)-((other>>i)&1)) for i in range(3))
                    self.assertAlmostEqual(lhs,rhs)

    def test_pairwise_data_with_same_first_moments(self):
        result=pairwise.demo()
        self.assertTrue(result["pairwise_training_decreased_nll"])
        self.assertAlmostEqual(result["nll"]["bce"],math.log(16))
        self.assertAlmostEqual(result["nll"]["anchor_fixed_count"],math.log(6))
        self.assertAlmostEqual(result["nll"]["bce_protocol_matched"],math.log(6))
        self.assertLess(result["nll"]["pairwise_fit"],0.73)
        self.assertLess(result["probability_on_invalid_2_sets"]["pairwise_fit"],0.02)
        self.assertAlmostEqual(result["probability_on_invalid_2_sets"]["anchor_fixed_count"],2/3)

    def test_pairwise_gradient_finite_difference(self):
        s=[.3,-.4,.2]
        J=[[0.0]*3 for _ in range(3)]
        J[0][1]=J[1][0]=0.2
        target={0b011:1.0}
        rho=pairwise.moments(pairwise.exact_distribution(s,interactions=J,counts=[2]),3)
        analytic=rho["pairwise_marginals"][0][1]-1
        eps=1e-5
        losses=[]
        for sign in (1,-1):
            J2=[list(row) for row in J]
            J2[0][1]+=sign*eps
            J2[1][0]+=sign*eps
            losses.append(pairwise.dataset_nll(pairwise.exact_distribution(s,interactions=J2,counts=[2]),target))
        self.assertAlmostEqual(analytic,(losses[0]-losses[1])/(2*eps),places=8)

    def test_reject_large_exact(self):
        with self.assertRaisesRegex(ValueError,"K <="):
            pairwise.exact_distribution([0.]*13)


class PlanTests(unittest.TestCase):
    def test_safe_conversion_never_uses_test_or_rm(self):
        template={"name":"paper1-eval-bce_continue-s43", "command":(
            'set -uo pipefail\nR=/inspire/jev\n'
            'rm -rf "/inspire/jev/runs/paper_static_eval/s44/seed_43/bce_continue"\n'
            'cd "/inspire/jev"\n'
            'python harness/eval_multiselect.py --input "/inspire/jev/data/multiselect_grouped_multi/test.jsonl" '
            '--expected-split test --output "/inspire/jev/runs/paper_static_eval/s44/seed_43/bce_continue" '
            '> "/inspire/jev/logs/paper1-eval-bce_continue-s43.log" 2>&1\n')}
        converted=planner.convert(template,"bce_continue",43)
        cmd=converted["command"]
        self.assertIn("--expected-split dev",cmd)
        self.assertIn("/dev.jsonl",cmd)
        self.assertIn("test ! -e",cmd)
        self.assertIn('test "$(git rev-parse --abbrev-ref HEAD)"',cmd)
        self.assertNotIn("rm -rf",cmd)
        self.assertNotIn("/test.jsonl",cmd)


if __name__=="__main__":
    unittest.main()
