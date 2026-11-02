"""CPU exact-MAP regression for both independent and anchored count heads.

The model is P(S|x)=rho_|S| * exp(sum_i s_i) / Z_|S|(s).
An independent head learns rho_m=softmax(c_m).
An anchored head learns rho_m=softmax(a_m+log Z_m(s)).
The two models share the SAME exact set-MAP decoder; count mode is NOT
necessarily the MAP cardinality. No gold count is consulted at inference.
"""
from __future__ import annotations

import itertools
import math
import random
import unittest

from harness.set_distillation import (
    structured_prediction, count_anchor_logits, allowed_counts, log_partitions,
    input_digest,
)
from harness.multiselect_metrics import prediction_records


def row(k, *, allow_empty=True, mode="multi"):
    return {
        "id": "set-map-oracle", "state": "state", "question": "select",
        "source": "unit", "lang": "en", "split": "dev",
        "options": [{"key": str(i), "desc": f"option {i}"} for i in range(k)],
        "n_options": k, "selection_mode": mode, "allow_empty": allow_empty,
        "target": [1] + [0] * (k - 1),
    }


def _logsum(xs):
    mx = max(xs)
    return mx + math.log(sum(math.exp(x - mx) for x in xs))


def exhaustive_map(scores, counts, question, *, anchor=False):
    """Mathematically independent oracle: enumerate actual subset log P."""
    k = len(scores)
    allowed = allowed_counts(question)
    z = log_partitions(scores)
    count_logits = count_anchor_logits(scores, counts) if anchor else counts
    normalizer = _logsum([count_logits[m] for m in allowed])
    probabilities = []
    for mask in range(1 << k):
        bits = [(mask >> i) & 1 for i in range(k)]
        m = sum(bits)
        if m not in allowed:
            continue
        logp = (count_logits[m] - normalizer
                + sum(score for i, score in enumerate(scores) if bits[i])
                - z[m])
        probabilities.append((logp, mask))
    maximum = max(p for p, _ in probabilities)
    best = [mask for p, mask in probabilities if abs(p-maximum) < 1e-9]
    return maximum, best


class ExactMap(unittest.TestCase):
    def test_count_mode_is_not_most_probable_subset(self):
        # count=2 has MORE aggregate mass, spread over 6 different sets.
        # Each singleton has probability .45/4=.1125 whereas each pair
        # has .55/6=.0916667. Correct MAP has ONE candidate.
        scores = [0.0]*4
        counts = [-1000.0, math.log(.45), math.log(.55), -1000.0, -1000.0]
        question = row(4, allow_empty=False)
        pred = structured_prediction(scores, counts, question)
        self.assertEqual(max(range(5), key=lambda m: pred["count_probs"][m]), 2)
        self.assertEqual(len(pred["predicted_indices"]), 1)
        self.assertAlmostEqual(
            math.exp(pred["predicted_set_log_probability"]), .45/4, places=11
        )

    def test_brute_force_all_regimes_various_K_and_protocols(self):
        rng = random.Random(20261102)
        for k in range(2, 8):
            for mode, empty in (("multi", True), ("multi", False), ("single", False)):
                question = row(k, allow_empty=empty, mode=mode)
                for anchor in (False, True):
                    for _ in range(7):
                        scores = [rng.uniform(-2.3, 2.3) for i in range(k)]
                        counts = [rng.uniform(-2.5, 2.5) for i in range(k+1)]
                        actual = structured_prediction(
                            scores, counts, question, count_anchor=anchor
                        )
                        reference, best_masks = exhaustive_map(
                            scores, counts, question, anchor=anchor
                        )
                        mask = sum(1 << i for i in actual["predicted_indices"])
                        self.assertIn(mask, best_masks)
                        self.assertAlmostEqual(
                            actual["predicted_set_log_probability"],
                            reference, places=9
                        )
                        self.assertAlmostEqual(
                            sum(actual["count_probs"]), 1, places=10
                        )
                        self.assertEqual(len(actual["p"]), k)

    def test_independent_head_is_invariant_to_global_option_score_shift(self):
        # P(S | |S|=m) is invariant under s_i -> s_i + C. The WRONG
        # raw c_m + top_m(s) implementation fails this identity.
        question = row(4, allow_empty=False)
        scores = [-1.8, -0.4, 0.3, 0.6]
        counts = [-100.0, 1.7, -0.3, 0.2, -1.0]
        original = structured_prediction(scores, counts, question)
        for offset in (-10000.0, 10000.0):
            shifted = structured_prediction(
                [x+offset for x in scores], counts, question
            )
            self.assertEqual(shifted["predicted_indices"],
                             original["predicted_indices"])
            self.assertAlmostEqual(
                shifted["predicted_set_log_probability"],
                original["predicted_set_log_probability"], places=8
            )
            for a, b in zip(shifted["p"], original["p"]):
                self.assertAlmostEqual(a, b, places=8)

    def test_anchor_map_reduces_to_independent_BCE_and_respects_shift(self):
        scores = [-1.5, .7, -.5, 2.0]
        question = row(4, allow_empty=True)
        predicted = structured_prediction(scores, [0.] * 5, question,
                                          count_anchor=True)
        self.assertEqual(predicted["predicted_indices"], [1, 3])
        shifted = structured_prediction([x+10000 for x in scores], [0.]*5,
                                        question, count_anchor=True)
        self.assertEqual(shifted["predicted_indices"], [0,1,2,3])

    def test_large_K_independent_head_exact_count_without_exponential_search(self):
        k = 100
        question = row(k, allow_empty=True)
        scores = [((i*17) % 23 - 11)*.1 for i in range(k)]
        counts = [-1e8]*(k+1)
        counts[5] = 0.0
        predicted = structured_prediction(scores, counts, question)
        shifted = structured_prediction([s+10000 for s in scores],
                                        counts, question)
        self.assertEqual(len(predicted["predicted_indices"]), 5)
        self.assertEqual(predicted["predicted_indices"],
                         shifted["predicted_indices"])
        self.assertAlmostEqual(predicted["count_probs"][5],1.0)
        self.assertAlmostEqual(
            predicted["predicted_set_log_probability"],
            shifted["predicted_set_log_probability"],places=8
        )

    def test_auditor_accepts_correct_independent_raw_count_logits(self):
        question=row(4, allow_empty=False)
        scores=[0.]*4
        counts=[-1000.,math.log(.45),math.log(.55),-1000.,-1000.]
        correct=structured_prediction(scores,counts,question,count_anchor=False)
        correct.update(id=question["id"], candidate_logits=scores,
                       count_residual_logits=counts,
                       option_keys=[o["key"] for o in question["options"]],
                       input_digest=input_digest(question))
        actual=prediction_records([question],[correct],threshold=.5,decoder="auto")
        self.assertEqual(len(actual[0]["predicted_indices"]),1)
        self.assertEqual(actual[0]["decoder"],"set_map")
        self.assertAlmostEqual(actual[0]["gold_set_nll"],
                               -math.log(.45/4),places=9)
        # Old buggy decoder incorrectly selected a pair because 0.55>0.45.
        invalid=dict(correct,predicted_indices=[0,1])
        with self.assertRaisesRegex(ValueError,"set_map indices"):
            prediction_records([question],[invalid],threshold=.5,decoder="auto")

    def test_auditor_accepts_anchor_with_raw_residual(self):
        question=row(5, allow_empty=True)
        scores=[-2.0, 1.0, -.4, .5, -1.2]
        counts=[.1, -.5, 1.2, -.1, .3, -.6]
        pred=structured_prediction(scores,counts,question,count_anchor=True)
        pred.update(id=question["id"],candidate_logits=scores,
                    count_residual_logits=counts,
                    option_keys=[o["key"] for o in question["options"]],
                    input_digest=input_digest(question))
        result=prediction_records([question],[pred],decoder="auto")
        self.assertEqual(result[0]["predicted_indices"],
                         pred["predicted_indices"])
        self.assertAlmostEqual(result[0]["predicted_set_log_probability"],
                               pred["predicted_set_log_probability"], places=8)


if __name__ == "__main__":
    unittest.main()
