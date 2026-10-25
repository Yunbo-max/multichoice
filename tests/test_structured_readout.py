"""Unit tests for the structured (6-candidate) set readout."""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from structured_readout import (
    baselines,
    reachable_successors,
    structured_logits,
    structured_nll,
    structured_predict,
    structured_probs,
)


class StructuredReadoutTests(unittest.TestCase):
    def test_reachable_successors_are_distinct_and_cover_all_shifts(self):
        state = [1, 1, 0, 0, 0, 0, 0]  # cells 0,1 active
        succ = reachable_successors(state)
        self.assertEqual(set(succ.keys()), {1, 2, 3, 4, 5, 6})
        sets = {tuple(v) for v in succ.values()}
        self.assertEqual(len(sets), 6, "six shifts must yield six distinct sets")

    def test_structured_logits_sum_the_shifted_cells(self):
        state = [1, 1, 0, 0, 0, 0, 0]
        scores = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        logits = structured_logits(state, scores)
        # shift by k=1 -> cells 1,2 active; scores[1]+scores[2] = 0
        self.assertEqual(logits[1], 0.0)
        # shift by k=6 -> cells 0,6 active? shift(+6)=shift(-1) -> cells 6,0; sum scores = 10
        self.assertAlmostEqual(logits[6], 10.0)

    def test_predict_returns_highest_energy_shift(self):
        state = [1, 0, 0, 0, 0, 0, 0]
        scores = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        pred = structured_predict(state, scores, tau=1.0)
        # cell 3 has high score; shift k=3 moves the active cell 0 -> 3
        self.assertEqual(pred["shift"], 3)
        self.assertEqual(pred["set"][3], 1)

    def test_probs_sum_to_one_and_tau_scales(self):
        state = [1, 0, 0, 0, 0, 0, 0]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
        p1 = structured_probs(state, scores, tau=1.0)
        p2 = structured_probs(state, scores, tau=100.0)
        self.assertAlmostEqual(sum(p1.values()), 1.0, places=9)
        self.assertAlmostEqual(sum(p2.values()), 1.0, places=9)
        # large tau flattens the distribution
        self.assertLess(max(p2.values()), max(p1.values()))

    def test_nll_and_baselines(self):
        state = [1, 0, 0, 0, 0, 0, 0]
        scores = [0.0] * 7
        # flat scores -> uniform over 6 shifts -> NLL = log 6
        self.assertAlmostEqual(
            structured_nll(state, scores, 1, tau=1.0), math.log(6), places=6
        )
        b = baselines(7)
        self.assertAlmostEqual(b["random_legal_shift_em"], 1 / 6, places=9)
        self.assertAlmostEqual(b["uniform_shift_nll"], math.log(6), places=9)
        self.assertAlmostEqual(b["uniform_subsets_nll"], math.log(21), places=9)


if __name__ == "__main__":
    unittest.main()
