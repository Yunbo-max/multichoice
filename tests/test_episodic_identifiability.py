"""Identifiability, protocol, and legacy-symmetry regression tests.

Run:
  python -m unittest discover -s tests -p test_episodic_identifiability.py -v

Designed to fail if anyone changes the synthetic memory benchmark back to a
ring with ambiguous states or a decoder capable of choosing every option.
"""
import itertools
import random
import unittest

import torch

from harness.gen_text_episodes import (
    ACTIVE_COUNT, CELLS, demo_text, gen_episode, rotate, validate_episode,
)
from harness.set_distillation import allowed_counts, input_digest
from harness.train_multiselect import make_body
from harness.recurrent_set_decision import set_map
from harness.build_episodic_features import encode_episode


class FiveCellIdentifiability(unittest.TestCase):
    def test_exhaustive_two_action_states_disambiguate_every_rule(self):
        self.assertEqual(len(CELLS), 5)
        self.assertEqual(ACTIVE_COUNT, 2)
        for selected in itertools.combinations(range(len(CELLS)), ACTIVE_COUNT):
            current = [int(i in selected) for i in range(len(CELLS))]
            left = rotate(current, -1)
            right = rotate(current, +1)
            self.assertNotEqual(left, right)
            self.assertEqual(sum(left), ACTIVE_COUNT)
            self.assertEqual(sum(right), ACTIVE_COUNT)
            # With identical observable state but hidden +/- rule, any one
            # predicted set can match at most ONE of these two answers.
            for guess in (left, right):
                self.assertEqual(int(guess == left) + int(guess == right), 1)

    def test_no_absorbing_full_set_or_empty_set_is_permitted(self):
        for seed in range(100):
            item = gen_episode(random.Random(seed), 12, 0.7)
            self.assertTrue(validate_episode(item))
            self.assertEqual(item["environment_schema"], "cyclic5_fixed2_v3")
            for step in item["steps"]:
                self.assertEqual(step["decision"]["allowed_cardinalities"], [2])
                self.assertEqual(sum(step["gold"]), 2)
                self.assertEqual(sum(step["executed_set"]), 2)
                self.assertNotEqual(
                    rotate(step["executed_set"], 1),
                    rotate(step["executed_set"], -1),
                )

    def test_demo_gives_example_not_explicit_direction_name(self):
        self.assertNotEqual(demo_text(-1), demo_text(1))
        for d in (-1, 1):
            self.assertNotIn("left shift", demo_text(d))
            self.assertNotIn("right shift", demo_text(d))
            self.assertNotIn("cyclic right", demo_text(d))

    def test_explicit_constraint_changes_prompt_and_density_support(self):
        row = {
            "state": "The active components are Alpha, Gamma.",
            "question": "Which cells should be active next?",
            "options": [
                {"key": f"cell-{i}", "desc": f"Activate {name}"}
                for i, name in enumerate(CELLS)
            ],
            "selection_mode": "multi",
            "allow_empty": False,
        }
        constrained = dict(row, allowed_cardinalities=[2])
        self.assertEqual(allowed_counts(constrained), [2])
        self.assertEqual(allowed_counts(row), list(range(1, 6)))
        self.assertIn("Select exactly 2", make_body(constrained, list("ABCDE")))
        self.assertNotIn("Select exactly 2", make_body(row, list("ABCDE")))
        self.assertNotEqual(input_digest(constrained), input_digest(row))
        self.assertEqual(allowed_counts({"options": row["options"], "selection_mode": "single"}), [1])
        for counts in ([0], [2, 2], [3, 2], [True], ["2"], [5, 6], None):
            with self.assertRaises(ValueError):
                allowed_counts(dict(constrained, allowed_cardinalities=counts))

    def test_cardinality_map_decodes_two_even_for_high_all_option_scores(self):
        scores = torch.tensor([8., 7., 6., 5., 4.])
        counts = torch.zeros(6)
        result = set_map(scores, counts, [2])
        self.assertEqual(result["indices"], [0, 1])
        self.assertEqual(result["count_probs"][2], 1.)
        self.assertEqual(len(result["indices"]), 2)

    def test_qwen_builder_whitelists_public_constraint_but_not_private_rule(self):
        episode = gen_episode(random.Random(27), 3, 0.6)

        class FakeEncoder:
            def __init__(self):
                self.rows = []
            def encode_question(self, row):
                self.rows.append(row)
                k = len(row["options"])
                return torch.zeros(1, 8), torch.zeros(1, k), torch.ones(1, k, dtype=torch.bool)
            def encode_option_meanings(self, row):
                return torch.zeros(1, len(row["options"]), 8)
            def encode_observation(self, value):
                return torch.zeros(1, 8)
            def encode_demonstrations(self, values):
                return torch.zeros(1, len(values), 8)

        fake = FakeEncoder()
        encoded = encode_episode(episode, fake)
        self.assertEqual(len(encoded["steps"]), 3)
        for row, recorded in zip(fake.rows, encoded["steps"]):
            self.assertEqual(row["allowed_cardinalities"], [2])
            self.assertEqual(recorded["allowed"], [2])
            for key in ("gold", "oracle_rule_before_decision", "post_observation"):
                self.assertNotIn(key, row)

    def test_corrupted_rule_feedback_and_gold_are_rejected_before_encoding(self):
        import copy
        episode = gen_episode(random.Random(31), 4, 1.0)

        class NeverCalledEncoder:
            def __getattr__(self, name):
                raise AssertionError("Qwen must not run on corrupted episode")

        corrupt = copy.deepcopy(episode)
        corrupt["steps"][0]["post_observation"] = "The rule changed yesterday."
        with self.assertRaises(ValueError):
            encode_episode(corrupt, NeverCalledEncoder())
        corrupt = copy.deepcopy(episode)
        corrupt["steps"][1]["gold"] = corrupt["steps"][0]["gold"]
        with self.assertRaises(ValueError):
            encode_episode(corrupt, NeverCalledEncoder())

    def test_old_reset_trajectory_297_is_old_symmetric_cohort_fraction(self):
        # Replicate old FOUR-cell generator's exact random call order.
        # 256 held-out episodes, seed 9043, horizon 8, switch p 0.25.
        rng = random.Random(9043)
        opposite = 0
        for _ in range(256):
            rng.choice((-1, 1))
            selected = rng.sample(range(4), 2)
            opposite += int((selected[0] - selected[1]) % 4 == 2)
            for _ in range(7):
                rng.random()
            rng.getrandbits(32)
        self.assertEqual(opposite, 76)
        self.assertAlmostEqual(opposite / 256, 0.296875)


if __name__ == "__main__":
    unittest.main()
