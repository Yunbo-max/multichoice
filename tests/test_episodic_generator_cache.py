"""Causal generated-rule and frozen-Qwen prompt-cache regression tests.

These checks run on CPU and do NOT load Qwen weights:
  python -m unittest discover -s tests -p 'test_episodic_generator_cache.py' -v
"""
from collections import OrderedDict
import random
import unittest

import torch

from harness.gen_text_episodes import (
    CELLS, gen_episode, rotate, active_text, demo_text, state_text, validate_episode,
)
from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
from harness.build_episodic_features import encode_episode


class CausalRuleGenerationTests(unittest.TestCase):
    @staticmethod
    def initial_state(episode):
        text = episode["steps"][0]["decision"]["state"]
        return [int(label in text) for label in CELLS]

    def test_transition_under_only_previously_observed_rule(self):
        for seed in range(30):
            e = gen_episode(random.Random(seed), horizon=9, rule_switch_prob=0.7)
            self.assertTrue(validate_episode(e))
            active = self.initial_state(e)
            initial_rule = 1 if "cyclic right shift" in e["demonstrations"][0] else -1
            self.assertEqual(e["steps"][0]["oracle_rule_before_decision"], initial_rule)
            for t, event in enumerate(e["steps"]):
                current_rule = event["oracle_rule_before_decision"]
                self.assertEqual(event["decision"]["state"], state_text(active))
                self.assertEqual(event["gold"], rotate(active, current_rule))
                self.assertEqual(event["executed_set"], event["gold"])
                observed = event["post_observation"]
                self.assertIn(active_text(event["executed_set"]), observed)
                if t + 1 < len(e["steps"]):
                    next_rule = e["steps"][t + 1]["oracle_rule_before_decision"]
                    if next_rule != current_rule:
                        self.assertIn("The rule changed.", observed)
                        self.assertIn(demo_text(next_rule), observed)
                    else:
                        self.assertNotIn("The rule changed.", observed)
                else:
                    self.assertNotIn("The rule changed.", observed)
                active = event["executed_set"]

    def test_always_switch_is_disclosed_after_previous_action(self):
        e = gen_episode(random.Random(1), horizon=5, rule_switch_prob=1.0)
        rules = [s["oracle_rule_before_decision"] for s in e["steps"]]
        for i in range(4):
            self.assertEqual(rules[i], -rules[i + 1])
            self.assertIn("The rule changed.", e["steps"][i]["post_observation"])
        self.assertNotIn("The rule changed.", e["steps"][-1]["post_observation"])

    def test_private_oracle_metadata_not_forwarded_to_encoder(self):
        e = gen_episode(random.Random(5), horizon=3, rule_switch_prob=1.0)

        class FakeEncoder:
            def __init__(self):
                self.prompts = []
            def encode_demonstrations(self, demos):
                return torch.zeros(1, len(demos), 6)
            def encode_question(self, row):
                self.prompts.append(row)
                k = len(row["options"])
                return torch.zeros(1, 6), torch.zeros(1, k), torch.ones(1, k, dtype=torch.bool)
            def encode_option_meanings(self, row):
                return torch.zeros(1, len(row["options"]), 6)
            def encode_observation(self, text):
                return torch.ones(1, 6)

        fake = FakeEncoder()
        encoded = encode_episode(e, fake)
        self.assertEqual(len(encoded["steps"]), 3)
        self.assertEqual(len(fake.prompts), 3)
        for prompt in fake.prompts:
            self.assertNotIn("oracle_rule_before_decision", prompt)
            self.assertNotIn("gold", prompt)
            self.assertNotIn("post_observation", prompt)


class FrozenPromptCacheTests(unittest.TestCase):
    def test_same_prompt_is_encoded_once_and_order_is_preserved(self):
        encoder = FrozenQwenDecisionEncoder.__new__(FrozenQwenDecisionEncoder)
        encoder.device = torch.device("cpu")
        encoder._slot_cache = OrderedDict()
        encoder._slot_cache_max = 3
        encoder._slot_batch_size = 2
        calls = []

        def fake_forward(prompts):
            calls.extend(prompts)
            return torch.tensor([
                [float(ord(s[0])), float(len(s))] for s in prompts
            ])
        encoder._uncached_slot_vectors = fake_forward
        first = encoder._slot_vectors(["a", "b", "a"])
        self.assertEqual(calls, ["a", "b"])
        self.assertTrue(torch.equal(first[0], first[2]))
        second = encoder._slot_vectors(["b", "a"])
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(second[:, 0].tolist(), [98.0, 97.0])
        encoder._slot_vectors(["c", "d"])
        self.assertEqual(calls, ["a", "b", "c", "d"])
        self.assertLessEqual(len(encoder._slot_cache), 3)
        encoder._slot_vectors(["b"])  # least-recently-used was evicted
        self.assertEqual(calls[-1], "b")


if __name__ == "__main__":
    unittest.main()
