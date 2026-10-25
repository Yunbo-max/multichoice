"""Online rule-environment tests using a fake encoder (no Qwen weights)."""
from copy import deepcopy
import random
import unittest
import torch

from harness.gen_text_episodes import gen_episode, active_text, CELLS
from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from harness.eval_episodic_text_online import initial_state, run_episode


class FakeEncoder:
    device = torch.device("cpu")
    hidden_size = 8

    def __init__(self):
        self.observed = []
        self.prompt_rows = []

    def encode_demonstrations(self, texts):
        return torch.zeros(1, len(texts), self.hidden_size)

    def encode_question(self, row):
        self.prompt_rows.append(row.copy())
        # Deliberately wrong fixed first-choice policy; verifies that
        # actual selected action, not oracle logged action, drives next state.
        return (
            torch.zeros(1, self.hidden_size),
            torch.tensor([[10., 9., -10., -10., -10.]]),
            torch.ones(1, len(CELLS), dtype=torch.bool),
        )

    def encode_observation(self, text):
        self.observed.append(text)
        return torch.zeros(1, self.hidden_size)

    def encode_option_meanings(self, row):
        return torch.zeros(1, len(CELLS), self.hidden_size)


class OnlineExecutionTests(unittest.TestCase):
    def make_agent(self):
        return EpisodicSetDecision(EpisodicSetConfig(
            context_dim=8, latent_dim=16, max_options=len(CELLS), inner_steps=0,
            require_semantic_actions=True,
        )).eval()

    def test_eval_executes_policy_not_oracle_and_rebuilds_feedback(self):
        ep = gen_episode(random.Random(43), horizon=5, rule_switch_prob=1.0)
        encoder = FakeEncoder()
        result = run_episode(encoder, self.make_agent(), ep, carry=True)
        self.assertEqual(result["n_steps"], 5)
        self.assertEqual(len(encoder.prompt_rows), 5)
        self.assertTrue(all(x == [1, 1, 0, 0, 0] for x in result["executed_actions"]))
        self.assertTrue(all(
            text.startswith("After this decision the active components are Alpha, Beta.")
            for text in result["observed_feedback"]
        ))
        for i in range(1, 5):
            self.assertEqual(encoder.prompt_rows[i]["state"],
                             "The active components are: Alpha, Beta.")
        self.assertTrue(any(
            result["observed_feedback"][i] != ep["steps"][i]["post_observation"]
            for i in range(5)
        ))

    def test_future_rule_not_inserted_into_current_prompt(self):
        ep = gen_episode(random.Random(12), horizon=4, rule_switch_prob=1.0)
        later = deepcopy(ep)
        later["steps"][1]["oracle_rule_before_decision"] *= -1
        enc1, enc2 = FakeEncoder(), FakeEncoder()
        # Both see exactly the same decision state at time 0.
        run_episode(enc1, self.make_agent(), ep, carry=True)
        run_episode(enc2, self.make_agent(), later, carry=True)
        self.assertEqual(enc1.prompt_rows[0], enc2.prompt_rows[0])

    def test_first_step_same_with_and_without_carry(self):
        ep = gen_episode(random.Random(5), horizon=4, rule_switch_prob=0.7)
        agent = self.make_agent()
        a = run_episode(FakeEncoder(), agent, ep, carry=True)
        b = run_episode(FakeEncoder(), agent, ep, carry=False)
        self.assertEqual(a["executed_actions"][0], b["executed_actions"][0])
        self.assertEqual(initial_state(ep), initial_state(ep))


if __name__ == "__main__":
    unittest.main()
