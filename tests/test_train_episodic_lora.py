"""CPU test: on-policy replay (C experiment) produces a finite loss and gradients."""

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import gen_text_episodes_shift as gen
from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from train_episodic_lora import replay, _initial


class FakeEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.hidden_size = 8

    def encode_question(self, row):
        k = len(row["options"])
        return (
            torch.zeros(1, self.hidden_size),
            torch.zeros(1, k),
            torch.ones(1, k, dtype=torch.bool),
        )

    def encode_observation(self, text):
        return torch.zeros(1, self.hidden_size)

    def encode_demonstrations(self, observations):
        if not observations:
            return torch.zeros(1, 0, self.hidden_size)
        return torch.zeros(1, len(observations), self.hidden_size)

    def encode_option_meanings(self, row):
        return torch.zeros(1, len(row["options"]), self.hidden_size)


class OnPolicyReplayTests(unittest.TestCase):
    def test_replay_finite_loss_and_gradients(self):
        episode = gen.gen_episode(random.Random(0), horizon=4, rule_switch_prob=0.5)
        episode["episode_id"] = "ep-0"
        agent = EpisodicSetDecision(
            EpisodicSetConfig(
                context_dim=8, latent_dim=16, max_options=7, inner_steps=2
            )
        )
        loss = replay(gen, FakeEncoder(), agent, episode)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # the decision core (score/count heads) must receive gradients
        grads = [p.grad for p in agent.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)

    def test_initial_state_parse(self):
        episode = gen.gen_episode(random.Random(1), horizon=2, rule_switch_prob=0.0)
        active = _initial(gen, episode)
        self.assertEqual(sum(active), 2)
        self.assertEqual(len(active), 7)


if __name__ == "__main__":
    unittest.main()
