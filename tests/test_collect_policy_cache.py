"""CPU test: collect_policy_cache output is accepted by read_episodes."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import gen_text_episodes_shift as gen
from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from collect_policy_cache import collect
from train_episodic_features import read_episodes


class FakeEncoder:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.hidden_size = 8
        self.k = len(gen.CELLS)

    def encode_demonstrations(self, texts):
        return torch.zeros(1, len(texts), self.hidden_size)

    def encode_question(self, row):
        return (
            torch.zeros(1, self.hidden_size),
            torch.zeros(1, self.k),
            torch.ones(1, self.k, dtype=torch.bool),
        )

    def encode_observation(self, text):
        return torch.zeros(1, self.hidden_size)

    def encode_option_meanings(self, row):
        return torch.zeros(1, self.k, self.hidden_size)


class CollectPolicyCacheTests(unittest.TestCase):
    def test_collect_output_matches_cache_schema(self):
        episode = gen.gen_episode(random.Random(0), horizon=3, rule_switch_prob=0.5)
        episode["episode_id"] = "ep-test-0"
        encoder = FakeEncoder()
        agent = EpisodicSetDecision(
            EpisodicSetConfig(
                context_dim=8, latent_dim=16, max_options=7, inner_steps=2
            )
        ).eval()
        cache_episodes = collect(gen, encoder, agent, [episode])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            torch.save(
                {
                    "schema": "episodic-set-feature-v1",
                    "model_identity": "fake",
                    "episodes": cache_episodes,
                },
                path,
            )
            loaded, _, _, _ = read_episodes(path)
            self.assertEqual(len(loaded["episodes"]), 1)
            # policy cache records the model's (empty-allowed) action, not oracle
            step = loaded["episodes"][0]["steps"][0]
            self.assertEqual(step["executed_set"].shape, (7,))
            self.assertEqual(step["gold"].shape, (7,))
            self.assertIn("rule", step)


if __name__ == "__main__":
    unittest.main()
