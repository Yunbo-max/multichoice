"""Qwen episodic log builder tests using a fake encoder (no HF weights)."""
import unittest
import torch

from harness.build_episodic_features import encode_episode


class FakeEncoder:
    def __init__(self):
        self.observed_prompts = []

    def encode_demonstrations(self, demos):
        return torch.zeros(1, len(demos), 12)

    def encode_question(self, row):
        # The current prompt must never receive hidden labels or future feedback.
        self.observed_prompts.append(row)
        for key in ("target", "gold", "executed_set", "post_observation"):
            assert key not in row
        k = len(row["options"])
        return torch.zeros(1, 12), torch.zeros(1, k), torch.ones(1, k, dtype=torch.bool)

    def encode_option_meanings(self, row):
        return torch.ones(1, len(row["options"]), 12)

    def encode_observation(self, text):
        return torch.ones(1, 12)


def episode():
    return {
        "episode_id": "ep-01",
        "demonstrations": ["known pair: input->output"],
        "steps": [{
            "step_index": 0,
            "decision": {
                "state": "visible state",
                "question": "Select all valid actions",
                "options": [
                    {"key": "a", "desc": "inspect"},
                    {"key": "b", "desc": "repair"},
                ],
                "selection_mode": "multi",
                "allow_empty": False,
                "target": [1, 0],  # must be removed before encoding
            },
            "action_ids": ["inspect-v1", "repair-v1"],
            "gold": [1, 0],
            "executed_set": [0, 1],
            "post_observation": "only observed after action",
            "done": True,
        }],
    }


class EpisodicBuilderTests(unittest.TestCase):
    def test_target_whitelist_and_shapes(self):
        fake = FakeEncoder()
        encoded = encode_episode(episode(), fake)
        self.assertEqual(tuple(encoded["demonstrations"].shape), (1, 12))
        self.assertEqual(tuple(encoded["steps"][0]["native_logits"].shape), (2,))
        self.assertEqual(tuple(encoded["steps"][0]["option_features"].shape), (2, 12))
        self.assertEqual(len(fake.observed_prompts), 1)
        self.assertEqual(encoded["steps"][0]["executed_set"].tolist(), [0., 1.])
        self.assertEqual(encoded["steps"][0]["gold"].tolist(), [1., 0.])

    def test_reject_illegal_chronology_or_protocol(self):
        example = episode()
        example["steps"][0]["step_index"] = 1
        with self.assertRaises(ValueError):
            encode_episode(example, FakeEncoder())
        example = episode()
        example["steps"][0]["executed_set"] = [0, 0]
        with self.assertRaises(ValueError):
            encode_episode(example, FakeEncoder())


if __name__ == "__main__":
    unittest.main()
