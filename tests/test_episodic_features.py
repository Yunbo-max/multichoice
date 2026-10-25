"""Offline episodic feature-cache schema and BPTT training regressions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from harness.train_episodic_features import SCHEMA, read_episodes, train


def make_episode(episode_id: str, width: int = 5, dim: int = 12):
    demonstrations = torch.randn(1, dim)
    steps = []
    for index in range(3):
        steps.append(
            {
                "step_index": index,
                "context": torch.randn(dim),
                "native_logits": torch.randn(width),
                "gold": torch.tensor([1.0, 0, 1, 0, 0]),
                "executed_set": torch.tensor([0.0, 1, 1, 0, 0]),
                "allowed": [1, 2, 3, 4, 5],
                "option_ids": ["a", "b", "c", "d", "e"],
                "option_features": None,
                "post_observation": torch.randn(dim),
                "done": index == 2,
            }
        )
    return {
        "episode_id": episode_id,
        "demonstrations": demonstrations,
        "steps": steps,
    }


class EpisodicFeatureTests(unittest.TestCase):
    def test_chronological_training_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train_path = directory / "train.pt"
            dev_path = directory / "dev.pt"
            output = directory / "out"
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_identity": "fixed-backbone",
                    "episodes": [make_episode("train-a"), make_episode("train-b")],
                },
                train_path,
            )
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_identity": "fixed-backbone",
                    "episodes": [make_episode("dev-a")],
                },
                dev_path,
            )
            read_episodes(train_path)
            read_episodes(dev_path)
            args = SimpleNamespace(
                train_cache=train_path,
                dev_cache=dev_path,
                output=output,
                device="cpu",
                seed=43,
                epochs=1,
                inner_steps=2,
                latent_dim=16,
                max_options=6,
                accumulate_episodes=3,
                lr=1e-3,
                grad_clip=1.0,
                rule_weight=0.0,
            )
            # Incomplete accumulation group (two episodes, requested three).
            train(args)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())

    def test_invalid_step_chronology_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pt"
            episode = make_episode("bad")
            episode["steps"][1]["step_index"] = 3
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_identity": "fixed",
                    "episodes": [episode],
                },
                path,
            )
            with self.assertRaises(ValueError):
                read_episodes(path)

    def test_changing_action_menu_needs_semantics_at_all_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pt"
            episode = make_episode("bad")
            episode["steps"][1]["option_ids"][0] = "new-action"
            episode["steps"][1]["option_features"] = torch.randn(5, 12)
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_identity": "fixed",
                    "episodes": [episode],
                },
                path,
            )
            with self.assertRaises(ValueError):
                read_episodes(path)

    def test_duplicate_episode_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pt"
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_identity": "fixed",
                    "episodes": [make_episode("same"), make_episode("same")],
                },
                path,
            )
            with self.assertRaises(ValueError):
                read_episodes(path)


if __name__ == "__main__":
    unittest.main()
