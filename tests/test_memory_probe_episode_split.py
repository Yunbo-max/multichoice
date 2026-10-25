"""Regression tests: no episodic memory leakage across probe train/test.

Run:
    python -m unittest discover -s tests -p test_memory_probe_episode_split.py -v
"""
from __future__ import annotations

import unittest
import torch

from harness.memory_probe import fit_probe_split, probe_predictions
import harness.eval_memory_probe as eval_probe


class EpisodeSplitTests(unittest.TestCase):
    def _make_rows(self):
        generator = torch.Generator().manual_seed(172)
        memories, labels, episodes = [], [], []
        for ep in range(30):
            # Each episode contains correlated observations, and sometimes a
            # change in rule midway. No episode is allowed in both groups.
            base = torch.randn(12, generator=generator)
            for step in range(4):
                memories.append(base + 0.01 * torch.randn(12, generator=generator))
                labels.append(1 + ((ep + int(step >= 2)) % 6))
                episodes.append(f"episode-{ep}")
        return memories, labels, episodes

    def test_train_and_test_never_share_an_episode(self):
        memories, labels, groups = self._make_rows()
        _, tr, te, split = fit_probe_split(
            memories, labels, n_classes=6, groups=groups,
            return_indices=True, steps=10, seed=7,
        )
        self.assertEqual(split["split_unit"], "episode")
        train_ids = {groups[i] for i in split["train_indices"]}
        test_ids = {groups[i] for i in split["test_indices"]}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids, set(groups))
        self.assertEqual(len(split["train_indices"]) + len(split["test_indices"]), len(groups))
        self.assertEqual(split["test_episodes"], len(test_ids))
        self.assertTrue(0 <= tr <= 1)
        self.assertTrue(0 <= te <= 1)

    def test_split_is_deterministic_and_returns_fixed_holdout(self):
        data = self._make_rows()
        _, _, _, s1 = fit_probe_split(*data[:2], groups=data[2],
                                      steps=3, seed=22, return_indices=True)
        _, _, _, s2 = fit_probe_split(*data[:2], groups=data[2],
                                      steps=3, seed=22, return_indices=True)
        self.assertEqual(s1["train_indices"], s2["train_indices"])
        self.assertEqual(s1["test_indices"], s2["test_indices"])

    def test_invalid_grouping_is_rejected(self):
        memories, labels, groups = self._make_rows()
        with self.assertRaises(ValueError):
            fit_probe_split(memories, labels, groups=groups[:3], steps=2)
        with self.assertRaises(ValueError):
            fit_probe_split(memories, labels, groups=["one"] * len(memories), steps=2)
        with self.assertRaises(ValueError):
            fit_probe_split(memories, labels, groups=[""] * len(memories), steps=2)

    def test_evaluator_import_and_holdout_probe_predictions(self):
        self.assertTrue(callable(eval_probe.collect_probe))
        self.assertTrue(callable(eval_probe.counterfactual))
        memories, labels, groups = self._make_rows()
        probe, _, accuracy, split = fit_probe_split(
            memories, labels, groups=groups, steps=15, return_indices=True
        )
        test = split["test_indices"]
        predictions = probe_predictions(probe, [memories[i] for i in test])
        manual = sum(int(p == labels[i]) for p, i in zip(predictions, test)) / len(test)
        self.assertAlmostEqual(accuracy, manual)


if __name__ == "__main__":
    unittest.main()
