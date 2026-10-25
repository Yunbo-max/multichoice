"""Unit tests for the memory probe (linear probe + counterfactual summary)."""

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from memory_probe import counterfactual_summary, fit_probe, probe_predictions


class MemoryProbeTests(unittest.TestCase):
    def test_probe_recovers_separable_rule_encoding(self):
        torch.manual_seed(0)
        memories, labels = [], []
        for k in range(1, 7):
            for _ in range(20):
                vec = torch.zeros(6)
                vec[k - 1] = 1.0
                vec += torch.randn(6) * 0.01
                memories.append(vec)
                labels.append(k)
        model, accuracy = fit_probe(memories, labels, n_classes=6, steps=150)
        self.assertGreater(accuracy, 0.9)
        self.assertEqual(
            probe_predictions(model, [torch.tensor([1.0, 0, 0, 0, 0, 0])]), [1]
        )

    def test_probe_chance_on_uninformative_memory(self):
        torch.manual_seed(1)
        memories = [torch.randn(8) for _ in range(60)]
        labels = [(i % 6) + 1 for i in range(60)]
        _, accuracy = fit_probe(memories, labels, n_classes=6, steps=120)
        # uninformative random memory stays far below the separable case (0.9+)
        self.assertLess(accuracy, 0.5)

    def test_counterfactual_summary(self):
        same = {1: [1, 0, 0], 2: [1, 0, 0], 3: [1, 0, 0]}
        self.assertFalse(counterfactual_summary(same)["responds_to_history"])
        different = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
        self.assertTrue(counterfactual_summary(different)["responds_to_history"])
        self.assertEqual(counterfactual_summary(different)["n_distinct_predictions"], 3)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            fit_probe([], [], n_classes=6)
        with self.assertRaises(ValueError):
            fit_probe([torch.randn(4)], [0], n_classes=6)  # shift label 0 out of range
        with self.assertRaises(ValueError):
            fit_probe(
                [torch.randn(4), torch.randn(4)], [1], n_classes=6
            )  # length mismatch


if __name__ == "__main__":
    unittest.main()
