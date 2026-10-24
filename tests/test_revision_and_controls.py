"""Tests for revision_nll and the non-recurrent mlp control."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from recurrent_set_decision import (
    RecurrentSetConfig,
    RecurrentSetDecision,
    revision_nll,
    revision_auxiliary_bce,
    gold_set_nll,
)

ALLOWED = [1, 2, 3, 4]


class RevisionNllTests(unittest.TestCase):
    def test_changed_only_auxiliary_is_lower_than_full_bce_augmented_nll(self):
        torch.manual_seed(0)
        scores = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)
        correction = torch.zeros(5)
        prev = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        new = torch.tensor([1, 1, 1, 0], dtype=torch.float32)  # only option1 changed
        pure = revision_nll(
            scores, correction, prev, new, ALLOWED, unchanged_weight=0.0
        )
        full = revision_nll(
            scores, correction, prev, new, ALLOWED, unchanged_weight=1.0
        )
        self.assertLess(float(pure), float(full))

    def test_no_change_has_zero_auxiliary_but_joint_nll_remains(self):
        scores = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)
        correction = torch.zeros(5)
        prev = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        auxiliary = revision_auxiliary_bce(scores, prev, prev)
        self.assertEqual(float(auxiliary), 0.0)
        combined = revision_nll(
            scores, correction, prev, prev, ALLOWED, unchanged_weight=0.0
        )
        joint = gold_set_nll(scores, correction, prev, ALLOWED)
        self.assertTrue(torch.allclose(combined, joint, atol=1e-6))

    def test_rejects_invalid(self):
        scores = torch.randn(4)
        correction = torch.randn(5)
        with self.assertRaises(ValueError):
            revision_nll(scores, correction, torch.zeros(3), torch.zeros(4), ALLOWED)
        with self.assertRaises(ValueError):
            revision_nll(
                scores,
                correction,
                torch.zeros(4),
                torch.zeros(4),
                ALLOWED,
                unchanged_weight=1.5,
            )
        with self.assertRaises(ValueError):
            revision_nll(
                scores,
                correction,
                torch.zeros(4),
                torch.tensor([2.0, 0, 0, 0]),
                ALLOWED,
            )


class MlpControlTests(unittest.TestCase):
    def test_mlp_runs_and_is_finite(self):
        torch.manual_seed(0)
        cfg = RecurrentSetConfig(
            context_dim=8, latent_dim=32, max_options=4, feedback="mlp", max_steps=4
        )
        model = RecurrentSetDecision(cfg)
        context = torch.randn(2, 8)
        native = torch.randn(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = model(context, native, mask, steps=4)[-1]
        self.assertTrue(torch.isfinite(out["scores"]).all())

    def test_mlp_requires_steps_le_max_steps(self):
        cfg = RecurrentSetConfig(
            context_dim=8, latent_dim=32, max_options=4, feedback="mlp", max_steps=4
        )
        model = RecurrentSetDecision(cfg)
        context = torch.randn(1, 8)
        native = torch.randn(1, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            model(context, native, mask, steps=5)

    def test_mlp_has_distinct_per_step_layers(self):
        cfg = RecurrentSetConfig(
            context_dim=8, latent_dim=32, max_options=4, feedback="mlp", max_steps=4
        )
        model = RecurrentSetDecision(cfg)
        self.assertEqual(len(model.state_stack), 4)
        # distinct layer parameters (no weight sharing)
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(
                    torch.equal(
                        model.state_stack[i][0].weight, model.state_stack[j][0].weight
                    )
                )


if __name__ == "__main__":
    unittest.main()
