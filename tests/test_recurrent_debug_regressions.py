"""Regression checks for the 2026-10-24 recurrent SetJev audit.

No Qwen weights, network calls, cloud jobs, or GPU needed:
    python -m unittest discover -s tests -p 'test_recurrent_debug_regressions.py' -v
"""
import unittest

import torch
import torch.nn.functional as F

from harness.recurrent_set_decision import (
    RecurrentSetConfig,
    RecurrentSetDecision,
    gold_set_nll,
    loss_over_steps,
    revision_auxiliary_bce,
    revision_nll,
)
from harness.run_recurrent_set import initial_native_map, evaluate_model
from harness.set_revision import revision_target
from harness.toy_sequential_set import make_episodes, rollout


class RevisionMathAudit(unittest.TestCase):
    def test_same_set_zero_change_bce_grad_but_nonzero_joint_nll_grad(self):
        scores = torch.tensor([1., -0.5, 0.2], requires_grad=True)
        corr = torch.zeros(4)
        old = torch.tensor([1., 0., 1.])
        aux = revision_auxiliary_bce(scores, old, old)
        self.assertAlmostEqual(float(aux), 0.0, places=7)
        grad_aux = torch.autograd.grad(aux, scores, retain_graph=True)[0]
        self.assertTrue(torch.allclose(grad_aux, torch.zeros_like(scores)))
        joint = gold_set_nll(scores, corr, old, [1, 2, 3])
        grad_joint = torch.autograd.grad(joint, scores)[0]
        self.assertGreater(float(grad_joint.abs().sum()), 0.0)

    def test_revision_nll_equals_joint_nll_plus_auxiliary(self):
        scores = torch.tensor([0.3, -0.6, 0.2, -0.4])
        corr = torch.zeros(5)
        old = torch.tensor([1., 0., 1., 0.])
        gold = torch.tensor([1., 1., 0., 0.])
        for unchanged_weight in (0., 0.4, 1.):
            expected = gold_set_nll(scores, corr, gold, [1, 2, 3, 4])
            expected += revision_auxiliary_bce(
                scores, old, gold, unchanged_weight=unchanged_weight
            )
            observed = revision_nll(
                scores, corr, old, gold, [1, 2, 3, 4],
                unchanged_weight=unchanged_weight,
            )
            self.assertTrue(torch.allclose(expected, observed, atol=1e-7))
        full_bce = F.binary_cross_entropy_with_logits(scores, gold)
        self.assertTrue(torch.allclose(
            revision_nll(scores, corr, old, gold, [1, 2, 3, 4], unchanged_weight=1.),
            gold_set_nll(scores, corr, gold, [1, 2, 3, 4]) + full_bce,
        ))

    def test_revision_weight_changes_actual_training_loss_and_gradients(self):
        torch.manual_seed(1)
        model = RecurrentSetDecision(
            RecurrentSetConfig(context_dim=8, latent_dim=16, max_options=4)
        )
        context = torch.randn(2, 8)
        native = torch.tensor([[1., 1., -2., 0.5], [-2., 1., 1., -1.]])
        mask = torch.ones_like(native, dtype=torch.bool)
        gold = torch.tensor([[0., 1., 1., 0.], [0., 1., 0., 1.]])
        allowed = [[1, 2, 3, 4], [1, 2, 3, 4]]
        previous = initial_native_map(native, mask, allowed)
        with_rev = model(context, native, mask, steps=2)
        plain = loss_over_steps(with_rev, gold, mask, allowed, step_supervision="last")
        revised = loss_over_steps(
            with_rev, gold, mask, allowed, step_supervision="last",
            revision_previous=previous, revision_weight=1.0,
        )
        self.assertGreater(float(revised), float(plain))
        g0 = torch.autograd.grad(plain, model.score_delta.weight, retain_graph=True)[0]
        g1 = torch.autograd.grad(revised, model.score_delta.weight)[0]
        self.assertFalse(torch.allclose(g0, g1))
        with self.assertRaises(ValueError):
            loss_over_steps(with_rev, gold, mask, allowed, revision_weight=1.0)

    def test_invalid_fractional_labels_rejected(self):
        with self.assertRaises(ValueError):
            revision_target([1, 0.5], [0, 1])


class RepairMetricsAudit(unittest.TestCase):
    def test_same_question_initial_baseline_and_zero_step_repair(self):
        torch.manual_seed(4)
        model = RecurrentSetDecision(
            RecurrentSetConfig(context_dim=8, latent_dim=16, max_options=3)
        )
        native = torch.tensor([[4., -2., -3.], [-2., 4., -3.]])
        cache = {
            "ids": ["x", "y"], "context": torch.randn(2, 8),
            "native_logits": native, "mask": torch.ones(2, 3, dtype=torch.bool),
            "gold": torch.tensor([[0., 1., 0.], [0., 1., 0.]]),
            "allowed": [[1], [1]],
        }
        report = evaluate_model(model, cache, torch.device("cpu"), 0)
        self.assertAlmostEqual(report["initial_em"], 0.5)
        self.assertAlmostEqual(report["question_em"], 0.5)
        self.assertEqual(report["rescued_questions"], 0)
        self.assertEqual(report["damaged_questions"], 0)

    def test_toy_metrics_compare_same_state_r0_rR(self):
        model = RecurrentSetDecision(
            RecurrentSetConfig(context_dim=12, latent_dim=16, max_options=4)
        )
        eps = make_episodes(3, 17)
        report = rollout(model, eps, horizon=2, steps=0, use_demo=True)
        stats = report["revision_vs_initial_decision"]
        self.assertEqual(stats["repair_rate"], 0.)
        self.assertEqual(stats["broken_rate"], 0.)
        self.assertEqual(stats["n_trajectories"], 3)


if __name__ == "__main__":
    unittest.main()
