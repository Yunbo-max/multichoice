"""CPU regression tests for the BDH-CQ-inspired Recurrent SetJev pilot.

Run: python -m unittest tests.test_recurrent_set_decision -v
These tests do not load Qwen weights or contact an API.
"""
import itertools
import math
import unittest

import torch

from harness.recurrent_set_decision import (
    RecurrentSetConfig, RecurrentSetDecision, gold_set_nll,
    loss_over_steps, set_log_partition, set_map,
)


class RecurrentSetTests(unittest.TestCase):
    def test_partition_matches_exhaustive_enumeration_and_gradients(self):
        torch.manual_seed(0)
        scores = torch.randn(5, dtype=torch.float64, requires_grad=True)
        corr = torch.randn(6, dtype=torch.float64, requires_grad=True)
        counts = [1, 2, 4]
        values = []
        for choice in itertools.product([0, 1], repeat=5):
            if sum(choice) in counts:
                chosen = torch.tensor(choice, dtype=torch.bool)
                values.append(scores[chosen].sum() + corr[sum(choice)])
        oracle = torch.logsumexp(torch.stack(values), dim=0)
        exact = set_log_partition(scores, corr, counts)
        self.assertTrue(torch.allclose(oracle, exact, atol=1e-10))
        direct = torch.autograd.grad(oracle, (scores, corr), retain_graph=True)
        dynamic = torch.autograd.grad(exact, (scores, corr))
        for x, y in zip(direct, dynamic):
            self.assertTrue(torch.allclose(x, y, atol=1e-10))

    def test_zero_residual_is_conditional_bernoulli(self):
        scores = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float64)
        corr = scores.new_zeros(4)
        all_logs = []
        for binary in itertools.product([0, 1], repeat=3):
            p = torch.sigmoid(scores)
            event = torch.as_tensor(binary, dtype=torch.bool)
            all_logs.append(
                torch.where(event, p.log(), torch.log1p(-p)).sum()
            )
        z = torch.logsumexp(torch.stack(all_logs)[1:], dim=0)
        for binary in itertools.product([0, 1], repeat=3):
            if not any(binary):
                continue
            gold = torch.tensor(binary, dtype=torch.float64)
            expected = torch.stack(all_logs)[
                sum(1 << (2 - j) for j, v in enumerate(binary) if v)
            ] - z
            got = -gold_set_nll(scores, corr, gold, [1, 2, 3], normalize=False)
            self.assertTrue(torch.allclose(got, expected, atol=1e-10))

    def test_map_is_not_naive_count_mode(self):
        # A two-candidate distribution with large cardinality correction.
        scores = torch.tensor([3.0, 2.0])
        corr = torch.tensor([0.0, -10.0, 0.0])
        result = set_map(scores, corr, [1, 2])
        self.assertEqual(result["indices"], [0, 1])
        # Brute-force all permitted subsets to verify the exact decoder.
        choices = []
        for c in itertools.product([0, 1], repeat=2):
            if sum(c) in [1, 2]:
                choices.append((sum(float(scores[i]) for i in range(2) if c[i]) +
                                float(corr[sum(c)]), [i for i in range(2) if c[i]]))
        self.assertEqual(result["indices"], max(choices)[1])
        self.assertAlmostEqual(sum(result["count_probs"]), 1.0, places=6)

    def test_gradients_steps_masking_and_memory(self):
        torch.manual_seed(44)
        cfg = RecurrentSetConfig(context_dim=16, latent_dim=32, max_options=6)
        head = RecurrentSetDecision(cfg)
        context = torch.randn(2, 16)
        logits = torch.randn(2, 6)
        valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        gold = torch.tensor([[1., 0., 1., 0., 0., 0.],
                             [0., 1., 1., 0., 1., 0.]])
        history = head(context, logits, valid, steps=4)
        self.assertEqual(len(history), 5)
        self.assertTrue(torch.allclose(history[0]["scores"], logits.masked_fill(~valid, 0)))
        self.assertTrue(torch.allclose(history[4]["scores"], history[0]["scores"]))
        objective = loss_over_steps(history, gold, valid, [[1, 2, 3], [1, 2, 3, 4, 5]])
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertGreater(float(head.score_delta.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.count_delta.weight.grad.abs().sum()), 0)

        demo = torch.randn(2, 3, 16)
        with torch.no_grad():
            without = head(context, logits, valid, steps=2)[-1]["latent"]
            with_memory = head(context, logits, valid, steps=2, demonstrations=demo)[-1]["latent"]
        self.assertFalse(torch.allclose(without, with_memory))

    def test_padding_values_cannot_affect_real_candidates(self):
        torch.manual_seed(6)
        head = RecurrentSetDecision(RecurrentSetConfig(context_dim=12, latent_dim=16, max_options=5))
        with torch.no_grad():
            head.score_delta.weight.normal_(std=0.05)
        context = torch.randn(1, 12)
        logits = torch.randn(1, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        alternative = logits.clone()
        alternative[:, 3:] = 10_000
        a = head(context, logits, mask, steps=4)
        b = head(context, alternative, mask, steps=4)
        for r in range(5):
            self.assertTrue(torch.allclose(a[r]["scores"][:, :3], b[r]["scores"][:, :3]))
            self.assertTrue(torch.allclose(a[r]["correction"], b[r]["correction"]))

    def test_recursion_is_not_just_temperature_scaling(self):
        torch.manual_seed(12)
        head = RecurrentSetDecision(RecurrentSetConfig(context_dim=12, latent_dim=16, max_options=4))
        with torch.no_grad():
            head.score_delta.weight.normal_(std=0.05)
            head.count_delta.weight.normal_(std=0.05)
        context = torch.randn(2, 12)
        initial = torch.randn(2, 4)
        mask = torch.ones_like(initial, dtype=torch.bool)
        dynamic = head(context, initial, mask, steps=4, feedback="set")
        static = head(context, initial, mask, steps=4, feedback="static")
        self.assertFalse(torch.allclose(dynamic[-1]["scores"], static[-1]["scores"]))
        # No oracle targets or future environment states are passed to inference.
        self.assertEqual(set(dynamic[-1]), {"scores", "correction", "latent"})

    def test_bad_support_is_rejected(self):
        scores = torch.tensor([1.0, 0.0])
        correction = torch.zeros(3)
        with self.assertRaises(ValueError):
            gold_set_nll(scores, correction, torch.tensor([1., 0.]), [2])
        with self.assertRaises(ValueError):
            set_log_partition(scores, correction, [3])
        model = RecurrentSetDecision(RecurrentSetConfig(context_dim=4, latent_dim=8, max_options=3))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 4), torch.randn(1, 3), torch.tensor([[1, 0, 1]]))


if __name__ == "__main__":
    unittest.main()
