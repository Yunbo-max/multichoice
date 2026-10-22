"""Information-matched controls: exhaustive laws, derivatives and boundary cases."""
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from set_distillation import (
    attach_set_targets, conditioned_bernoulli_stats, scalar_moment_cross_entropy,
    scalar_set_cross_entropy, structured_log_marginals, structured_marginals,
    validate_set_loss_options,
)


def question(k, mode="multi", empty=False):
    return {"options": [{"key": str(i), "desc": str(i)} for i in range(k)],
            "target": [1] + [0] * (k - 1), "selection_mode": mode,
            "allow_empty": empty}


def count_distribution(logits, support):
    scale = max(logits[m] for m in support)
    weights = [math.exp(c - scale) if m in support else 0.0 for m, c in enumerate(logits)]
    return [w / sum(weights) for w in weights]


def enumerated_marginals(scores, counts, support):
    weights = {bits: math.exp(sum(s * b for s, b in zip(scores, bits)))
               for bits in itertools.product((0, 1), repeat=len(scores)) if sum(bits) in support}
    normalizers = {m: sum(w for bits, w in weights.items() if sum(bits) == m) for m in support}
    rho = count_distribution(counts, support)
    distribution = {bits: rho[sum(bits)] * w / normalizers[sum(bits)] for bits, w in weights.items()}
    return [sum(p * bits[i] for bits, p in distribution.items()) for i in range(len(scores))]


def finite_difference(function, values, epsilon=1e-5):
    answer = []
    for i in range(len(values)):
        plus, minus = list(values), list(values)
        plus[i] += epsilon
        minus[i] -= epsilon
        answer.append((function(plus) - function(minus)) / (2 * epsilon))
    return answer


class MomentControlTests(unittest.TestCase):
    def test_induced_logs_agree_with_exhaustive_subsets(self):
        rng = random.Random(123)
        for k in range(2, 7):
            scores = [rng.uniform(-3, 3) for _ in range(k)]
            counts = [rng.uniform(-3, 3) for _ in range(k + 1)]
            for support in ([1], [0, k], list(range(1, k + 1)), list(range(k + 1))):
                lp, ln = structured_log_marginals(scores, counts, support)
                expected = enumerated_marginals(scores, counts, support)
                for a, b, p in zip(lp, ln, expected):
                    self.assertAlmostEqual(math.exp(a), p, places=11)
                    self.assertAlmostEqual(math.exp(b), 1 - p, places=11)

    def test_both_control_values_match_direct_formulas(self):
        scores, counts = [.2, -.5, 1.4], [.1, -.7, .4, .3]
        support = [1, 2, 3]
        q, target_rho = conditioned_bernoulli_stats([.4, .6, .8], support)
        student_rho = count_distribution(counts, support)
        for family in ("independent_marginal", "induced_marginal"):
            p = ([1 / (1 + math.exp(-s)) for s in scores] if family == "independent_marginal"
                 else enumerated_marginals(scores, counts, support))
            for weight in (0.0, .4, 1.0, 2.0):
                expected = -sum(t * math.log(v) + (1 - t) * math.log1p(-v) for t, v in zip(q, p))
                expected -= weight * sum(target_rho[m] * math.log(student_rho[m]) for m in support)
                actual = scalar_moment_cross_entropy(scores, counts, q, target_rho, support,
                                                     family, weight)
                self.assertAlmostEqual(actual, expected / len(scores), places=12)

    def test_single_choice_uses_softmax_marginals_but_bce_is_not_categorical_ce(self):
        scores, counts, q, rho = [.2, -.5, 1.4], [3., 9., -4., 2.], [0, 0, 1], [0, 1, 0, 0]
        lp, ln = structured_log_marginals(scores, counts, [1])
        logz = math.log(sum(math.exp(s) for s in scores))
        for actual, score in zip(lp, scores):
            self.assertAlmostEqual(actual, score - logz)
        expected = -(lp[2] + ln[0] + ln[1]) / 3
        actual = scalar_moment_cross_entropy(scores, counts, q, rho, [1])
        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, scalar_set_cross_entropy(scores, counts, q, rho, [1]))

    def test_realizable_moments_are_stationary_for_both_set_ce_and_induced_control(self):
        scores, counts, support = [.2, -.5, 1.4], [.1, -.7, .4, .3], [1, 2, 3]
        rho = count_distribution(counts, support)
        q = enumerated_marginals(scores, counts, support)
        for objective in (scalar_moment_cross_entropy, scalar_set_cross_entropy):
            for varying_scores in (True, False):
                derivatives = finite_difference(
                    lambda v: objective(v if varying_scores else scores,
                                        counts if varying_scores else v, q, rho, support),
                    scores if varying_scores else counts)
                for derivative in derivatives:
                    self.assertAlmostEqual(derivative, 0, places=8)

    def test_induced_marginal_term_updates_count_head_even_with_zero_count_weight(self):
        scores, counts, support = [.2, -.5, 1.4], [.1, -.7, .4, .3], [1, 2, 3]
        q, rho = conditioned_bernoulli_stats([.4, .6, .8], support)
        derivatives = {}
        for family in ("independent_marginal", "induced_marginal"):
            derivatives[family] = finite_difference(
                lambda c: scalar_moment_cross_entropy(scores, c, q, rho, support, family, 0), counts)
        self.assertTrue(any(abs(v) > 1e-4 for v in derivatives["induced_marginal"]))
        self.assertTrue(all(abs(v) < 1e-10 for v in derivatives["independent_marginal"]))

    def test_only_induced_control_is_invariant_to_common_score_shift(self):
        scores, counts = [.2, -.5, 1.4], [.1, -.7, .4, .3]
        q, rho = conditioned_bernoulli_stats([.4, .6, .8], [1, 2, 3])
        for family in ("independent_marginal", "induced_marginal"):
            original = scalar_moment_cross_entropy(scores, counts, q, rho, [1, 2, 3], family)
            shifted = scalar_moment_cross_entropy([s + 10000 for s in scores], counts, q, rho,
                                                  [1, 2, 3], family)
            if family == "induced_marginal":
                self.assertAlmostEqual(original, shifted, places=10)
            else:
                self.assertGreater(shifted, original + 100)

    def test_extreme_scores_preserve_large_log_loss_without_clipping(self):
        lp, ln = structured_log_marginals([10000., -10000.], [0., 0., 0.], [1])
        self.assertEqual(lp, [0., -20000.])
        self.assertEqual(ln, [-20000., 0.])
        self.assertEqual(scalar_moment_cross_entropy([10000., -10000.], [0., 0., 0.],
                                                    [0., 1.], [0., 1., 0.], [1]), 20000.)

    def test_deterministic_count_protocol_endpoints_do_not_multiply_zero_by_inf(self):
        for support, q, rho in (([0], [0, 0], [1, 0, 0]), ([2], [1, 1], [0, 0, 1])):
            actual = scalar_moment_cross_entropy([10000., -10000.], [0., 0., 0.], q, rho, support)
            self.assertEqual(actual, 0)

    def test_invalid_loss_option_combinations_are_rejected_without_loading_torch(self):
        for family, weight, target in (("unknown", 1, "teacher"), ("set_ce", .5, "teacher"),
                                      ("induced_marginal", 1, "gold"),
                                      ("independent_marginal", 1, "gold"),
                                      ("induced_marginal", -1, "teacher"),
                                      ("induced_marginal", math.nan, "teacher"),
                                      ("induced_marginal", True, "teacher")):
            with self.assertRaises(ValueError):
                validate_set_loss_options(family, weight, target)
        validate_set_loss_options("set_ce", 1, "gold")  # Backward compatibility.


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchMomentControlTests(unittest.TestCase):
    def test_both_losses_and_all_gradients_match_scalar_finite_differences(self):
        import torch
        from set_distillation import set_loss_terms
        s, c, support = [.2, -.5, 1.4], [.1, -.7, .4, .3], [1, 2, 3]
        q, rho = conditioned_bernoulli_stats([.4, .6, .8], support)
        item = question(3)
        item.update(train_target=q, train_count_probs=rho)
        for family in ("independent_marginal", "induced_marginal"):
            for weight in (0., .7):
                scores = torch.tensor([s], requires_grad=True)
                counts = torch.tensor([c], requires_grad=True)
                loss = set_loss_terms(scores, counts, [item], loss_family=family,
                                      count_weight=weight)[0].sum()
                loss.backward()
                scalar = lambda a, b: scalar_moment_cross_entropy(a, b, q, rho, support, family, weight)
                self.assertAlmostEqual(loss.item(), scalar(s, c), places=6)
                ds = finite_difference(lambda a: scalar(a, c), s)
                dc = finite_difference(lambda b: scalar(s, b), c)
                self.assertTrue(torch.allclose(scores.grad, torch.tensor([ds]), atol=2e-6))
                self.assertTrue(torch.allclose(counts.grad, torch.tensor([dc]), atol=2e-6))

    def test_mixed_protocol_and_extreme_score_gradients_are_finite_and_not_clipped(self):
        import torch
        from set_distillation import set_loss_terms
        items = [question(2, "single"), question(4), question(3, empty=True)]
        items[0]["target"] = [0, 1]
        attach_set_targets(items)
        for family in ("independent_marginal", "induced_marginal"):
            scores = torch.tensor([[10000., -10000., 2., 3.], [-9., -8., -7., -6.],
                                   [0., 0., 0., 0.]], requires_grad=True)
            counts = torch.zeros(3, 5, requires_grad=True)
            losses = set_loss_terms(scores, counts, items, loss_family=family)[0]
            losses.sum().backward()
            self.assertTrue(torch.isfinite(losses).all())
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertTrue(torch.isfinite(counts.grad).all())
            self.assertGreater(abs(scores.grad[0, 0].item()), .4)
            if family == "induced_marginal":
                self.assertAlmostEqual(losses[0].item(), 20000.)

    def test_log_dp_backward_matches_exhaustive_probability_marginals(self):
        import torch
        from set_distillation import torch_structured_log_marginals
        for support in ([1], [0, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]):
            s = torch.tensor([.4, -.3, 1.7, -.8], dtype=torch.double, requires_grad=True)
            c = torch.tensor([.2, -.5, 1.4, .7, -.2], dtype=torch.double, requires_grad=True)
            lc = torch.log_softmax(c[support], 0)
            lp, ln = torch_structured_log_marginals(s, lc, support)
            subsets = [bits for bits in itertools.product((0, 1), repeat=4) if sum(bits) in support]
            potentials = torch.stack([sum(s[i] * bits[i] for i in range(4)) for bits in subsets])
            masses = []
            for j, bits in enumerate(subsets):
                count = sum(bits)
                same_count = [i for i, b in enumerate(subsets) if sum(b) == count]
                masses.append(torch.exp(lc[support.index(count)] + potentials[j]
                                        - torch.logsumexp(potentials[same_count], 0)))
            probabilities = torch.stack([sum(p * bits[i] for bits, p in zip(subsets, masses))
                                         for i in range(4)])
            self.assertTrue(torch.allclose(lp.exp(), probabilities, atol=1e-12))
            self.assertTrue(torch.allclose(ln.exp(), 1 - probabilities, atol=1e-12))
            probe = torch.tensor([.7, .2, .8, .4], dtype=torch.double)
            actual = -(probe * lp + (1 - probe) * ln).sum()
            expected = -(probe * probabilities.log() + (1 - probe) * torch.log1p(-probabilities)).sum()
            actual_grads = torch.autograd.grad(actual, (s, c), retain_graph=True)
            expected_grads = torch.autograd.grad(expected, (s, c))
            for a, b in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(a, b, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
