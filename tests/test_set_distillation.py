"""Small exact-enumeration oracles, plus optional real PyTorch gradient checks."""

import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from set_distillation import (
    TARGET_SCHEMA,
    allowed_counts,
    attach_set_targets,
    conditioned_bernoulli_stats,
    count_anchor_logits,
    input_digest,
    log_partitions,
    scalar_set_cross_entropy,
    structured_marginals,
    structured_prediction,
    validate_set_targets,
)


def row(k=4, mode="multi", empty=False):
    return {
        "id": "q1",
        "state": "context",
        "question": "Which are true?",
        "options": [{"key": str(i), "desc": f"option {i}"} for i in range(k)],
        "target": [1] + [0] * (k - 1),
        "selection_mode": mode,
        "allow_empty": empty,
    }


def enumerate_teacher(b, allowed):
    mass = {}
    for bits in itertools.product((0, 1), repeat=len(b)):
        if sum(bits) in allowed:
            mass[bits] = math.prod(v if bit else 1 - v for v, bit in zip(b, bits))
    total = sum(mass.values())
    q = [
        sum(value * bits[i] for bits, value in mass.items()) / total
        for i in range(len(b))
    ]
    rho = [
        sum(value for bits, value in mass.items() if sum(bits) == m) / total
        for m in range(len(b) + 1)
    ]
    return q, rho


def enumerate_student(scores, rho):
    sums = {
        bits: math.exp(sum(s * bit for s, bit in zip(scores, bits)))
        for bits in itertools.product((0, 1), repeat=len(scores))
    }
    z = [
        sum(value for bits, value in sums.items() if sum(bits) == m)
        for m in range(len(scores) + 1)
    ]
    return {bits: rho[sum(bits)] * value / z[sum(bits)] for bits, value in sums.items()}


class SetMathTests(unittest.TestCase):
    def assertVectorClose(self, a, b, places=10):
        self.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right, places=places)

    def test_conditioned_targets_against_enumeration(self):
        rng = random.Random(43)
        for k in range(2, 7):
            b = [rng.random() for _ in range(k)]
            for support in ([1], list(range(1, k + 1)), list(range(k + 1)), [0, k]):
                actual = conditioned_bernoulli_stats(b, support)
                expected = enumerate_teacher(b, support)
                self.assertVectorClose(actual[0], expected[0])
                self.assertVectorClose(actual[1], expected[1])

    def test_endpoint_probabilities_and_impossible_protocol(self):
        self.assertEqual(
            conditioned_bernoulli_stats([1, 0, 1], [2]),
            ([1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]),
        )
        with self.assertRaises(ValueError):
            conditioned_bernoulli_stats([0, 0, 0], [1, 2, 3])
        with self.assertRaises(ValueError):
            conditioned_bernoulli_stats([1, 1], [1])

    def test_very_rare_nonempty_event_is_stable(self):
        q, rho = conditioned_bernoulli_stats([1e-200, 2e-200], [1, 2])
        self.assertVectorClose(q, [1 / 3, 2 / 3])
        self.assertAlmostEqual(rho[1], 1)

    def test_protocol_never_uses_gold_count(self):
        question = row()
        self.assertEqual(allowed_counts(question), [1, 2, 3, 4])
        question["target"] = [1, 1, 1, 0]
        self.assertEqual(allowed_counts(question), [1, 2, 3, 4])
        question["selection_mode"] = "single"
        self.assertEqual(allowed_counts(question), [1])

    def test_single_choice_teacher_is_softmax_of_log_odds(self):
        b = [0.2, 0.5, 0.8]
        odds = [v / (1 - v) for v in b]
        q, rho = conditioned_bernoulli_stats(b, [1])
        self.assertVectorClose(q, [v / sum(odds) for v in odds])
        self.assertVectorClose(rho, [0, 1, 0, 0])

    def test_feasibility_stronger_than_expected_count_identity(self):
        # Expected count agrees, but q1=1 is impossible when the set is empty half the time.
        with self.assertRaises(ValueError):
            validate_set_targets([1, 0], [0.5, 0, 0.5], [0, 1, 2])

    def test_dp_and_exact_marginals(self):
        scores, rho = [0.4, -1.2, 2.1, 0.8], [0.0, 0.2, 0.5, 0.2, 0.1]
        distribution = enumerate_student(scores, rho)
        self.assertAlmostEqual(sum(distribution.values()), 1)
        expected = [
            sum(p * bits[i] for bits, p in distribution.items()) for i in range(4)
        ]
        self.assertVectorClose(structured_marginals(scores, rho), expected)
        self.assertAlmostEqual(sum(expected), sum(m * v for m, v in enumerate(rho)))

    def test_map_compares_count_normalized_subset_scores(self):
        # Count mode is m=1 (.6), but each singleton is .15: full set has .4.
        question = row()
        counts = [-1000, math.log(0.6), -1000, -1000, math.log(0.4)]
        result = structured_prediction([0, 0, 0, 0], counts, question)
        self.assertEqual(result["predicted_indices"], [0, 1, 2, 3])
        self.assertVectorClose(result["p"], [0.55] * 4)

    def test_map_matches_enumeration_and_shift_invariance(self):
        question = row()
        scores, counts = [0.4, -1.2, 2.1, 0.8], [0, -0.7, 1.2, 0.1, -0.2]
        result = structured_prediction(scores, counts, question)
        distribution = enumerate_student(scores, result["count_probs"])
        best = max(distribution, key=distribution.get)
        self.assertEqual(
            result["predicted_indices"], [i for i, bit in enumerate(best) if bit]
        )
        shifted = structured_prediction([s + 10000 for s in scores], counts, question)
        self.assertEqual(shifted["predicted_indices"], result["predicted_indices"])
        self.assertVectorClose(shifted["p"], result["p"], places=9)

    def test_map_log_probabilities_match_distribution_and_retain_rare_counts(self):
        question = row(3)
        scores, counts = [0.4, -1.2, 2.1], [0.0, -10000.0, 1.2, -0.2]
        result = structured_prediction(scores, counts, question)
        distribution = enumerate_student(scores, result["count_probs"])
        selected = tuple(int(i in result["predicted_indices"]) for i in range(3))
        self.assertAlmostEqual(
            result["predicted_set_log_probability"], math.log(distribution[selected])
        )
        self.assertIsNone(result["count_log_probs"][0])
        self.assertEqual(
            result["count_probs"][1], 0.0
        )  # Underflow is expected in probability space.
        self.assertLess(result["count_log_probs"][1], -10000)
        self.assertTrue(math.isfinite(result["count_log_probs"][1]))

    def test_single_choice_loss_reduces_to_ce(self):
        scores, q = [0.2, -0.5, 1.4], [0, 0, 1]
        actual = scalar_set_cross_entropy(
            scores, [2, -5, 1, 3], q, [0, 1, 0, 0], [1], False
        )
        self.assertAlmostEqual(
            actual, math.log(sum(math.exp(s) for s in scores)) - scores[2]
        )

    def test_bce_is_a_nested_special_case(self):
        scores = [0.2, -0.5, 1.4]
        q, rho = conditioned_bernoulli_stats([0.4, 0.6, 0.8], range(4))
        actual = scalar_set_cross_entropy(
            scores, log_partitions(scores), q, rho, range(4), False
        )
        expected = sum(math.log1p(math.exp(s)) - t * s for s, t in zip(scores, q))
        self.assertAlmostEqual(actual, expected)

    def test_same_membership_can_hide_different_count_distributions(self):
        q = [0.5] * 4
        rho_a, rho_b = [0, 0.5, 0, 0.5, 0], [0, 0, 1, 0, 0]
        counts = [0, -2, 3, -2, -2]
        loss_a = scalar_set_cross_entropy([0] * 4, counts, q, rho_a, [1, 2, 3, 4])
        loss_b = scalar_set_cross_entropy([0] * 4, counts, q, rho_b, [1, 2, 3, 4])
        self.assertNotAlmostEqual(loss_a, loss_b)

    def test_mixing_components_does_not_equal_product_of_means(self):
        qa, ra = conditioned_bernoulli_stats([1, 0, 0], [1, 2, 3])
        qb, rb = conditioned_bernoulli_stats([1, 1, 1], [1, 2, 3])
        mixed_q = [(a + b) / 2 for a, b in zip(qa, qb)]
        mixed_rho = [(a + b) / 2 for a, b in zip(ra, rb)]
        _, wrong_rho = conditioned_bernoulli_stats(mixed_q, [1, 2, 3])
        self.assertEqual(mixed_rho, [0, 0.5, 0, 0.5])
        self.assertVectorClose(wrong_rho, [0, 0.25, 0.5, 0.25])

    def test_loss_score_and_count_gradients_by_finite_differences(self):
        s, c = [0.2, -0.5, 1.4], [0.1, -0.7, 0.4, 0.3]
        q, rho = conditioned_bernoulli_stats([0.4, 0.6, 0.8], [1, 2, 3])
        expected_s = [(a - b) / 3 for a, b in zip(structured_marginals(s, rho), q)]
        predicted_rho = structured_prediction(s, c, row(3))["count_probs"]
        expected_c = [(a - b) / 3 for a, b in zip(predicted_rho, rho)]
        eps = 1e-5
        for vector, expected, is_score in (
            (s, expected_s, True),
            (c, expected_c, False),
        ):
            for i, target in enumerate(expected):
                plus, minus = list(vector), list(vector)
                plus[i] += eps
                minus[i] -= eps
                lp = scalar_set_cross_entropy(
                    plus if is_score else s, c if is_score else plus, q, rho, [1, 2, 3]
                )
                lm = scalar_set_cross_entropy(
                    minus if is_score else s,
                    c if is_score else minus,
                    q,
                    rho,
                    [1, 2, 3],
                )
                self.assertAlmostEqual((lp - lm) / (2 * eps), target, places=8)

    def test_hard_soft_mixture_is_coherent_and_digest_checked(self):
        question = row(3)
        q, rho = conditioned_bernoulli_stats([0.4, 0.6, 0.8], [1, 2, 3])
        target = {
            "schema": TARGET_SCHEMA,
            "q": q,
            "count_probs": rho,
            "option_keys": ["0", "1", "2"],
            "input_digest": input_digest(question),
            "allowed_counts": [1, 2, 3],
        }
        errors, stats = attach_set_targets([question], {"q1": target}, gold_weight=0.3)
        self.assertFalse(errors)
        self.assertEqual(stats["from_file"], 1)
        validate_set_targets(
            question["train_target"], question["train_count_probs"], [1, 2, 3]
        )
        question["question"] = "Different content"
        self.assertTrue(attach_set_targets([question], {"q1": target})[0])

    def test_count_anchor_zero_correction_is_score_implied_count(self):
        rng = random.Random(7)
        for k in range(2, 6):
            scores = [rng.uniform(-3, 3) for _ in range(k)]
            anchored = count_anchor_logits(scores, [0.0] * (k + 1))
            centered = [s - max(scores) for s in scores]
            for actual, expected in zip(anchored, log_partitions(centered)):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_structured_prediction_anchor_recovers_score_implied_count(self):
        rng = random.Random(11)
        question = row(4)
        scores = [rng.uniform(-2, 2) for _ in range(4)]
        item = structured_prediction(scores, [0.0] * 5, question, count_anchor=True)
        centered = [s - max(scores) for s in scores]
        logz = log_partitions(centered)
        support = allowed_counts(question)
        total = sum(math.exp(logz[m]) for m in support)
        expected = [
            math.exp(logz[m]) / total if m in support else 0.0 for m in range(5)
        ]
        for actual, target in zip(item["count_probs"], expected):
            self.assertAlmostEqual(actual, target, places=12)

    def test_count_anchor_is_shift_invariant(self):
        # max-centering makes the anchor invariant to a constant candidate-score shift.
        rng = random.Random(19)
        scores = [rng.uniform(-2, 2) for _ in range(4)]
        shift = 3.7
        anchored = count_anchor_logits(scores, [0.0] * 5)
        shifted = count_anchor_logits([s + shift for s in scores], [0.0] * 5)
        for a, b in zip(anchored, shifted):
            self.assertAlmostEqual(a, b, places=12)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchSetTests(unittest.TestCase):
    def test_autograd_matches_exact_moment_gradient(self):
        import torch
        from set_distillation import set_loss_terms

        question = row(3)
        q, rho = conditioned_bernoulli_stats([0.4, 0.6, 0.8], [1, 2, 3])
        question.update(train_target=q, train_count_probs=rho)
        scores = torch.tensor([[0.2, -0.5, 1.4]], requires_grad=True)
        counts = torch.tensor([[0.1, -0.7, 0.4, 0.3]], requires_grad=True)
        loss = set_loss_terms(scores, counts, [question])[0].sum()
        loss.backward()
        expected_s = [
            (a - b) / 3
            for a, b in zip(structured_marginals(scores[0].tolist(), rho), q)
        ]
        expected_rho = structured_prediction(
            scores[0].tolist(), counts[0].tolist(), question
        )["count_probs"]
        self.assertTrue(
            torch.allclose(scores.grad, torch.tensor([expected_s]), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                counts.grad,
                torch.tensor([[(a - b) / 3 for a, b in zip(expected_rho, rho)]]),
                atol=1e-6,
            )
        )

    def test_extreme_logits_and_mixed_protocol_batch_have_finite_gradients(self):
        import torch
        from set_distillation import set_loss_terms

        questions = [row(2, "single"), row(4), row(3, empty=True)]
        attach_set_targets(questions)
        scores = torch.tensor(
            [
                [10000.0, -10000.0, 2.0, 3.0],
                [-9.0, -8.0, -7.0, -6.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )
        counts = torch.zeros(3, 5, requires_grad=True)
        loss = set_loss_terms(scores, counts, questions)[0].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(counts.grad).all())

    def test_count_head_export_roundtrip_and_label_validation(self):
        import torch
        from train_multiselect import Readout, save_count_head, load_count_head

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(8, 10, bias=False)

            def get_output_embeddings(self):
                return self.head

        model = Readout(Base(), [1, 2, 3, 4], True, objective="set")
        with torch.no_grad():
            model.count_head.weight.fill_(0.7)
        with tempfile.TemporaryDirectory() as tmp:
            save_count_head(model, tmp)
            fresh = Readout(Base(), [1, 2, 3, 4], True, objective="set")
            load_count_head(fresh, tmp)
            self.assertTrue(
                torch.equal(model.count_head.weight, fresh.count_head.weight)
            )
            wrong = Readout(Base(), [2, 1, 3, 4], True, objective="set")
            with self.assertRaises(ValueError):
                load_count_head(wrong, tmp)

    def test_torch_count_anchor_matches_scalar(self):
        import torch
        from set_distillation import torch_count_anchor_logits

        rng = random.Random(13)
        for k in range(2, 7):
            scores = torch.tensor([rng.uniform(-3, 3) for _ in range(k)])
            head = torch.zeros(k + 1)
            actual = torch_count_anchor_logits(scores, head).tolist()
            expected = count_anchor_logits(scores.tolist(), [0.0] * (k + 1))
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e, places=4)

    def test_count_anchor_loss_has_finite_gradient_at_zero_head(self):
        import torch
        from set_distillation import set_loss_terms

        questions = [row(4), row(3, empty=True)]
        attach_set_targets(questions)
        scores = torch.randn(2, 5, requires_grad=True)
        counts = torch.zeros(2, 5, requires_grad=True)
        loss = set_loss_terms(scores, counts, questions, count_anchor=True)[0].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(counts.grad).all())


if __name__ == "__main__":
    unittest.main()
