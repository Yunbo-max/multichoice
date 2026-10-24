"""Causal-memory and BPTT regression tests for episodic SetJev.

Run from repository root:
 python -m unittest discover -s tests -p 'test_episodic_set_decision.py' -v
"""
import random
import unittest

import torch

from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from harness.train_episodic_set import (
    context_dim,
    demo_features,
    native_option_logits,
    observation_features,
    one_episode_batch,
    sample_batch,
    shift_sets,
)
from harness.recurrent_set_decision import gold_set_nll


class PersistentMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.width = 6
        self.config = EpisodicSetConfig(
            context_dim=context_dim(self.width), latent_dim=16,
            max_options=self.width, inner_steps=2,
        )
        self.model = EpisodicSetDecision(self.config)
        self.x = torch.rand(3, self.width).round().float()
        self.native = native_option_logits(self.x)
        self.query = observation_features(self.x, 0, 4)
        self.mask = torch.ones_like(self.native, dtype=torch.bool)
        self.init = self.model.empty_memory(3, device="cpu")

    def test_reset_only_changes_specified_batch_members(self):
        memory = torch.randn(3, 16)
        saved = memory.clone()
        flagged = torch.tensor([True, False, True])
        reset = self.model.reset(memory, flagged)
        self.assertTrue(torch.equal(reset[0], torch.zeros(16)))
        self.assertTrue(torch.equal(reset[2], torch.zeros(16)))
        self.assertTrue(torch.equal(reset[1], memory[1]))
        self.assertTrue(torch.equal(memory, saved))
        with self.assertRaises(ValueError):
            self.model.reset(memory, torch.tensor([1, 0, 1]))

    def test_priming_changes_persistent_memory_and_current_workspace(self):
        x = torch.zeros(3, self.width)
        x[:, 1] = 1
        demonstration = demo_features(
            x, torch.roll(x, 1, dims=1)
        ).unsqueeze(1)
        primed = self.model.prime(self.init, demonstration)
        self.assertFalse(torch.allclose(primed, self.init))
        unprimed_state = self.model.decide(
            self.init, self.query, self.native, self.mask
        )[1]["latent"]
        primed_state = self.model.decide(
            primed, self.query, self.native, self.mask
        )[1]["latent"]
        self.assertFalse(torch.allclose(unprimed_state, primed_state))

    def test_decide_does_not_mutate_memory(self):
        memory = torch.randn_like(self.init)
        saved = memory.clone()
        self.model.decide(memory, self.query, self.native, self.mask)
        self.assertTrue(torch.equal(memory, saved))

    def test_post_action_observation_cannot_change_past_prediction(self):
        prediction = self.model.decide(
            self.init, self.query, self.native, self.mask
        )[-1]
        memory_a = self.model.observe(
            self.init, self.query, self.x, self.mask, self.query
        )
        memory_b = self.model.observe(
            self.init, self.query, self.x, self.mask, self.query + 0.3
        )
        same_prediction = self.model.decide(
            self.init, self.query, self.native, self.mask
        )[-1]
        self.assertTrue(torch.equal(prediction["scores"], same_prediction["scores"]))
        self.assertFalse(torch.allclose(memory_a, memory_b))

    def test_bptt_reaches_memory_ingestion_and_environment_transition(self):
        known_input = torch.zeros(3, self.width)
        known_input[:, 0] = 1
        demo = demo_features(
            known_input, torch.roll(known_input, 1, dims=1)
        ).unsqueeze(1)
        # Zero-init final heads mask upstream gradients at the first step.
        # Nonzero weights isolate whether the BPTT path is structurally sound.
        with torch.no_grad():
            self.model.decision.score_delta.weight.normal_(std=0.05)
            self.model.decision.count_delta.weight.normal_(std=0.05)
        memory0 = self.model.prime(self.init, demo)
        action = torch.tensor([[1., 0, 0, 1, 0, 0]] * 3)
        after = observation_features(action, 1, 4)
        memory1 = self.model.observe(
            memory0, self.query, action, self.mask, after
        )
        prediction = self.model.decide(
            memory1, after, native_option_logits(action), self.mask
        )[-1]
        target = torch.roll(action, 1, dims=1)
        loss = torch.stack([
            gold_set_nll(
                prediction["scores"][i],
                prediction["correction"][i, : self.width + 1],
                target[i], list(range(1, self.width + 1)),
            )
            for i in range(3)
        ]).mean()
        loss.backward()
        self.assertGreater(
            float(self.model.transition_context.weight.grad.abs().sum()), 0
        )
        self.assertGreater(
            float(self.model.decision.memory_cell.weight_ih.grad.abs().sum()), 0
        )

    def test_action_validation_and_terminal_resets(self):
        invalid = self.x.clone()
        invalid[0, 0] = 0.5
        with self.assertRaises(ValueError):
            self.model.observe(self.init, self.query, invalid, self.mask)
        mask = self.mask.clone()
        mask[0, -1] = False
        invalid = self.x.clone()
        invalid[0, -1] = 1
        with self.assertRaises(ValueError):
            self.model.observe(self.init, self.query, invalid, mask)
        memory = self.model.observe(
            self.init, self.query, self.x, self.mask, self.query,
            done=torch.tensor([True, False, True]),
        ).detach()
        self.assertEqual(float(memory[0].abs().sum()), 0)
        self.assertEqual(float(memory[2].abs().sum()), 0)
        self.assertGreater(float(memory[1].abs().sum()), 0)

    def test_semantic_actions_are_invariant_to_option_order(self):
        config = EpisodicSetConfig(
            context_dim=context_dim(self.width), latent_dim=16,
            max_options=self.width, inner_steps=2,
            require_semantic_actions=True,
        )
        model = EpisodicSetDecision(config)
        features = torch.randn(3, self.width, context_dim(self.width))
        selected = model.observe(
            self.init, self.query, self.x, self.mask, self.query,
            option_features=features,
        )
        order = torch.tensor([2, 4, 1, 0, 5, 3])
        reordered = model.observe(
            self.init, self.query, self.x[:, order],
            self.mask[:, order], self.query,
            option_features=features[:, order],
        )
        self.assertTrue(torch.allclose(selected, reordered, atol=1e-6))
        with self.assertRaises(ValueError):
            model.observe(self.init, self.query, self.x, self.mask, self.query)

    def test_rule_switch_and_on_policy_transitions_are_finite(self):
        rng = random.Random(43)
        state, shift, _ = sample_batch(
            4, self.width, rng, torch.device("cpu")
        )
        target = shift_sets(state, shift)
        self.assertEqual(tuple(target.shape), (4, self.width))
        result = one_episode_batch(
            self.model, batch=4, width=self.width, horizon=4,
            rng=rng, device=torch.device("cpu"),
            teacher_force_prob=0.5, train=True, rule_switch_prob=1.0,
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreaterEqual(result["step_em"], 0)
        self.assertLessEqual(result["step_em"], 1)


if __name__ == "__main__":
    unittest.main()
