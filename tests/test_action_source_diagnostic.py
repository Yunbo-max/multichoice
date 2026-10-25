"""Expert-action evidence and paired online-memory diagnostic regressions.

Run:
 python -m unittest discover -s tests -p test_action_source_diagnostic.py -v

No Qwen checkpoint or network required. Uses the real evaluation routine and
a deterministic fake agent; proves expert trajectories carry information that
cannot be assumed available under an incorrect on-policy action.
"""
from __future__ import annotations

import itertools
import random
import unittest
import torch

from harness import gen_text_episodes_shift as gen
from harness.eval_memory_probe import collect_probe


class FakeEncoder:
    device = torch.device("cpu")

    def encode_demonstrations(self, text):
        return torch.zeros(1, len(text), 8)

    def encode_question(self, row):
        assert "oracle_rule_before_decision" not in row
        scores = torch.tensor([[10., 9., -10., -10., -10., -10., -10.]])
        return torch.zeros(1, 8), scores, torch.ones(1, 7, dtype=torch.bool)

    def encode_observation(self, text):
        return torch.zeros(1, 8)

    def encode_option_meanings(self, row):
        return torch.zeros(1, 7, 8)


class FakeAgent:
    def empty_memory(self, batch_size, *, device):
        return torch.zeros(batch_size, 8, device=device)

    def prime(self, memory, demos):
        return memory + 0.5

    def decide(self, memory, context, native_logits, mask):
        return [{
            "scores": native_logits,
            "correction": torch.zeros(1, 8),
            "latent": memory,
        }]

    def observe(self, memory, context, action, mask, obs,
                *, option_features=None, done=None):
        # The memory is intentionally dependent on the actually executed set.
        return memory + torch.nn.functional.pad(action, (0, 1))


class ActionSourceDiagnostics(unittest.TestCase):
    def test_expert_executed_action_identifies_rule_exactly(self):
        n = len(gen.CELLS)
        for pair in itertools.combinations(range(n), 2):
            current = [int(i in pair) for i in range(n)]
            successors = [gen.shift_set(current, k) for k in range(1, n)]
            self.assertEqual(len({tuple(v) for v in successors}), n - 1)
            for rule in range(1, n):
                target = gen.shift_set(current, rule)
                inferred = [k for k in range(1, n) if gen.shift_set(current, k) == target]
                self.assertEqual(inferred, [rule])

    def test_probe_state_changes_with_expert_vs_model_actions(self):
        # Pick an episode where the fake model's fixed action is wrong at t0.
        ep = None
        for seed in range(10, 90):
            candidate = gen.gen_episode(random.Random(seed), 5, 0.3)
            if candidate["steps"][0]["gold"] != [1, 1, 0, 0, 0, 0, 0]:
                ep = candidate
                break
        self.assertIsNotNone(ep)
        encoder, agent = FakeEncoder(), FakeAgent()
        expert_mem, expert_labels, expert_sw, expert_ids = collect_probe(
            gen, encoder, agent, [ep], action_source="expert"
        )
        policy_mem, policy_labels, policy_sw, policy_ids = collect_probe(
            gen, encoder, agent, [ep], action_source="policy"
        )
        self.assertEqual(expert_labels, policy_labels)
        self.assertEqual(expert_sw, policy_sw)
        self.assertEqual(expert_ids, policy_ids)
        self.assertEqual(len(expert_mem), len(ep["steps"]))
        self.assertTrue(torch.allclose(expert_mem[0], policy_mem[0]))
        self.assertFalse(torch.allclose(expert_mem[1], policy_mem[1]))

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            collect_probe(gen, FakeEncoder(), FakeAgent(), [], action_source="oracle-oracle")


if __name__ == "__main__":
    unittest.main()
