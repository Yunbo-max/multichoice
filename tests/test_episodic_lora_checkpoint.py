"""CPU regression tests for strict episodic LoRA checkpoint and on-policy DEV.

Run:
 python -m unittest discover -s tests -p test_episodic_lora_checkpoint.py -v
"""
from __future__ import annotations

import inspect
import random
import unittest

import torch
from torch import nn

from harness.episodic_lora_checkpoint import (
    check_lora_gradients,
    load_lora_tensors_verified,
    lora_tensors,
)
from harness.train_episodic_lora import TrainableQwenEncoder, evaluate_onpolicy_dev
from harness.episodic_set_decision import EpisodicSetDecision, EpisodicSetConfig
from harness import gen_text_episodes_shift as gen


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


class LoRACheckpointTests(unittest.TestCase):
    def test_all_lora_keys_restored_exactly(self):
        torch.manual_seed(1)
        model = TinyLoRA()
        saved = lora_tensors(model)
        self.assertEqual(set(saved), {"lora_A.weight", "lora_B.weight"})
        with torch.no_grad():
            model.lora_A.weight.fill_(7)
            model.lora_B.weight.fill_(9)
        n = load_lora_tensors_verified(model, saved)
        self.assertEqual(n, 2)
        self.assertTrue(torch.equal(model.lora_A.weight.detach(), saved["lora_A.weight"]))
        self.assertTrue(torch.equal(model.lora_B.weight.detach(), saved["lora_B.weight"]))

    def test_missing_extra_wrong_shapes_are_rejected(self):
        model = TinyLoRA()
        src = lora_tensors(model)
        corrupted = dict(src)
        del corrupted["lora_B.weight"]
        with self.assertRaises(ValueError):
            load_lora_tensors_verified(model, corrupted)
        corrupted = dict(src, **{"lora_fake.weight": torch.randn(2, 4)})
        with self.assertRaises(ValueError):
            load_lora_tensors_verified(model, corrupted)
        corrupted = dict(src)
        corrupted["lora_A.weight"] = torch.randn(1, 4)
        with self.assertRaises(ValueError):
            load_lora_tensors_verified(model, corrupted)

    def test_lora_gradient_guard(self):
        torch.manual_seed(2)
        model = TinyLoRA()
        for name, p in model.named_parameters():
            if "lora" not in name:
                p.requires_grad_(False)
        with self.assertRaises(RuntimeError):
            check_lora_gradients(model)
        model(torch.randn(3, 4)).square().mean().backward()
        diagnostics = check_lora_gradients(model)
        self.assertEqual(diagnostics["trainable_lora_tensors"], 2)
        self.assertGreater(diagnostics["lora_gradient_l2"], 0)


class FakeEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = nn.Linear(1, 1)

    def encode_question(self, row):
        k = len(row["options"])
        return torch.zeros(1, 8), torch.zeros(1, k), torch.ones(1, k, dtype=torch.bool)

    def encode_demonstrations(self, data):
        return torch.zeros(1, len(data), 8)

    def encode_option_meanings(self, row):
        return torch.zeros(1, len(row["options"]), 8)

    def encode_observation(self, text):
        return torch.zeros(1, 8)


class LoaderIntegrityTests(unittest.TestCase):
    def test_qwen_base_is_loaded_exactly_once(self):
        implementation = inspect.getsource(TrainableQwenEncoder.__init__)
        self.assertEqual(
            implementation.count("self.model = cls.from_pretrained("), 1
        )


class DevSelectionTests(unittest.TestCase):
    def test_dev_reads_policy_actions_and_returns_bounded_em(self):
        episodes = [gen.gen_episode(random.Random(seed), 4, 0.25) for seed in range(3)]
        model = EpisodicSetDecision(
            EpisodicSetConfig(context_dim=8, latent_dim=16, max_options=7, inner_steps=2)
        )
        result = evaluate_onpolicy_dev(gen, FakeEncoder(), model, episodes)
        self.assertEqual(result["n_episodes"], 3)
        self.assertEqual(result["n_steps"], 12)
        self.assertTrue(0.0 <= result["step_em"] <= 1.0)
        self.assertTrue(0.0 <= result["trajectory_em"] <= 1.0)
        self.assertTrue(result["normalized_set_nll"] >= 0)


if __name__ == "__main__":
    unittest.main()
