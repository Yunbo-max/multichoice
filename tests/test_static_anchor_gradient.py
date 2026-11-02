"""Regression: anchored full/detached variant has identical forward, distinct gradient."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

try:
    import torch
    from set_distillation import set_loss_terms
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "torch unavailable")
class AnchorGradientTests(unittest.TestCase):
    def test_same_forward_different_score_gradient(self):
        def run(detach):
            s = torch.tensor([[0.4, -0.3, 0.8]], requires_grad=True)
            a = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
            row = {"options": [{"key": k} for k in "abc"],
                   "selection_mode": "multi", "allow_empty": False,
                   "target": [1, 0, 1], "train_target": [1.0, 0.0, 1.0],
                   "train_count_probs": [0.0, 0.0, 1.0, 0.0]}
            loss, _, _ = set_loss_terms(
                s, a, [row], count_anchor=True,
                count_anchor_detach=detach, loss_family="set_ce")
            loss.sum().backward()
            return loss.item(), s.grad.detach(), a.grad.detach()
        full, gs, ga = run(False)
        detached, hs, ha = run(True)
        self.assertAlmostEqual(full, detached, places=6)
        self.assertTrue(torch.allclose(ga, ha, atol=1e-6))
        self.assertFalse(torch.allclose(gs, hs, atol=1e-5))
    def test_detach_requires_anchor(self):
        s = torch.zeros((1, 3)); a = torch.zeros((1, 4))
        with self.assertRaisesRegex(ValueError, "requires count_anchor"):
            set_loss_terms(s, a, [], count_anchor_detach=True)

if __name__ == "__main__":
    unittest.main()
