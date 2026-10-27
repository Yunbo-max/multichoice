"""Torch-only shape, padding and gradient tests for symbolic grid controls."""
import unittest
from treejev.sokoban import DEMO_MAP
try:
    import torch
    from treejev.grid_model import grid_tensor, make_model, loss_terms
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"torch unavailable")
class GridBaselineTests(unittest.TestCase):
    def test_logits_value_gradient(self):
        tensor=grid_tensor(["\n".join(DEMO_MAP)],max_side=10)
        model=make_model(width=16)
        logits,value=model(tensor)
        self.assertEqual(tuple(logits.shape),(1,4))
        self.assertEqual(tuple(value.shape),(1,))
        self.assertTrue(torch.isfinite(value).all())
        row={"target":[0,1,1,0],"distance_to_goal":3}
        loss,_,_=loss_terms(logits,value,[row])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in model.parameters()))

    def test_board_too_large_rejected(self):
        with self.assertRaises(ValueError):
            grid_tensor(["\n".join(DEMO_MAP)],max_side=3)

if __name__=="__main__":unittest.main()
