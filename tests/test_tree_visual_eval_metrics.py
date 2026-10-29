"""CPU validation of multiple-optimal-action evaluation aggregation."""
import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"torch unavailable")
class EvaluationMetricsTest(unittest.TestCase):
    def test_multiple_optima_and_stratification(self):
        from treejev.visual_eval_metrics import summarize_visual_actions
        rows=[
          {"target_probs":{"up":0., "right":.5, "down":.5, "left":0.}},
          {"target_probs":{"up":1., "right":0., "down":0., "left":0.}},
        ]
        logits=torch.tensor([[0.,2.,1.,0.],[0.,2.,0.,0.]])
        scores=summarize_visual_actions(rows,logits)
        self.assertEqual(scores["action_optimal_accuracy"],.5)
        self.assertEqual(scores["by_optimal_action_count"]["2"]["n"],1)
        self.assertEqual(scores["by_optimal_action_count"]["1"]["n"],1)
        self.assertGreater(scores["mass_on_optimal_actions"],0)
        self.assertGreater(scores["soft_cross_entropy"],0)
    def test_invalid_dimensions_rejected(self):
        from treejev.visual_eval_metrics import summarize_visual_actions
        with self.assertRaises(ValueError):
            summarize_visual_actions([{"target_probs":{}}],torch.zeros((1,5)))
if __name__=="__main__":
    unittest.main()
