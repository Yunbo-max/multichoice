"""Correct proper-scoring metrics for tied-optimal Sokoban action targets."""
from __future__ import annotations


def summarize_visual_actions(rows, logits):
    import torch
    import torch.nn.functional as F
    from treejev.data import ACTION_ORDER
    if not rows or logits.shape != (len(rows), len(ACTION_ORDER)):
        raise ValueError("action predictions must be N x 4 for N nonempty rows")
    targets = torch.tensor(
        [[r["target_probs"][a] for a in ACTION_ORDER] for r in rows],
        dtype=torch.float32, device=logits.device
    )
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("nonfinite action scores or labels")
    if not torch.allclose(targets.sum(dim=-1), torch.ones(len(rows),device=logits.device),
                          atol=1e-5):
        raise ValueError("non-normalized tied-optimal distribution")
    mask = targets > 0
    logits = logits.float()
    probabilities = logits.softmax(dim=-1)
    pred_idx = logits.argmax(dim=-1)
    hits = mask[torch.arange(len(rows), device=logits.device),pred_idx]
    total_mass = (probabilities*mask).sum(dim=-1)
    per_row_ce = -(targets*F.log_softmax(logits,dim=-1)).sum(dim=-1)
    per_row_brier = (probabilities-targets).square().sum(dim=-1)
    n_opt = mask.sum(dim=-1)
    result = {
        "n":len(rows),
        "action_optimal_accuracy":float(hits.float().mean()),
        "soft_cross_entropy":float(per_row_ce.mean()),
        "mass_on_optimal_actions":float(total_mass.mean()),
        "categorical_brier":float(per_row_brier.mean()),
        "by_optimal_action_count":{},
    }
    for count in sorted(set(n_opt.cpu().tolist())):
        selected = n_opt == count
        result["by_optimal_action_count"][str(count)] = {
            "n":int(selected.sum().item()),
            "accuracy":float(hits[selected].float().mean()),
            "soft_cross_entropy":float(per_row_ce[selected].mean()),
            "mass_on_optimal_actions":float(total_mass[selected].mean()),
        }
    return result
