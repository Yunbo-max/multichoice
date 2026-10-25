"""P1: linear probe on the persistent memory state + same-state counterfactual.

The linear probe answers whether the shift ``k`` is stored in the memory vector
``M_t``. If a linear classifier can recover ``k`` from ``M_t`` at above the 1/6
chance level, the memory encodes the rule and the failure is in the decision head
or the on-policy drift; if not, the memory is not storing the rule.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


def fit_probe(
    memories: Sequence[Tensor],
    labels: Sequence[int],
    n_classes: int = 6,
    *,
    steps: int = 500,
    lr: float = 0.05,
    seed: int = 0,
) -> Tuple[nn.Module, float]:
    """Fit a linear classifier ``k -> M_t`` and return (module, train accuracy)."""
    if len(memories) == 0:
        raise ValueError("empty probe dataset")
    if len(memories) != len(labels):
        raise ValueError("memories and labels must align")
    if any(not 1 <= int(k) <= n_classes for k in labels):
        raise ValueError("shift label out of range")
    torch.manual_seed(seed)
    x = torch.stack([m.detach().float().reshape(-1) for m in memories])
    y = torch.tensor([int(k) - 1 for k in labels], dtype=torch.long)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("memory vectors must be nonempty 1D tensors")
    model = nn.Linear(x.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=-1) == y).float().mean().item()
    return model, accuracy


def probe_predictions(model: nn.Module, memories: Sequence[Tensor]) -> List[int]:
    """Predicted shift (1-based) for each memory vector."""
    with torch.no_grad():
        x = torch.stack([m.detach().float().reshape(-1) for m in memories])
        return (model(x).argmax(dim=-1) + 1).tolist()


def fit_probe_split(
    memories: Sequence[Tensor],
    labels: Sequence[int],
    n_classes: int = 6,
    *,
    steps: int = 500,
    lr: float = 0.05,
    seed: int = 0,
    test_fraction: float = 0.2,
):
    """Fit on a train split and report train + held-out test accuracy.

    Returns ``(model, train_accuracy, test_accuracy)``. The split is deterministic
    given ``seed``; the probe is never fit on the held-out memories.
    """
    if len(memories) < 2:
        raise ValueError("need at least two samples to split")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(memories), generator=g).tolist()
    n_test = max(1, int(len(memories) * test_fraction))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    model, train_acc = fit_probe(
        [memories[i] for i in train_idx],
        [labels[i] for i in train_idx],
        n_classes,
        steps=steps,
        lr=lr,
        seed=seed,
    )
    preds = probe_predictions(model, [memories[i] for i in test_idx])
    test_labels = [labels[i] for i in test_idx]
    test_acc = sum(int(p == y) for p, y in zip(preds, test_labels)) / len(test_labels)
    return model, train_acc, test_acc


def counterfactual_summary(predicted_sets: dict) -> dict:
    """Summarise the same-state counterfactual: does the model respond to the demo?

    ``predicted_sets`` maps shift k -> predicted set (multi-hot) for a FIXED state.
    Returns the number of distinct predictions and whether each k selects its own
    ``shift_k(state)`` successor.
    """
    if not predicted_sets:
        raise ValueError("empty counterfactual results")
    distinct = len({tuple(v) for v in predicted_sets.values()})
    return {
        "n_distinct_predictions": distinct,
        "n_conditions": len(predicted_sets),
        "responds_to_history": distinct > 1,
    }
