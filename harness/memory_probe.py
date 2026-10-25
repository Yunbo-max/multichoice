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
    groups: Sequence[str] | None = None,
    return_indices: bool = False,
):
    """Fit a probe without reusing the same episode on both sides.

    If groups are supplied, split entire episodes first; only then train on
    their time steps. Without groups, the legacy per-example split is kept
    for generic IID use, but must NOT be used for recurrent memory evaluation.

    The optional fourth return value contains the fixed train/test indices
    so switch/no-switch diagnostics can evaluate the SAME trained probe
    rather than fitting separate incomparable classifiers.
    """
    if len(memories) != len(labels) or len(memories) < 2:
        raise ValueError("memories and labels must align and contain >=2 rows")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    g = torch.Generator().manual_seed(seed)
    if groups is not None:
        if len(groups) != len(memories) or any(
            not isinstance(item, str) or not item for item in groups
        ):
            raise ValueError("episode group IDs must be aligned nonempty strings")
        episodes = list(dict.fromkeys(groups))
        if len(episodes) < 2:
            raise ValueError("need at least two distinct episodes for held-out probe")
        perm = torch.randperm(len(episodes), generator=g).tolist()
        n_test_groups = min(len(episodes) - 1, max(1, int(len(episodes) * test_fraction)))
        heldout = {episodes[j] for j in perm[:n_test_groups]}
        test_idx = [i for i, episode_id in enumerate(groups) if episode_id in heldout]
        train_idx = [i for i, episode_id in enumerate(groups) if episode_id not in heldout]
    else:
        perm = torch.randperm(len(memories), generator=g).tolist()
        n_test = min(len(memories) - 1, max(1, int(len(memories) * test_fraction)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    model, train_acc = fit_probe(
        [memories[i] for i in train_idx],
        [labels[i] for i in train_idx],
        n_classes,
        steps=steps, lr=lr, seed=seed,
    )
    preds = probe_predictions(model, [memories[i] for i in test_idx])
    test_acc = sum(
        int(p == labels[i]) for p, i in zip(preds, test_idx)
    ) / len(test_idx)
    if not return_indices:
        return model, train_acc, test_acc
    metadata = {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_episodes": len({groups[i] for i in train_idx}) if groups is not None else None,
        "test_episodes": len({groups[i] for i in test_idx}) if groups is not None else None,
        "split_unit": "episode" if groups is not None else "individual_step",
    }
    return model, train_acc, test_acc, metadata


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
