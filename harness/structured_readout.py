"""Structured set readout: constrain the model to the reachable successors.

P0 of the v4 diagnosis. The shift environment only ever reaches the 6 sets
obtained by shifting the current state by k=1..6, yet the raw set-MAP readout
distributes mass over all C(7,2)=21 two-subsets. This module re-reads the model's
7 option scores ONLY over the 6 reachable successors:

    E_k = sum_{i in R_k(S)} s_i,   P(k) = softmax(E_k / tau),   predict R_argmax_k(S).

This answers whether the low on-policy accuracy is mostly a *readout* problem
(wasting probability on the 15 impossible sets) rather than a memory problem.

Dependency-free; ``scores`` is a length-N list, ``state`` a length-N multi-hot.
"""

from __future__ import annotations

import math
from typing import List, Sequence


def shift(state: Sequence[int], k: int) -> List[int]:
    """Shift each active cell forward by k (mod N)."""
    n = len(state)
    return [state[(i - k) % n] for i in range(n)]


def reachable_successors(state: Sequence[int]) -> dict:
    """The N-1 reachable successors of a state (one per distinct shift)."""
    n = len(state)
    return {k: shift(state, k) for k in range(1, n)}


def structured_logits(state: Sequence[int], scores: Sequence[float]) -> dict:
    """E_k = sum of option scores over the cells of shift_k(state)."""
    if len(state) != len(scores):
        raise ValueError("state and scores must have the same length")
    n = len(state)
    return {
        k: sum(scores[i] for i, bit in enumerate(shift(state, k)) if bit)
        for k in range(1, n)
    }


def structured_probs(
    state: Sequence[int], scores: Sequence[float], tau: float = 1.0
) -> dict:
    """P(k) = softmax(E_k / tau) over the N-1 shifts."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    energy = structured_logits(state, scores)
    keys = sorted(energy)
    values = [energy[k] / tau for k in keys]
    peak = max(values)
    denom = sum(math.exp(v - peak) for v in values)
    return {k: math.exp(energy[k] / tau - peak) / denom for k in keys}


def structured_predict(
    state: Sequence[int], scores: Sequence[float], tau: float = 1.0
) -> dict:
    """Predicted set = shift_argmax_k(state), plus the shift distribution."""
    probs = structured_probs(state, scores, tau)
    k_best = max(probs, key=lambda k: probs[k])
    return {"shift": k_best, "set": shift(state, k_best), "probs": probs}


def structured_nll(
    state: Sequence[int], scores: Sequence[float], gold_k: int, tau: float = 1.0
) -> float:
    """-log P(k_gold) under the 6-candidate constrained readout."""
    n = len(state)
    if gold_k not in range(1, n):
        raise ValueError("gold shift out of range")
    probs = structured_probs(state, scores, tau)
    return -math.log(max(probs[gold_k], 1e-300))


def baselines(n_cells: int) -> dict:
    """Theory baselines for the shift environment (N-1 reachable successors)."""
    n_shift = n_cells - 1
    return {
        "random_legal_shift_em": 1.0 / n_shift,
        "uniform_shift_nll": math.log(n_shift),
        "uniform_subsets_nll": math.log(math.comb(n_cells, 2)),
    }


def shift_head_predict(
    state: Sequence[int], shift_logits: Sequence[float], tau: float = 1.0
) -> dict:
    """P(k) = softmax(shift_logits/tau) over shifts 1..K-1; predict successor set.

    ``shift_logits`` is the memory-conditioned shift-head output (K-1 entries).
    """
    n = len(state)
    if len(shift_logits) != n - 1:
        raise ValueError("shift_logits must have K-1 entries")
    if tau <= 0:
        raise ValueError("tau must be positive")
    peak = max(shift_logits)
    probs = {k: math.exp(shift_logits[k - 1] / tau - peak / tau) for k in range(1, n)}
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    k_best = max(probs, key=lambda k: probs[k])
    return {"shift": k_best, "set": shift(state, k_best), "probs": probs}
