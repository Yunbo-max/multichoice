"""Set-revision supervision and metrics (experiment D of the recurrent plan).

Core hypothesis (research plan section 7): learning to *revise* a set — predict
the additions/removals that transform the previous set into the current one — is
more accurate, more efficient, and accumulates better over trajectories than
re-predicting the whole set from scratch.

Dependency-free. Labels are binary multi-hot lists over a fixed candidate order.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def _validate_labels(*labels: Sequence[int]) -> int:
    if not labels:
        raise ValueError("at least one label required")
    n = len(labels[0])
    for label in labels:
        if len(label) != n:
            raise ValueError("all labels must have the same length")
        if any(int(v) not in (0, 1) for v in label):
            raise ValueError("labels must be binary multi-hot")
    return n


def revision_target(prev: Sequence[int], new: Sequence[int]) -> dict:
    """Decompose the transition prev -> new into additions and removals.

    ``additions`` are options entering the set (new=1, prev=0); ``removals`` are
    options leaving it (prev=1, new=0); ``unchanged`` are kept as-is.
    """
    n = _validate_labels(prev, new)
    additions = [int(bool(new[i]) and not bool(prev[i])) for i in range(n)]
    removals = [int(bool(prev[i]) and not bool(new[i])) for i in range(n)]
    unchanged = [int(bool(prev[i]) == bool(new[i])) for i in range(n)]
    return {
        "additions": additions,
        "removals": removals,
        "unchanged": unchanged,
        "n_added": sum(additions),
        "n_removed": sum(removals),
    }


def revision_membership_target(prev: Sequence[int], new: Sequence[int]) -> List[int]:
    """Per-option revision signal: +1 (add), -1 (remove), 0 (keep)."""
    n = _validate_labels(prev, new)
    return [(1 if new[i] else 0) - (1 if prev[i] else 0) for i in range(n)]


def revision_count_target(prev: Sequence[int], new: Sequence[int]) -> int:
    """The cardinality the set should have after the revision."""
    _validate_labels(prev, new)
    return sum(int(v) for v in new)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def revision_metrics(
    prev_sets: Sequence[Sequence[int]],
    new_sets: Sequence[Sequence[int]],
    pred_sets: Sequence[Sequence[int]],
    trajectory_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Aggregate revision diagnostics across steps and (optionally) trajectories.

    Returns repair rate, broken rate, addition/removal precision/recall/F1, and
    full-trajectory exact success. ``trajectory_ids`` groups consecutive steps
    into episodes for the trajectory metric; when omitted, each step is its own
    trajectory.
    """
    if not (len(prev_sets) == len(new_sets) == len(pred_sets)):
        raise ValueError("prev/new/pred sequences must be equally long")
    if trajectory_ids is not None and len(trajectory_ids) != len(prev_sets):
        raise ValueError("trajectory_ids must match the sequence length")
    steps = len(prev_sets)
    if steps == 0:
        raise ValueError("empty sequence")

    repairs = broken = 0
    n_revisable = 0  # steps where prev was wrong (repairable)
    n_held = 0  # steps where prev was right (breakable)
    add_tp = add_fp = add_fn = 0
    rem_tp = rem_fp = rem_fn = 0
    per_step_correct = []
    for prev, new, pred in zip(prev_sets, new_sets, pred_sets):
        _validate_labels(prev, new, pred)
        prev_correct = list(prev) == list(new)
        pred_correct = list(pred) == list(new)
        per_step_correct.append(pred_correct)
        if not prev_correct:
            n_revisable += 1
            if pred_correct:
                repairs += 1
        else:
            n_held += 1
            if not pred_correct:
                broken += 1
        true = revision_target(prev, new)
        done = revision_target(prev, pred)
        for a_gt, a_pd in zip(true["additions"], done["additions"]):
            add_tp += int(a_gt and a_pd)
            add_fp += int(a_pd and not a_gt)
            add_fn += int(a_gt and not a_pd)
        for r_gt, r_pd in zip(true["removals"], done["removals"]):
            rem_tp += int(r_gt and r_pd)
            rem_fp += int(r_pd and not r_gt)
            rem_fn += int(r_gt and not r_pd)

    # Trajectory success: a trajectory counts only if every step is correct.
    if trajectory_ids is None:
        trajectory_ids = list(range(steps))
    traj_correct: dict = {}
    traj_total: dict = {}
    for tid, ok in zip(trajectory_ids, per_step_correct):
        traj_total[tid] = traj_total.get(tid, 0) + 1
        if tid not in traj_correct:
            traj_correct[tid] = True
        traj_correct[tid] = traj_correct[tid] and ok
    trajectory_success = sum(1 for v in traj_correct.values() if v) / len(traj_total)

    return {
        "n_steps": steps,
        "n_trajectories": len(traj_total),
        "repair_rate": repairs / n_revisable if n_revisable else 0.0,
        "broken_rate": broken / n_held if n_held else 0.0,
        "additions": _prf(add_tp, add_fp, add_fn),
        "removals": _prf(rem_tp, rem_fp, rem_fn),
        "trajectory_success": trajectory_success,
    }
