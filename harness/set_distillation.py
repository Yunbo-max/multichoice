"""Coherent membership/cardinality targets and exact set readout.

The scalar inference/target helpers use only the standard library. PyTorch is
imported only by the differentiable loss. Neither targets nor teacher weights
depend on the student's current parameters. See docs/multiteacher_set_distillation.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TARGET_SCHEMA = "decision_set_target_v1"
READOUT_SCHEMA = "decision_set_readout_v1"
SET_LOSS_FAMILIES = ("set_ce", "independent_marginal", "induced_marginal")


def allowed_counts(row):
    """Use the declared task protocol, never the annotated answer count."""
    k = len(row["options"])
    mode = row.get("selection_mode", "multi")
    if mode not in ("single", "multi"):
        raise ValueError(f"unknown selection_mode: {mode!r}")
    empty = row.get("allow_empty", False)
    if not isinstance(empty, bool):
        raise ValueError("allow_empty must be a boolean")
    if mode == "single":
        if empty:
            raise ValueError("single-choice protocol cannot allow an empty answer")
        return [1]
    return list(range(0 if empty else 1, k + 1))


def input_digest(row):
    """Content identity excludes gold, split and transient generation metadata."""
    payload = {name: row[name] for name in ("state", "question", "options")}
    payload.update(
        selection_mode=row.get("selection_mode", "multi"),
        allow_empty=row.get("allow_empty", False),
    )
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _probabilities(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or not 0 <= v <= 1
        for v in values
    ):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return [float(v) for v in values]


def _support(allowed, k):
    values = list(allowed)
    if (
        not values
        or len(set(values)) != len(values)
        or any(
            isinstance(m, bool) or not isinstance(m, int) or not 0 <= m <= k
            for m in values
        )
    ):
        raise ValueError(f"invalid count support for K={k}: {values}")
    return sorted(values)


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def _logsum(values):
    total = -math.inf
    for value in values:
        total = _logadd(total, value)
    return total


def conditioned_bernoulli_stats(probabilities, allowed):
    """Moments of a Bernoulli product conditioned on allowed counts.

    Endpoints 0 and 1 are handled exactly (no hidden clipping). A log-domain
    forward/backward dynamic program runs in O(K^2) time and memory. Conditioning
    an event with exactly zero probability raises, rather than inventing targets.
    """
    b = _probabilities(probabilities, "membership")
    k = len(b)
    support = _support(allowed, k)
    include = [math.log(v) if v > 0 else -math.inf for v in b]
    exclude = [math.log1p(-v) if v < 1 else -math.inf for v in b]
    forward = [[0.0]]
    for j in range(k):
        previous = forward[-1]
        current = []
        for m in range(j + 2):
            out = previous[m] + exclude[j] if m <= j else -math.inf
            inn = previous[m - 1] + include[j] if m else -math.inf
            current.append(_logadd(out, inn))
        forward.append(current)
    log_normalizer = _logsum(forward[-1][m] for m in support)
    if not math.isfinite(log_normalizer):
        raise ValueError(
            "teacher assigns zero probability to the declared task protocol"
        )
    rho = [
        math.exp(v - log_normalizer) if m in support else 0.0
        for m, v in enumerate(forward[-1])
    ]
    # Reverse differentiation w.r.t. each include log potential gives E[Y_i].
    adjoint, q = list(rho), [0.0] * k
    for j in range(k, 0, -1):
        previous = [0.0] * j
        for m, mass in enumerate(adjoint):
            if mass == 0 or forward[j][m] == -math.inf:
                continue
            if m:
                term = forward[j - 1][m - 1] + include[j - 1]
                routed = mass * math.exp(term - forward[j][m])
                q[j - 1] += routed
                previous[m - 1] += routed
            if m < j:
                term = forward[j - 1][m] + exclude[j - 1]
                previous[m] += mass * math.exp(term - forward[j][m])
        adjoint = previous
    # Round-off only; no alteration of teacher support or zero-mass events.
    q = [min(1.0, max(0.0, value)) for value in q]
    validate_set_targets(q, rho, support)
    return q, rho


def validate_set_targets(q, rho, allowed, tolerance=1e-7):
    q = _probabilities(q, "q")
    rho = _probabilities(rho, "count_probs")
    k = len(q)
    support = _support(allowed, k)
    if len(rho) != k + 1:
        raise ValueError(f"count_probs needs K+1={k + 1} entries")
    if abs(sum(rho) - 1.0) > tolerance:
        raise ValueError("count_probs must sum to one")
    if any(rho[m] > tolerance for m in range(k + 1) if m not in support):
        raise ValueError("count_probs assigns mass outside the task protocol")
    if abs(sum(q) - sum(m * mass for m, mass in enumerate(rho))) > tolerance * max(
        1, k
    ):
        raise ValueError("membership/count expectations disagree")
    # Necessary and sufficient feasibility inequalities for marginals coupled
    # to this count law: the marginal polytope is a mixture of hypersimplices.
    ordered = sorted(q, reverse=True)
    for j in range(1, k + 1):
        upper = sum(min(j, m) * mass for m, mass in enumerate(rho))
        if sum(ordered[:j]) > upper + tolerance * k:
            raise ValueError(
                "membership/count statistics cannot describe one set distribution"
            )


def log_partitions(scores, return_table=False):
    """log Z_m = log sum_{|S|=m} exp(sum_{i in S} scores_i)."""
    if not scores or any(not math.isfinite(s) for s in scores):
        raise ValueError("scores must be nonempty and finite")
    table = [[0.0]]
    for score in scores:
        old = table[-1]
        table.append(
            [old[0]]
            + [_logadd(old[m], score + old[m - 1]) for m in range(1, len(old))]
            + [score + old[-1]]
        )
    return table if return_table else table[-1]


def structured_marginals(scores, count_probs):
    """Exact P(i in S), in O(K^2), from the separate count distribution."""
    rho = _probabilities(count_probs, "count_probs")
    k = len(scores)
    if len(rho) != k + 1 or abs(sum(rho) - 1) > 1e-7:
        raise ValueError("invalid normalized count distribution")
    # Centering is exact because each fixed-size subset gains the same m*c.
    center = max(scores)
    scores = [s - center for s in scores]
    table = log_partitions(scores, return_table=True)
    adjoint, p = list(rho), [0.0] * k
    for j in range(k, 0, -1):
        previous = [0.0] * j
        for m, mass in enumerate(adjoint):
            if m:
                routed = mass * math.exp(
                    scores[j - 1] + table[j - 1][m - 1] - table[j][m]
                )
                p[j - 1] += routed
                previous[m - 1] += routed
            if m < j:
                previous[m] += mass * math.exp(table[j - 1][m] - table[j][m])
        adjoint = previous
    return [min(1.0, max(0.0, v)) for v in p]


def structured_prediction(scores, count_logits, row, count_anchor=False):
    """Coherent marginals and exact set MAP; never uses the gold answer count."""
    k = len(scores)
    if k != len(row["options"]) or len(count_logits) < k + 1:
        raise ValueError("readout dimensions do not match options")
    support = allowed_counts(row)
    if count_anchor:
        count_logits = count_anchor_logits(scores, count_logits)
    if any(not math.isfinite(count_logits[m]) for m in support):
        raise ValueError("nonfinite count logits")
    log_total = _logsum(count_logits[m] for m in support)
    log_rho = {m: count_logits[m] - log_total for m in support}
    rho = [math.exp(log_rho[m]) if m in log_rho else 0.0 for m in range(k + 1)]
    center = max(scores)
    centered = [v - center for v in scores]
    logz = log_partitions(centered)
    order = sorted(range(k), key=lambda i: (-centered[i], i))
    prefix = [0.0]
    for index in order:
        prefix.append(prefix[-1] + centered[index])
    # Tie: smallest m, then lowest option indices. Count mode alone is not MAP.
    best_m = max(support, key=lambda m: (log_rho[m] + prefix[m] - logz[m], -m))
    return {
        "p": structured_marginals(centered, rho),
        "count_probs": rho,
        "count_log_probs": [log_rho.get(m) for m in range(k + 1)],
        "predicted_set_log_probability": log_rho[best_m]
        + prefix[best_m]
        - logz[best_m],
        "predicted_indices": sorted(order[:best_m]),
        "decoder": "set_map",
    }


def scalar_set_cross_entropy(scores, count_logits, q, rho, allowed, normalize=True):
    """Numerical oracle; the differentiable implementation is below."""
    validate_set_targets(q, rho, allowed)
    support = _support(allowed, len(scores))
    centered = [v - max(scores) for v in scores]
    logz = log_partitions(centered)
    lognorm = _logsum(count_logits[m] for m in support)
    value = sum(rho[m] * (lognorm - count_logits[m] + logz[m]) for m in support)
    value -= sum(a * b for a, b in zip(q, centered))
    return value / len(scores) if normalize else value


def structured_log_marginals(scores, count_logits, allowed):
    """Stable log P(i in S) and log P(i not in S), including rare events.

    Unlike taking log of ``structured_marginals``, the log-domain reverse DP
    retains probabilities smaller than floating-point exponentiation can hold.
    Both branches are computed explicitly; no ``log(1 - rounded_probability)``
    or numerical clipping is used. Runtime and memory are O(K^2).
    """
    if not scores or any(not math.isfinite(s) for s in scores):
        raise ValueError("scores must be nonempty and finite")
    k = len(scores)
    support = _support(allowed, k)
    if len(count_logits) < k + 1 or any(
        not math.isfinite(count_logits[m]) for m in support
    ):
        raise ValueError("count logits must be finite on the allowed count support")
    centered = [v - max(scores) for v in scores]
    table = log_partitions(centered, return_table=True)
    normalizer = _logsum(count_logits[m] for m in support)
    adjoint = {m: count_logits[m] - normalizer for m in support}
    log_p, log_not_p = [None] * k, [None] * k
    for j in range(k, 0, -1):
        previous, included, excluded = {}, [], []
        for m, mass in adjoint.items():
            if m:
                value = mass + centered[j - 1] + table[j - 1][m - 1] - table[j][m]
                included.append(value)
                previous[m - 1] = _logadd(previous.get(m - 1, -math.inf), value)
            if m < j:
                value = mass + table[j - 1][m] - table[j][m]
                excluded.append(value)
                previous[m] = _logadd(previous.get(m, -math.inf), value)
        inn, out = _logsum(included), _logsum(excluded)
        pair_normalizer = _logadd(inn, out)
        log_p[j - 1], log_not_p[j - 1] = inn - pair_normalizer, out - pair_normalizer
        adjoint = previous
    return log_p, log_not_p


def validate_set_loss_options(
    loss_family="set_ce", count_weight=1.0, count_target="teacher"
):
    """Reject controls that would silently change the mathematical comparison."""
    if loss_family not in SET_LOSS_FAMILIES:
        raise ValueError(f"unknown set loss family: {loss_family!r}")
    if (
        isinstance(count_weight, bool)
        or not isinstance(count_weight, (int, float))
        or not math.isfinite(count_weight)
        or count_weight < 0
    ):
        raise ValueError("count_weight must be finite and nonnegative")
    if count_target not in ("teacher", "gold"):
        raise ValueError("count_target must be teacher or gold")
    if loss_family == "set_ce" and count_weight != 1:
        raise ValueError(
            "set_ce requires count_weight=1; weighting changes exact set cross-entropy"
        )
    if count_target == "gold" and loss_family != "set_ce":
        raise ValueError(
            "legacy count_target=gold cannot be combined with a moment control"
        )


def scalar_moment_cross_entropy(
    scores,
    count_logits,
    q,
    rho,
    allowed,
    loss_family="induced_marginal",
    count_weight=1.0,
    normalize=True,
):
    """Dependency-free numerical oracle for the two information-matched controls.

    Both controls receive the same q and rho as exact set CE. ``induced_marginal``
    supervises the actual marginals of the normalized set predictor, whereas
    ``independent_marginal`` supervises raw sigmoid scores and a separate count
    head. Fitting those raw scores to q does not generally fit the structured
    decoder's induced marginals to q, even at the auxiliary loss optimum.
    """
    validate_set_loss_options(loss_family, count_weight)
    if loss_family == "set_ce":
        raise ValueError("use scalar_set_cross_entropy for set_ce")
    validate_set_targets(q, rho, allowed)
    support = _support(allowed, len(scores))
    if len(q) != len(scores) or len(count_logits) < len(scores) + 1:
        raise ValueError("score/count dimensions do not match target")
    if any(not math.isfinite(s) for s in scores) or any(
        not math.isfinite(count_logits[m]) for m in support
    ):
        raise ValueError("scores and supported count logits must be finite")
    if loss_family == "induced_marginal":
        log_p, log_not_p = structured_log_marginals(scores, count_logits, support)
    else:
        log_p = [-_logadd(0.0, -s) for s in scores]
        log_not_p = [-_logadd(0.0, s) for s in scores]
    # Zero targets must not multiply -inf at deterministic protocol endpoints.
    value = -sum(t * p for t, p in zip(q, log_p) if t > 0)
    value -= sum((1 - t) * p for t, p in zip(q, log_not_p) if t < 1)
    lognorm = _logsum(count_logits[m] for m in support)
    value += count_weight * sum(rho[m] * (lognorm - count_logits[m]) for m in support)
    return value / len(scores) if normalize else value


def torch_log_partitions(scores, return_table=False):
    """Autograd-safe log DP: no logaddexp(-inf,-inf) boundary gradients."""
    import torch

    old = scores.new_zeros(1)
    table = [old]
    for score in scores:
        middle = torch.logaddexp(old[1:], score + old[:-1])
        old = torch.cat((old[:1], middle, (score + old[-1]).reshape(1)))
        if return_table:
            table.append(old)
    return table if return_table else old


def count_anchor_logits(scores, count_logits):
    """Residual count-head anchor: ``count_logits + log Z_m(scores)`` (center=max).

    Modification 2 of ``docs/setjev_v2_plan.md``. ``log Z_m(s)`` is the candidate-
    score-implied count normalizer, so a zero correction recovers the count
    distribution implied by the candidate scores (cold start from BCE) instead of
    a randomly-initialized count head. ``center=max`` matches
    ``structured_prediction``/``structured_marginals``; the learned correction
    absorbs any constant shift, and the convention must be identical in training
    and inference.
    """
    if not scores or any(not math.isfinite(s) for s in scores):
        raise ValueError("scores must be nonempty and finite")
    if len(count_logits) != len(scores) + 1:
        raise ValueError("count_logits needs K+1 entries")
    center = max(scores)
    logz = log_partitions([s - center for s in scores])
    return [count_logits[m] + logz[m] for m in range(len(count_logits))]


def torch_count_anchor_logits(scores, count_logits):
    """Autograd-safe residual count-head anchor (see ``count_anchor_logits``)."""
    import torch

    centered = scores - scores.max()
    logz = torch_log_partitions(centered)
    return count_logits + logz


def torch_structured_log_marginals(scores, log_count, allowed):
    """Differentiable log inclusion/exclusion marginals of the set predictor.

    ``log_count`` contains normalized count log-probabilities in the same order
    as ``allowed``. It must retain its gradient: the induced-marginal objective
    updates both candidate scores and the count head. The reverse DP stores only
    structurally possible states, avoiding logaddexp(-inf, -inf) gradients.
    """
    import torch

    supplied = list(allowed)
    support = _support(supplied, len(scores))
    if supplied != support or len(log_count) != len(support):
        raise ValueError("log_count must align with sorted unique allowed counts")
    centered = scores - scores.mean()
    table = torch_log_partitions(centered, return_table=True)
    adjoint = {m: log_count[i] for i, m in enumerate(support)}
    log_p, log_not_p = [None] * len(scores), [None] * len(scores)
    for j in range(len(scores), 0, -1):
        previous, included, excluded = {}, [], []
        for m, mass in adjoint.items():
            if m:
                value = mass + centered[j - 1] + table[j - 1][m - 1] - table[j][m]
                included.append(value)
                previous[m - 1] = (
                    torch.logaddexp(previous[m - 1], value)
                    if m - 1 in previous
                    else value
                )
            if m < j:
                value = mass + table[j - 1][m] - table[j][m]
                excluded.append(value)
                previous[m] = (
                    torch.logaddexp(previous[m], value) if m in previous else value
                )
        inn = (
            torch.logsumexp(torch.stack(included), 0)
            if included
            else scores.new_tensor(-math.inf)
        )
        out = (
            torch.logsumexp(torch.stack(excluded), 0)
            if excluded
            else scores.new_tensor(-math.inf)
        )
        pair_normalizer = torch.logaddexp(inn, out)
        log_p[j - 1], log_not_p[j - 1] = inn - pair_normalizer, out - pair_normalizer
        adjoint = previous
    return torch.stack(log_p), torch.stack(log_not_p)


def set_loss_terms(
    label_logits,
    count_logits,
    rows,
    count_target="teacher",
    *,
    loss_family="set_ce",
    count_weight=1.0,
    count_anchor=False,
):
    """Per-row set objective, raw-score BCE diagnostic, and unweighted count CE/K.

    All three loss families share architecture, q/rho targets and set-MAP decoder.
    set_ce is exact set cross-entropy. independent_marginal uses sigmoid scores;
    induced_marginal uses actual marginals of the normalized set distribution.
    The moment controls add count_weight * CE(rho_target, rho_student)/K to
    candidate-averaged BCE. set_ce requires count_weight=1 by definition.

    count_target='gold' replaces teacher cardinality with the gold one-hot while
    retaining teacher membership only in an auxiliary BCE objective. This is
    the explicit marginal-only ablation, not an incoherent set target.
    """
    validate_set_loss_options(loss_family, count_weight, count_target)
    import torch
    import torch.nn.functional as F

    losses, bces, count_ces = [], [], []
    for i, row in enumerate(rows):
        k = len(row["options"])
        scores = label_logits[i, :k].float()
        q = scores.new_tensor(row["train_target"])
        rho = scores.new_tensor(row["train_count_probs"])
        support = allowed_counts(row)
        index = torch.tensor(support, dtype=torch.long, device=scores.device)
        head = count_logits[i, : k + 1].float()
        if count_anchor:
            head = torch_count_anchor_logits(scores, head)
        log_count = F.log_softmax(head[index], dim=0)
        bce = F.binary_cross_entropy_with_logits(scores, q, reduction="mean")
        if count_target == "gold":
            count_ce = -log_count[support.index(sum(row["target"]))] / k
            # This control keeps the same head and decoder, distilling only q.
            loss = bce + count_ce
        else:
            count_ce = -(rho[index] * log_count).sum() / k
            if loss_family == "set_ce":
                centered = scores - scores.mean()
                logz = torch_log_partitions(centered)
                loss = (
                    count_ce
                    + ((rho[index] * logz[index]).sum() - (q * centered).sum()) / k
                )
            elif loss_family == "independent_marginal":
                loss = bce + count_weight * count_ce
            else:
                log_p, log_not_p = torch_structured_log_marginals(
                    scores, log_count, support
                )
                positive, negative = q > 0, q < 1
                marginal_ce = -(q[positive] * log_p[positive]).sum()
                marginal_ce -= ((1 - q[negative]) * log_not_p[negative]).sum()
                loss = marginal_ce / k + count_weight * count_ce
        losses.append(loss)
        bces.append(bce.detach())  # diagnostic sigmoid BCE, not structured marginals
        count_ces.append(count_ce)
    return torch.stack(losses), torch.stack(bces), torch.stack(count_ces)


def read_set_target_file(path):
    mapping, errors = {}, []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            ident = item.get("id")
            if not isinstance(ident, str) or not ident or ident in mapping:
                raise ValueError("missing or duplicate target id")
            if (
                item.get("schema") != TARGET_SCHEMA
                or item.get("q_semantics") != "set_membership"
            ):
                raise ValueError(
                    "expected coherent set targets from build_multiteacher_targets.py"
                )
            validate_set_targets(item["q"], item["count_probs"], item["allowed_counts"])
            mapping[ident] = item
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"{path}:{number}: {exc}")
    if not mapping and not errors:
        errors.append(f"{path}: empty target file")
    return mapping, errors


def attach_set_targets(rows, mapping=None, gold_weight=0.0, missing="error"):
    """Attach a fixed gold/teacher mixture; gold-only is a delta set law."""
    if not 0 <= gold_weight <= 1 or not math.isfinite(gold_weight):
        raise ValueError("gold_weight must be in [0, 1]")
    if missing not in ("error", "gold"):
        raise ValueError(
            "set targets support only missing=error or explicit gold fallback"
        )
    errors, used, absent = [], 0, []
    for row in rows:
        try:
            k = len(row["options"])
            support = allowed_counts(row)
            gold = [float(v) for v in row["target"]]
            gold_count = [float(m == sum(gold)) for m in range(k + 1)]
            validate_set_targets(gold, gold_count, support)
            item = None if mapping is None else mapping.get(row["id"])
            if item is None:
                if mapping is not None:
                    absent.append(row["id"])
                    if missing == "error":
                        continue
                q, rho = gold, gold_count
            else:
                if item.get("option_keys") != [o["key"] for o in row["options"]]:
                    raise ValueError("teacher option order/identity mismatch")
                if item.get("input_digest") != input_digest(row):
                    raise ValueError("teacher target input content/protocol mismatch")
                if item.get("allowed_counts") != support:
                    raise ValueError("teacher target protocol mismatch")
                validate_set_targets(item["q"], item["count_probs"], support)
                if len(item["q"]) != k:
                    raise ValueError("teacher target K mismatch")
                q = [
                    (1 - gold_weight) * a + gold_weight * b
                    for a, b in zip(item["q"], gold)
                ]
                rho = [
                    (1 - gold_weight) * a + gold_weight * b
                    for a, b in zip(item["count_probs"], gold_count)
                ]
                used += 1
            row["train_target"], row["train_count_probs"] = q, rho
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{row.get('id')}: {exc}")
    if absent and missing == "error":
        errors.append(f"{len(absent)} missing teacher targets, e.g. {absent[:5]}")
    return errors, {
        "source": "gold" if mapping is None else "set_teacher",
        "from_file": used,
        "missing": len(absent),
        "missing_mode": missing,
        "gold_weight": 1.0 if mapping is None else gold_weight,
        "extra_ids": 0
        if mapping is None
        else len(set(mapping) - {r["id"] for r in rows}),
    }
