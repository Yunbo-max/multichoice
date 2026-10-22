"""Dependency-free validation, set decoding and multi-select evaluation metrics.

No threshold fitting occurs here. Explicit single-choice questions use argmax;
all other rows use independent probabilities at the supplied frozen threshold.
A one-positive gold target does not turn a multi-select question into single.
``decoder='provided'`` (or auto with decoder='set_map') scores a model's supplied
set, permitting structured decoders without giving the model the gold count.

The separate oracle_gold_count diagnostic uses gold cardinality, so it is not a
deployable prediction rule. Exact score ties use original option order; random
tie expectation and strict-separation EM are also reported.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROBABILITY_EPSILON = 1e-12


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def validate_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be a finite number in [0, 1]")
    return float(value)


def _target_and_protocol(row):
    ident = row.get("id", "<missing>")
    options, target = row.get("options"), row.get("target")
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        raise ValueError(f"{ident}: options must contain 2..255 entries")
    if not isinstance(target, list) or len(target) != len(options):
        raise ValueError(f"{ident}: target must match the option count")
    if any(type(value) is not int or value not in (0, 1) for value in target):
        raise ValueError(f"{ident}: target must be a multi-hot integer list")
    mode = row.get("selection_mode", "multi")
    if mode not in ("single", "multi"):
        raise ValueError(f"{ident}: selection_mode must be single or multi")
    allow_empty = row.get("allow_empty", False)
    if not isinstance(allow_empty, bool):
        raise ValueError(f"{ident}: allow_empty must be boolean")
    if mode == "single" and (sum(target) != 1 or allow_empty):
        raise ValueError(f"{ident}: explicit single protocol requires exactly one gold and allow_empty=false")
    if not allow_empty and not any(target):
        raise ValueError(f"{ident}: gold target is empty but allow_empty=false")
    return target, mode, allow_empty


def read_dataset(path, expected_split="test"):
    """Validate test/dev/train rows without importing torch or transformers."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    seen = set()
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident:
            raise ValueError(f"{path}: id must be a nonempty string")
        if ident in seen:
            raise ValueError(f"{path}: duplicate id {ident!r}")
        seen.add(ident)
        if expected_split and row.get("split") not in (None, expected_split):
            raise ValueError(f"{ident}: split {row.get('split')!r} != {expected_split!r}")
        for name in ("source", "lang"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"{ident}: {name} must be a nonempty string")
        if "state" not in row or not isinstance(row.get("question"), str):
            raise ValueError(f"{ident}: state and a string question are required")
        _target_and_protocol(row)
        options = row["options"]
        if any(not isinstance(option, dict) or not isinstance(option.get("key"), str)
               or not option["key"] or not isinstance(option.get("desc"), str)
               for option in options):
            raise ValueError(f"{ident}: each option needs a nonempty key and string desc")
        if len({option["key"] for option in options}) != len(options):
            raise ValueError(f"{ident}: option keys must be unique")
        if row.get("n_options") != len(options):
            raise ValueError(f"{ident}: n_options does not match options")
        letters = row.get("gold_letters")
        if isinstance(letters, str) and letters:
            expected = "".join(option["key"] for option, value in zip(options, row["target"]) if value)
            if sorted(letters) != sorted(expected):
                raise ValueError(f"{ident}: gold_letters does not match target")
    return rows


def dataset_stats(rows):
    def counts(values):
        return {str(key): count for key, count in sorted(Counter(values).items(), key=lambda x: str(x[0]))}

    return {"n": len(rows), "by_source": counts(r.get("source") for r in rows),
            "by_lang": counts(r.get("lang") for r in rows),
            "option_counts": counts(len(r["options"]) for r in rows),
            "positive_counts": counts(sum(r["target"]) for r in rows),
            "selection_modes": counts(r.get("selection_mode", "multi") for r in rows),
            "legacy_protocol_defaults": sum("selection_mode" not in r for r in rows)}


def _numbers(values, length, ident, name, probabilities=False):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{ident}: {name} must contain {length} values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError(f"{ident}: {name} must contain finite numbers")
    if probabilities and any(not 0 <= value <= 1 for value in values):
        raise ValueError(f"{ident}: {name} values must be in [0, 1]")
    return [float(value) for value in values]


def _mean_bce(target, probabilities, item):
    if "logits" in item:
        logits = _numbers(item["logits"], len(target), item["id"], "logits")
        # softplus(s) - y*s, written to remain stable at extreme logits.
        return sum(max(score, 0.0) - value * score + math.log1p(math.exp(-abs(score)))
                   for score, value in zip(logits, target)) / len(target), "logits"
    if "bce" in item:
        value = item["bce"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{item['id']}: bce must be finite and nonnegative")
        return float(value), "supplied"
    clipped = [min(1 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, p)) for p in probabilities]
    return -sum(math.log(p) if value else math.log1p(-p)
                for value, p in zip(target, clipped)) / len(target), "clipped_probabilities"


def _oracle_ranking(target, scores):
    gold = {i for i, value in enumerate(target) if value}
    count = len(gold)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected = set(order[:count])
    exact = int(selected == gold)
    boundary_tie = bool(0 < count < len(scores) and scores[order[count - 1]] == scores[order[count]])
    strict = int(not gold or len(gold) == len(scores)
                 or min(scores[i] for i in gold) > max(scores[i] for i in range(len(scores)) if i not in gold))
    expected = float(exact)
    if boundary_tie:
        cutoff = scores[order[count - 1]]
        above = {i for i, score in enumerate(scores) if score > cutoff}
        tied = {i for i, score in enumerate(scores) if score == cutoff}
        compatible = above <= gold and gold <= above | tied
        expected = 1.0 / math.comb(len(tied), count - len(above)) if compatible else 0.0
    return {"predicted_indices": sorted(selected), "question_em": exact,
            "cutoff_tie": boundary_tie, "question_em_expected_random_ties": expected,
            "strict_separation_question_em": strict}


def prediction_records(rows, predictions, threshold=0.5, decoder="protocol"):
    """Align and validate all IDs, then export probabilities and decoded sets.

    Provided sets are used only when explicitly requested, or in auto mode for
    records marked decoder='set_map'. Otherwise saved sets are recomputed at the
    supplied threshold; they cannot silently carry a different tuned threshold.
    """
    threshold = validate_threshold(threshold)
    if decoder not in ("protocol", "provided", "auto"):
        raise ValueError("decoder must be protocol, provided or auto")
    expected, by_id = set(), {}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in expected:
            raise ValueError("dataset IDs must be unique nonempty strings")
        expected.add(ident)
    for item in predictions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("predictions must be objects with string IDs")
        if item["id"] in by_id:
            raise ValueError(f"duplicate prediction for {item['id']}")
        by_id[item["id"]] = item
    if set(by_id) != expected:
        raise ValueError(f"prediction IDs mismatch: missing={sorted(expected - set(by_id))[:8]}, "
                         f"extra={sorted(set(by_id) - expected)[:8]}")
    result = []
    for row in rows:
        ident, item = row["id"], by_id[row["id"]]
        target, mode, allow_empty = _target_and_protocol(row)
        k = len(target)
        probs = _numbers(item.get("p", item.get("probs")), k, ident, "p", probabilities=True)
        scores = item.get("candidate_logits", item.get("logits", probs))
        scores = _numbers(scores, k, ident, "ranking scores")
        use_provided = decoder == "provided" or (decoder == "auto" and item.get("decoder") == "set_map")
        if use_provided:
            selected = item.get("predicted_indices")
            if (not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < k for i in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError(f"{ident}: provided predicted_indices must be unique indices in [0, K)")
            selected = sorted(selected)
            method = item.get("decoder", "provided")
        elif mode == "single":
            # Sigmoid can round several distinct large logits to 1.0. Preserve
            # their true ordering whenever raw scores accompanied the export.
            selected = [max(range(k), key=lambda i: (scores[i], -i))]
            method = "argmax_explicit_single"
        else:
            selected = [i for i, probability in enumerate(probs) if probability >= threshold]
            method = "threshold_multi"
        bce, bce_origin = _mean_bce(target, probs, item)
        keys = [option.get("key", str(i)) for i, option in enumerate(row["options"])]
        record = {
            "id": ident, "source": row.get("source"), "lang": row.get("lang"),
            "split": row.get("split"), "selection_mode": mode, "allow_empty": allow_empty,
            "n_options": k, "option_keys": keys, "target": list(target), "p": probs,
            "probabilities": dict(zip(keys, probs)), "predicted_indices": selected,
            "predicted_keys": [keys[i] for i in selected], "predicted_count": len(selected),
            "gold_count": sum(target), "threshold": threshold, "decoder": method,
            "bce": bce, "bce_origin": bce_origin,
            "protocol_violation": (mode == "single" and len(selected) != 1)
                                  or (not allow_empty and not selected),
            "oracle_gold_count": _oracle_ranking(target, scores),
        }
        for name in ("group_id", "question_id", "original_idx"):
            if name in row:
                record[name] = row[name]
        for name in ("logits", "candidate_logits", "input_tokens"):
            if name in item:
                record[name] = item[name]
        if item.get("decoder") == "set_map" and "count_probs" not in item:
            raise ValueError(f"{ident}: a structured set_map prediction requires count_probs")
        if "count_probs" in item:
            counts = _numbers(item["count_probs"], k + 1, ident, "count_probs", probabilities=True)
            if not math.isclose(sum(counts), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{ident}: count_probs must sum to one")
            if item.get("decoder") == "set_map":
                support = {1} if mode == "single" else set(range(0 if allow_empty else 1, k + 1))
                if any(probability > 1e-6 for count, probability in enumerate(counts) if count not in support):
                    raise ValueError(f"{ident}: structured count_probs violates the declared task protocol")
                expected_count = sum(count * probability for count, probability in enumerate(counts))
                if abs(sum(probs) - expected_count) > 1e-6 * k:
                    raise ValueError(f"{ident}: structured membership/count expectations disagree")
            record["count_probs"] = counts
            record["count_mode"] = max(range(k + 1), key=lambda i: (counts[i], -i))
        result.append(record)
    return result


def _report(records):
    n = len(records)
    tp = fp = fn = tn = exact = count_correct = violations = 0
    count_error = expected_count_error = bce_sum = 0.0
    oracle_exact = oracle_ties = oracle_strict = 0
    oracle_expected = 0.0
    count_nll = count_brier = count_mode_error = count_mean_error = 0.0
    count_mode_correct = count_rows = 0
    for record in records:
        selected = set(record["predicted_indices"])
        gold = {i for i, value in enumerate(record["target"]) if value}
        tp += len(selected & gold)
        fp += len(selected - gold)
        fn += len(gold - selected)
        tn += record["n_options"] - len(selected | gold)
        exact += selected == gold
        count_correct += len(selected) == len(gold)
        count_error += abs(len(selected) - len(gold))
        expected_count_error += abs(sum(record["p"]) - len(gold))
        violations += bool(record["protocol_violation"])
        bce_sum += record["bce"]
        diagnostic = record["oracle_gold_count"]
        oracle_exact += diagnostic["question_em"]
        oracle_ties += diagnostic["cutoff_tie"]
        oracle_expected += diagnostic["question_em_expected_random_ties"]
        oracle_strict += diagnostic["strict_separation_question_em"]
        if "count_probs" in record:
            counts = record["count_probs"]
            count_rows += 1
            count_nll -= math.log(max(PROBABILITY_EPSILON, counts[len(gold)]))
            count_brier += sum((p - int(i == len(gold))) ** 2 for i, p in enumerate(counts))
            count_mode_correct += record["count_mode"] == len(gold)
            count_mode_error += abs(record["count_mode"] - len(gold))
            count_mean_error += abs(sum(i * p for i, p in enumerate(counts)) - len(gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "n_questions": n, "n_candidates": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "question_em_count": exact, "question_em": exact / n if n else 0.0,
        "count_accuracy_count": count_correct, "count_accuracy": count_correct / n if n else 0.0,
        "count_mae": count_error / n if n else 0.0,
        "marginal_expected_count_mae": expected_count_error / n if n else 0.0,
        "protocol_violation_count": violations, "protocol_violation_rate": violations / n if n else 0.0,
        "bce": bce_sum / n if n else 0.0,
        "oracle_gold_count": {
            "uses_gold_count": True, "deployable": False,
            "description": "ranking diagnostic only: top gold_count options, not a model prediction",
            "tie_break": "descending score then ascending original option index; never gold labels",
            "score_basis": "candidate logits when supplied, otherwise probabilities",
            "question_em_count": oracle_exact, "question_em": oracle_exact / n if n else 0.0,
            "cutoff_tie_questions": oracle_ties,
            "question_em_expected_random_ties": oracle_expected / n if n else 0.0,
            "strict_separation_question_em": oracle_strict / n if n else 0.0,
        },
        "count_distribution_n_questions": count_rows,
        "count_nll": count_nll / count_rows if count_rows else None,
        "count_brier": count_brier / count_rows if count_rows else None,
        "count_mode_accuracy": count_mode_correct / count_rows if count_rows else None,
        "count_mode_mae": count_mode_error / count_rows if count_rows else None,
        "count_distribution_mean_mae": count_mean_error / count_rows if count_rows else None,
    }
    return result


def compute_metrics(rows, predictions, threshold=0.5, decoder="protocol"):
    """Micro candidate metrics, exact-set/count metrics, groups and an oracle diagnostic."""
    records = prediction_records(rows, predictions, threshold, decoder)
    if not records:
        raise ValueError("cannot score an empty dataset")
    grouped = {name: defaultdict(list) for name in ("source", "lang", "k", "k_bucket", "selection_mode")}
    for record in records:
        k = record["n_options"]
        values = {"source": record["source"], "lang": record["lang"], "k": k,
                  "k_bucket": "2-4" if k <= 4 else "5" if k == 5 else "6+",
                  "selection_mode": record["selection_mode"]}
        for name, value in values.items():
            grouped[name][str(value)].append(record)
    result = _report(records)
    result.update({
        "threshold": validate_threshold(threshold), "decoder": decoder,
        "threshold_fitted_on_input": False,
        "probability_clipping_epsilon": PROBABILITY_EPSILON,
        "bce_origins": dict(Counter(record["bce_origin"] for record in records)),
        "groups": {name: {key: _report(group) for key, group in sorted(values.items())}
                   for name, values in grouped.items()},
    })
    return result
