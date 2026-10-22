#!/usr/bin/env python3
"""Compare direct decisions and reasoning-teacher quality on one held-out probe.

This is a proxy audit, not a test of teaching utility. All paired comparisons
use the same intersection of covered questions across every declared generation.
Teacher probabilities come from the full mixture of per-chain conditional
Bernoulli laws, not a product of averaged marginals. Exact mixture MAP is bounded
by K; probability metrics remain available above that bound.

Example: python -I harness/audit_teacher_generations.py --input dev.jsonl \
    --manifest generations.json --output teacher_generation_audit.json

Each manifest generation declares id, checkpoint_id, direct_predictions and
teachers (a build_multiteacher_targets.py manifest). Paths resolve beside this
manifest. An explicit checkpoint ID is an operator attestation, not an automatic
hash of model weights. No model is loaded and no training targets are written.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multiteacher_targets as builder
import multiselect_metrics as metrics
from set_distillation import allowed_counts, input_digest

SCHEMA = "decision_teacher_generation_audit_v1"
EPSILON = 1e-12
GENERATION_KEYS = ("seed", "temperature", "top_p", "top_k", "max_tokens")


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low - high))


def _logsum(values):
    result = -math.inf
    for value in values:
        result = _logadd(result, value)
    return result


def _component(vector, weight, support):
    """Log normalization of a Bernoulli product on the public count support."""
    include = [math.log(p) if p else -math.inf for p in vector]
    exclude = [math.log1p(-p) if p < 1 else -math.inf for p in vector]
    old = [0.0]
    for yes, no in zip(include, exclude):
        old = [_logadd(old[m] + no if m < len(old) else -math.inf,
                       old[m - 1] + yes if m else -math.inf)
               for m in range(len(old) + 1)]
    normalizer = _logsum(old[m] for m in support)
    if not math.isfinite(normalizer):
        raise ValueError("teacher assigns zero mass to the public count protocol")
    return {"include": include, "exclude": exclude,
            "log_weight": math.log(weight) if weight > 0 else -math.inf,
            "log_normalizer": normalizer,
            "log_count_probs": [value - normalizer if m in support else -math.inf
                                for m, value in enumerate(old)]}


def mixture_log_probability(components, selected):
    selected = set(selected)
    return min(0.0, _logsum(
        part["log_weight"] - part["log_normalizer"]
        + sum(yes if i in selected else no
              for i, (yes, no) in enumerate(zip(part["include"], part["exclude"])))
        for part in components))


def teacher_row_metrics(row, aggregate, components, max_map_candidates):
    q, counts, target = aggregate["q"], aggregate["count_probs"], row["target"]
    k, gold = len(q), [i for i, value in enumerate(target) if value]
    log_probability = mixture_log_probability(components, gold)
    count_log_probability = min(0.0, _logsum(
        part["log_weight"] + part["log_count_probs"][len(gold)] for part in components))
    clipped = [min(1 - EPSILON, max(EPSILON, p)) for p in q]
    report = {
        "q": q, "count_probs": counts,
        "membership_bce_clipped": -sum(math.log(p) if y else math.log1p(-p)
                                       for p, y in zip(clipped, target)) / k,
        "membership_brier": sum((p - y) ** 2 for p, y in zip(q, target)) / k,
        "count_gold_log_probability": count_log_probability if math.isfinite(count_log_probability) else None,
        "count_nll": -count_log_probability if math.isfinite(count_log_probability) else None,
        "count_nll_clipped": min(-math.log(EPSILON), -count_log_probability),
        "count_zero_gold_probability": count_log_probability == -math.inf,
        "count_brier": sum((p - int(m == len(gold))) ** 2 for m, p in enumerate(counts)),
        "gold_set_probability": math.exp(log_probability),
        "gold_set_log_probability": log_probability if math.isfinite(log_probability) else None,
        "gold_set_zero_probability": log_probability == -math.inf,
        "gold_set_nll": -log_probability if math.isfinite(log_probability) else None,
        "gold_set_nll_clipped": min(-math.log(EPSILON), -log_probability),
        "map_computed": k <= max_map_candidates,
        "map_predicted_indices": None, "map_question_em": None,
    }
    if report["map_computed"]:
        best_set, best_log = None, -math.inf
        # Deterministic ties: fewer options, then lexicographic option indices.
        for count in allowed_counts(row):
            for selected in itertools.combinations(range(k), count):
                value = mixture_log_probability(components, selected)
                if best_set is None or value > best_log:
                    best_set, best_log = list(selected), value
        report.update(map_predicted_indices=best_set, map_question_em=int(best_set == gold))
    return report


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _teacher_summary(records):
    keys = ("membership_bce_clipped", "membership_brier", "count_nll_clipped",
            "count_brier", "gold_set_probability", "gold_set_nll_clipped")
    result = {name: _mean([record[name] for record in records]) for name in keys}
    zero = sum(record["gold_set_zero_probability"] for record in records)
    count_zero = sum(record["count_zero_gold_probability"] for record in records)
    exact = [record["map_question_em"] for record in records if record["map_computed"]]
    result.update(n_questions=len(records), gold_set_zero_probability_count=zero,
                  count_zero_gold_probability_count=count_zero,
                  count_nll=None if count_zero else _mean([r["count_nll"] for r in records]),
                  gold_set_nll=None if zero else _mean([r["gold_set_nll"] for r in records]),
                  map_n_questions=len(exact), map_question_em=_mean(exact),
                  map_skipped_questions=len(records) - len(exact))
    return result


def _identity(record, checkpoint_id, context, *, direct=False):
    key = "checkpoint_id" if direct else "checkpoint"
    if record.get(key) != checkpoint_id:
        raise ValueError(f"{context}: {key} does not match the declared generation")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("checkpoint_identity_kind") != "explicit":
        raise ValueError(f"{context}: explicit checkpoint identity is required")
    if provenance.get("uses_gold_answers") is True or provenance.get("gold_in_prompt") is True:
        raise ValueError(f"{context}: gold-conditioned predictions are not a held-out probe")


def _signature(record):
    provenance = record.get("provenance", {})
    generation = []
    known = True
    for metadata in record.get("per_chain_metadata", []) + record.get("excluded_chain_metadata", []):
        config = metadata.get("generation")
        if config is None:
            if metadata.get("included"):
                known = False
            continue
        if not isinstance(config, dict) or any(key not in config for key in GENERATION_KEYS):
            known = False
            continue
        # A generation length cap also constrains available context. Read it
        # from the generation record, never assume the readout CLI used it.
        context = config.get("max_model_len")
        if type(context) is not int or context < 1:
            known = False
        generation.append({**{key: config[key] for key in GENERATION_KEYS}, "max_model_len": context})
    unique = {json.dumps(value, sort_keys=True, allow_nan=False) for value in generation}
    if len(unique) > 1:
        raise ValueError(f"{record['id']}: different generation settings within one teacher row")
    if not unique or not provenance.get("reasoning_prompt_version"):
        known = False
    binary = provenance.get("binary_tokens")
    if (not isinstance(binary, dict)
            or any(not isinstance(binary.get(key), str) or not binary[key] for key in ("yes_token", "no_token"))
            or any(type(binary.get(key)) is not int or binary[key] < 0 for key in ("yes_token_id", "no_token_id"))):
        known = False
    readout_version = provenance.get("readout_prompt_version")
    readout_budget = provenance.get("readout_max_input_tokens", provenance.get("max_input_tokens"))
    if not readout_version or type(readout_budget) is not int or readout_budget < 1:
        known = False
    return {
        "teacher_view": record.get("teacher_view"), "readout": record.get("readout"),
        "reasoning_prompt_version": provenance.get("reasoning_prompt_version"),
        "readout_prompt_version": readout_version, "binary_tokens": binary,
        "readout_max_input_tokens": readout_budget,
        "requested_chains": record.get("n_chains"),
        "truncated_policy": provenance.get("truncated_policy"),
        "generation": json.loads(next(iter(unique))) if unique else None,
    }, known


def _completion(records, total_questions):
    requested = sum(record.get("n_chains", 0) for record in records)
    included = [m for r in records for m in r.get("per_chain_metadata", [])]
    excluded = [m for r in records for m in r.get("excluded_chain_metadata", [])]
    all_metadata = included + excluded
    completed = sum(not m["truncated"] and m.get("finish_reason") not in
                    (None, "length", "missing", "error", "abort", "prompt_too_long") for m in all_metadata)
    return {
        "teacher_rows": len(records), "missing_teacher_rows": total_questions - len(records),
        "requested_chains_in_present_rows": requested, "usable_chains": len(included),
        "completed_reasoning_chains": completed,
        "completion_rate_in_present_rows": completed / requested if requested else None,
        "usable_chain_rate_in_present_rows": len(included) / requested if requested else None,
        "included_truncated_chains": sum(m["truncated"] for m in included),
        "excluded_truncated_chains": sum(m["truncated"] for m in excluded),
        "unreported_chain_slots": requested - len(all_metadata),
        "finish_reasons": dict(Counter(str(m.get("finish_reason")) for m in all_metadata)),
        "denominator_note": "Rates cover present rows; absent teacher rows are reported separately.",
    }


def _coverage_by_source(rows, valid):
    result = {}
    for source in sorted({row["source"] for row in rows}):
        chosen = [row for row in rows if row["source"] == source]
        retained = sum(row["id"] in valid for row in chosen)
        result[source] = {"total": len(chosen), "covered": retained, "coverage": retained / len(chosen)}
    return result


def _read_generation(item, base, rows, max_map_candidates):
    ident, checkpoint = item.get("id"), item.get("checkpoint_id")
    if not isinstance(ident, str) or not ident.strip() or not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("each generation needs nonempty id and checkpoint_id strings")
    for key in ("direct_predictions", "teachers"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"{ident}: {key} must be a path")
    by_id = {row["id"]: row for row in rows}
    direct_path, teacher_manifest = ((base / item[key]).resolve() for key in ("direct_predictions", "teachers"))
    direct, direct_hash = builder.read_jsonl(direct_path)
    if set(record["id"] for record in direct) - set(by_id):
        raise ValueError(f"{ident}: direct prediction IDs are outside the probe")
    for record in direct:
        row = by_id[record["id"]]
        _identity(record, checkpoint, f"{ident}/{row['id']}", direct=True)
        if record.get("input_digest") != input_digest(row):
            raise ValueError(f"{ident}/{row['id']}: direct input digest mismatch")
        if record.get("option_keys") != [o["key"] for o in row["options"]]:
            raise ValueError(f"{ident}/{row['id']}: direct option order mismatch")
        if record.get("threshold", 0.5) != 0.5 and record.get("decoder") != "set_map":
            raise ValueError(f"{ident}/{row['id']}: direct BCE audit requires the fixed 0.5 threshold")
    direct_ids = {record["id"] for record in direct}
    direct_rows = [row for row in rows if row["id"] in direct_ids]
    # Recompute BCE from supplied probabilities/logits, not a possibly stale metric.
    clean_direct = [{k: v for k, v in record.items() if k != "bce"} for record in direct]
    direct_records = metrics.prediction_records(direct_rows, clean_direct, decoder="auto")
    direct_by_id = {record["id"]: record for record in direct_records}
    teachers, teacher_hash = builder.load_manifest(teacher_manifest)
    teacher_records, completion, signatures, settings_known = {}, {}, {}, True
    sources = {direct_path, teacher_manifest}
    for teacher in teachers:
        sources.add(teacher["path"])
        records, teacher["file_sha256"] = builder.read_jsonl(teacher["path"])
        if set(record["id"] for record in records) - set(by_id):
            raise ValueError(f"{ident}/{teacher['id']}: teacher IDs are outside the probe")
        current_signatures = []
        for record in records:
            _identity(record, checkpoint, f"{ident}/{teacher['id']}/{record['id']}")
            # Validate malformed records even when another teacher is missing.
            try:
                builder.validate_teacher_record(record, by_id[record["id"]], teacher["id"])
            except builder.UnavailableTeacher:
                pass
            signature, known = _signature(record)
            settings_known = settings_known and known
            current_signatures.append(signature)
        common = current_signatures[0] if current_signatures else None
        for signature in current_signatures[1:]:
            # Missing chain metadata cannot manufacture known generation settings.
            for key in signature:
                if key == "generation" and (signature[key] is None or common[key] is None):
                    continue
                if signature[key] != common[key]:
                    raise ValueError(f"{ident}/{teacher['id']}: teacher protocol varies across probe rows")
        if common is None:
            settings_known = False
        elif common["generation"] is None:
            common = next((s for s in current_signatures if s["generation"] is not None), common)
        signatures[teacher["id"]] = {"normalized_weight": teacher["normalized_weight"], "protocol": common}
        teacher_records[teacher["id"]] = {record["id"]: record for record in records}
        completion[teacher["id"]] = _completion(records, len(rows))
    qualities, unavailable = {}, []
    for row in rows:
        try:
            aggregate = builder.aggregate_row(row, teacher_records, teachers)
        except builder.UnavailableTeacher as error:
            unavailable.append({"id": row["id"], "reason": str(error)})
            continue
        components = []
        for teacher in teachers:
            vectors = teacher_records[teacher["id"]][row["id"]]["per_chain_membership"]
            for vector in vectors:
                components.append(_component(vector, teacher["normalized_weight"] / len(vectors), allowed_counts(row)))
        qualities[row["id"]] = teacher_row_metrics(row, aggregate, components, max_map_candidates)
    return {
        "id": ident, "checkpoint_id": checkpoint, "direct": direct_by_id, "teacher": qualities,
        "signatures": signatures, "settings_known": settings_known, "sources": sources,
        "coverage": {"total": len(rows), "direct_rows": len(direct_by_id), "teacher_rows": len(qualities),
                     "joint_covered_rows": len(set(direct_by_id) & set(qualities)),
                     "teacher_coverage_by_source": _coverage_by_source(rows, qualities),
                     "missing_direct_ids": sorted(set(by_id) - set(direct_by_id)),
                     "unavailable_teacher_rows": unavailable},
        "completion": completion,
        "files": {"direct_predictions": {"path": str(direct_path), "sha256": direct_hash},
                  "teacher_manifest": {"path": str(teacher_manifest), "sha256": teacher_hash},
                  "teachers": [{"id": t["id"], "path": str(t["path"]), "sha256": t["file_sha256"]}
                               for t in teachers]},
    }


def _exact(record):
    return int(record["predicted_indices"] == [i for i, y in enumerate(record["target"]) if y])


def _paired(before, after, common_ids, settings_matched):
    records, direct_improved, direct_worsened, teacher_improved, teacher_worsened = [], 0, 0, 0, 0
    for ident in common_ids:
        old, new = before["teacher"][ident], after["teacher"][ident]
        direct_delta = _exact(after["direct"][ident]) - _exact(before["direct"][ident])
        map_delta = new["map_question_em"] - old["map_question_em"] if old["map_computed"] else None
        direct_improved += direct_delta == 1
        direct_worsened += direct_delta == -1
        teacher_improved += map_delta == 1
        teacher_worsened += map_delta == -1
        records.append({"id": ident, "direct_em_delta": direct_delta, "teacher_map_em_delta": map_delta,
                        **{f"teacher_{key}_delta": new[key] - old[key] for key in
                           ("gold_set_nll_clipped", "count_nll_clipped", "membership_brier")},
                        **{f"teacher_{key}_delta": new[key] - old[key]
                           if old[key] is not None and new[key] is not None else None
                           for key in ("gold_set_nll", "count_nll")}})
    direct_delta = _mean([r["direct_em_delta"] for r in records])
    nll_delta = _mean([r["teacher_gold_set_nll_clipped_delta"] for r in records])
    finite = {key: [r[f"teacher_{key}_delta"] for r in records if r[f"teacher_{key}_delta"] is not None]
              for key in ("gold_set_nll", "count_nll")}
    exact_nll_delta = _mean(finite["gold_set_nll"])
    # Use the unbounded proper score whenever defined on the full paired cohort.
    # If true zero events make extended-real means ambiguous, expose finite-only
    # pairs separately and label the full-cohort clipped fallback explicitly.
    use_exact = bool(records) and len(finite["gold_set_nll"]) == len(records)
    pattern_nll_delta = exact_nll_delta if use_exact else nll_delta
    map_delta = _mean([r["teacher_map_em_delta"] for r in records if r["teacher_map_em_delta"] is not None])
    descriptive_pattern = bool(records and direct_delta > 0 and (pattern_nll_delta > 0 or map_delta is not None and map_delta < 0))
    return {
        "before": before["id"], "after": after["id"], "n_paired_questions": len(common_ids),
        "direct_em_delta": direct_delta, "teacher_map_em_delta": map_delta,
        "teacher_gold_set_nll_clipped_delta": nll_delta,
        "teacher_count_nll_clipped_delta": _mean([r["teacher_count_nll_clipped_delta"] for r in records]),
        **{f"teacher_{key}_delta": _mean(values) for key, values in finite.items()},
        **{f"teacher_{key}_paired_finite_questions": len(values) for key, values in finite.items()},
        "proxy_nll_comparison_basis": "exact_full_cohort" if use_exact else "clipped_full_cohort",
        "direct_discordant": {"improved": direct_improved, "worsened": direct_worsened},
        "teacher_map_discordant": {"improved": teacher_improved, "worsened": teacher_worsened},
        "descriptive_direct_up_teacher_down_pattern": descriptive_pattern,
        "matched_teacher_settings_verified": settings_matched,
        "matched_proxy_pattern": descriptive_pattern if settings_matched else None,
        "teaching_utility_measured": False, "statistical_significance_tested": False,
        "per_question": records,
    }


def audit_generations(input_path, manifest_path, output_path=None, *, validate_only=False, overwrite=False):
    input_path, manifest_path = Path(input_path).resolve(), Path(manifest_path).resolve()
    rows = metrics.read_dataset(input_path, "dev")
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("the exploratory probe requires an explicit dev split on every row")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("generations"), list) or len(manifest["generations"]) < 2:
        raise ValueError("manifest must contain at least two generations")
    cap = manifest.get("max_map_candidates", 12)
    if type(cap) is not int or not 1 <= cap <= 16:
        raise ValueError("max_map_candidates must be an integer in [1, 16]")
    if any(not isinstance(item, dict) for item in manifest["generations"]):
        raise ValueError("each generation must be an object")
    generations = [_read_generation(item, manifest_path.parent, rows, cap) for item in manifest["generations"]]
    if len({g["id"] for g in generations}) != len(generations):
        raise ValueError("generation IDs must be unique")
    common = set(row["id"] for row in rows)
    for generation in generations:
        common &= set(generation["direct"]) & set(generation["teacher"])
    common_rows = [row for row in rows if row["id"] in common]
    common_ids = [row["id"] for row in common_rows]
    settings_match = all(g["settings_known"] for g in generations) and all(
        g["signatures"] == generations[0]["signatures"] for g in generations[1:])
    summaries = []
    for generation in generations:
        own_direct = [row for row in rows if row["id"] in generation["direct"]]
        direct_full = (metrics.compute_metrics(own_direct, list(generation["direct"].values()), decoder="auto")
                       if own_direct else None)
        direct_common = (metrics.compute_metrics(common_rows, [generation["direct"][i] for i in common_ids], decoder="auto")
                         if common_rows else None)
        summaries.append({key: generation[key] for key in ("id", "checkpoint_id", "coverage", "completion", "files")}
                         | {"teacher_settings": generation["signatures"],
                            "teacher_settings_known": generation["settings_known"],
                            "direct_on_available": direct_full, "direct_on_common": direct_common,
                            "teacher_on_available": _teacher_summary(list(generation["teacher"].values())),
                            "teacher_on_common": _teacher_summary([generation["teacher"][i] for i in common_ids]),
                            "per_question_teacher": generation["teacher"]})
    report = {
        "schema": SCHEMA, "purpose": "held_out_generation_quality_proxy_audit",
        "input": {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "split": "dev", "n_questions": len(rows)},
        "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(raw_manifest).hexdigest()},
        "common_cohort": {"ids": common_ids, "n_questions": len(common_ids), "coverage": len(common_ids) / len(rows),
                          "by_source": _coverage_by_source(rows, common)},
        "generations": summaries,
        "paired": [_paired(a, b, common_ids, settings_match) for a, b in zip(generations, generations[1:])],
        "teacher_settings_matched": settings_match, "max_map_candidates": cap,
        "probability_clipping_epsilon": EPSILON,
        "checkpoint_identity_verification": "explicit operator declaration; weight contents not independently hashed",
        "teacher_probability_semantics": "mixture of per-chain Bernoulli products conditioned before mixing",
        "teacher_map_ties": "fewest selected options, then lexicographic input option indices",
        "teaching_utility_measured": False,
        "limitations": [
            "Direct and reasoning quality are proxies; teaching utility requires matched next-generation students.",
            "Teacher MAP is the declared readout-law MAP, not correctness of generated prose.",
            "Known matched settings do not establish weight identity, semantic absence of gold leakage, or causal lineage.",
            "Unknown or different teacher settings disable the matched proxy-pattern flag.",
            "Available-cohort scores are descriptive; generation deltas use the identical common cohort only.",
            "No significance test, GPU run, model training, or reference-KL intervention is performed by this audit.",
        ],
    }
    if validate_only:
        if output_path is not None:
            raise ValueError("validate_only writes no files; omit output_path")
    else:
        if output_path is None:
            raise ValueError("output is required unless validate_only is enabled")
        output_path = Path(output_path).resolve()
        sources = {input_path, manifest_path} | set().union(*(g["sources"] for g in generations))
        if output_path in sources:
            raise ValueError("output must be distinct from all audit inputs")
        if output_path.exists() and (not overwrite or output_path.is_dir()):
            raise FileExistsError(f"refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
            if overwrite:
                os.replace(temporary, output_path)
            else:
                os.link(temporary, output_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_generations(args.input, args.manifest, args.output,
                                   validate_only=args.validate_only, overwrite=args.overwrite)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
