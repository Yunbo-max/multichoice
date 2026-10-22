#!/usr/bin/env python3
"""Audit matched next-student experiments, not teacher-accuracy proxies.

Consumes native training/evaluation artifacts and recomputes held-out metrics.
The common initialization includes the count head when present. Identity strings
are pinned operator attestations; this tool does not hash model weight shards.
Use --allow-planned only to inspect an entirely unexecuted manifest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiselect_metrics import read_dataset, read_jsonl, compute_metrics, prediction_records
from plan_self_distillation_rounds import (
    UTILITY_SCHEMA, check_split_separation, pinned_identity, sha256_file, write_json_exclusive,
)
from set_distillation import attach_set_targets, input_digest, read_set_target_file
from audit_teacher_generations import _signature as teacher_protocol_signature


MATCHED_FIELDS = (
    "train_sha256", "dev_sha256", "modules_sha256", "init_checkpoint_id", "checkpoint_identity_kind",
    "objective", "set_loss", "count_weight", "count_target", "lambda_g", "graph", "lr",
    "global_batch", "epochs", "max_input_tokens", "max_batch_tokens", "eval_every_frac", "max_steps",
    "limit", "limit_dev", "world_size", "planned_steps", "stop_step", "start_step", "grad_clip",
    "lora", "lora_r", "lora_alpha", "gold_weight", "optimizer", "schedule",
)
EVAL_MATCHED_FIELDS = ("input_sha256", "expected_split", "decoder", "threshold",
                       "max_input_tokens", "max_batch_tokens", "batch_size", "mode")


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _resolve(base, value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected nonempty path")
    return (base / value).resolve()


def _require_fields(value, names, context):
    for name in names:
        if name not in value:
            raise ValueError(f"{context}: missing {name}")


def _check_hash(path, expected, field):
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{field}: current file does not match recorded SHA256")


def _evidence_paths(base, run):
    training = _resolve(base, run.get("training_dir"), "training_dir")
    evaluation = _resolve(base, run.get("evaluation_dir"), "evaluation_dir")
    return {"training_config": training / "training_config.json",
            "training_complete": training / "training_complete.json",
            "evaluation_config": evaluation / "evaluation_config.json",
            "metrics": evaluation / "metrics.json", "predictions": evaluation / "predictions.jsonl"}


def _check_metrics(saved, actual, name):
    # Recompute from aligned predictions; a summary number alone is not evidence.
    for key in ("n_questions", "question_em", "question_em_count", "f1", "bce", "count_nll", "count_brier"):
        if key not in saved:
            raise ValueError(f"{name}: metrics missing {key}")
        a, b = saved[key], actual[key]
        if a is None or b is None:
            valid = a is None and b is None
        else:
            valid = isinstance(a, (int, float)) and math.isfinite(a) and math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8)
        if not valid:
            raise ValueError(f"{name}: saved metrics disagree with recomputed {key}")


def _load_run(base, run, rows, input_path):
    evidence = _evidence_paths(base, run)
    train_cfg, complete, eval_cfg, saved = (_read(evidence[key]) for key in
                                          ("training_config", "training_complete", "evaluation_config", "metrics"))
    _require_fields(train_cfg, MATCHED_FIELDS + ("seed", "target_mode", "resume", "loaded_model", "model", "targets"), run["id"])
    _require_fields(eval_cfg, EVAL_MATCHED_FIELDS + ("checkpoint_id", "checkpoint_identity_kind", "model"), run["id"])
    checkpoint_id = pinned_identity(run.get("checkpoint_id"))
    pinned_identity(train_cfg["init_checkpoint_id"])
    if train_cfg["checkpoint_identity_kind"] != "explicit" or eval_cfg["checkpoint_identity_kind"] != "explicit":
        raise ValueError(f"{run['id']}: explicit checkpoint identities are required")
    if eval_cfg["checkpoint_id"] != checkpoint_id:
        raise ValueError(f"{run['id']}: evaluated checkpoint identity differs from manifest")
    if train_cfg["seed"] != run["seed"]:
        raise ValueError(f"{run['id']}: seed mismatch")
    if (train_cfg["resume"] is not None or train_cfg["start_step"] != 0
            or train_cfg["loaded_model"] != train_cfg["model"]):
        raise ValueError(f"{run['id']}: probes must start from common --model with fresh optimizer, not --resume")
    if train_cfg["limit"] or train_cfg["limit_dev"] or train_cfg["lora"]:
        raise ValueError(f"{run['id']}: audit requires full declared data and the full-rank protocol")
    if (train_cfg["objective"] != "set" or train_cfg["count_target"] != "teacher"
            or train_cfg["lambda_g"] != 0 or train_cfg["graph"] != "none"):
        raise ValueError(f"{run['id']}: unexpected readout/loss family")
    steps = train_cfg["max_steps"]
    if (type(steps) is not int or steps <= 0 or train_cfg["eval_every_frac"] != 1.0
            or train_cfg["stop_step"] != steps or train_cfg["planned_steps"] < steps
            or complete.get("steps") != steps or complete.get("start_step") != 0
            or complete.get("selected", {}).get("step") != steps):
        raise ValueError(f"{run['id']}: terminal checkpoint and actual update budget must match")
    training_dir = evidence["training_config"].parent
    if Path(eval_cfg["model"]).resolve() != (training_dir / "final").resolve():
        raise ValueError(f"{run['id']}: evaluation must load this run's final export")
    if eval_cfg["mode"] != "checkpoint-evaluation" or eval_cfg.get("threshold_fitted_on_input") is not False:
        raise ValueError(f"{run['id']}: native frozen-rule checkpoint evaluation required")
    if eval_cfg["decoder"] != "provided" or eval_cfg["expected_split"] != "dev":
        raise ValueError(f"{run['id']}: matched exploratory probes require provided set decoder on dev")
    _check_hash(input_path, eval_cfg["input_sha256"], "evaluation input")
    train_path = Path(train_cfg["train"])
    dev_path = Path(train_cfg["dev"])
    _check_hash(train_path, train_cfg["train_sha256"], "training input")
    _check_hash(dev_path, train_cfg["dev_sha256"], "validation input")
    _check_hash(Path(train_cfg["modules"]), train_cfg["modules_sha256"], "trainable module list")
    if dev_path.resolve() != input_path.resolve():
        raise ValueError(f"{run['id']}: evaluation data differs from declared probe dev")
    train_rows = read_dataset(train_path, "train")
    check_split_separation(train_rows, rows)
    teacher_recipe = None
    if run["arm"] == "gold":
        if (train_cfg["target_mode"] != "gold" or train_cfg.get("target_file") is not None
                or run.get("teacher_checkpoint_id") is not None or run.get("teacher_targets") is not None):
            raise ValueError(f"{run['id']}: gold reference contains a teacher treatment")
    else:
        teacher_id = pinned_identity(run.get("teacher_checkpoint_id"))
        target_path = _resolve(base, run.get("teacher_targets"), "teacher_targets")
        if train_cfg["target_mode"] != "file" or Path(train_cfg.get("target_file", "")).resolve() != target_path:
            raise ValueError(f"{run['id']}: teacher target path mismatch")
        _check_hash(target_path, train_cfg.get("target_file_sha256"), "teacher targets")
        targets, errors = read_set_target_file(target_path)
        if errors:
            raise ValueError("; ".join(errors[:3]))
        errors, target_stats = attach_set_targets(train_rows, targets, train_cfg["gold_weight"], "error")
        if errors or target_stats["extra_ids"]:
            raise ValueError(f"{run['id']}: teacher targets do not cover exactly the training cohort: {errors[:3]}")
        for target in targets.values():
            if target.get("source_split") != "train":
                raise ValueError(f"{run['id']}: teacher targets must explicitly originate from train")
            provenance = target.get("teacher_provenance")
            if not isinstance(provenance, list) or not provenance:
                raise ValueError(f"{run['id']}: targets lack teacher provenance")
            current_recipe = {}
            for component in provenance:
                if (component.get("checkpoint") != teacher_id
                        or component.get("provenance", {}).get("checkpoint_identity_kind") != "explicit"):
                    raise ValueError(f"{run['id']}: target checkpoint identity contamination")
                if any(component["provenance"].get(field) is True for field in ("uses_gold_answers", "gold_in_prompt")):
                    raise ValueError(f"{run['id']}: gold-conditioned teacher cannot enter this matched protocol")
                logical_id = component.get("teacher_id")
                weight = component.get("normalized_weight")
                if (not isinstance(logical_id, str) or not logical_id or logical_id in current_recipe
                        or isinstance(weight, bool) or not isinstance(weight, (int, float))
                        or not math.isfinite(weight) or not 0 <= weight <= 1):
                    raise ValueError(f"{run['id']}: missing/invalid teacher mixture identity or weight")
                signature, known = teacher_protocol_signature({**component, "id": target["id"],
                                                              "n_chains": component.get("requested_chains")})
                if not known:
                    raise ValueError(f"{run['id']}: teacher generation settings are incomplete; regenerate matched targets")
                current_recipe[logical_id] = {"normalized_weight": weight, "protocol": signature}
            if not math.isclose(sum(item["normalized_weight"] for item in current_recipe.values()), 1., abs_tol=1e-8):
                raise ValueError(f"{run['id']}: teacher weights are not normalized")
            if teacher_recipe is None:
                teacher_recipe = current_recipe
            elif teacher_recipe != current_recipe:
                raise ValueError(f"{run['id']}: teacher recipe varies across training rows")
        if train_cfg["targets"].get("missing", 0) or train_cfg["gold_weight"] >= 1:
            raise ValueError(f"{run['id']}: missing targets or absent teaching treatment")
    predictions = read_jsonl(evidence["predictions"])
    by_id = {row["id"]: row for row in rows}
    for item in predictions:
        row = by_id.get(item.get("id"))
        if (row is None or item.get("input_digest") != input_digest(row)
                or item.get("option_keys") != [option["key"] for option in row["options"]]):
            raise ValueError(f"{run['id']}: prediction input identity/order mismatch")
        if (item.get("checkpoint_id") != checkpoint_id
                or item.get("provenance", {}).get("checkpoint_identity_kind") != "explicit"):
            raise ValueError(f"{run['id']}: prediction checkpoint identity mismatch")
        # Avoid trusting a supplied diagnostic loss over probability recomputation.
        item.pop("bce", None)
        item.pop("logits", None)
    recomputed = compute_metrics(rows, predictions, eval_cfg["threshold"], decoder="provided")
    _check_metrics(saved, recomputed, run["id"])
    records = prediction_records(rows, predictions, eval_cfg["threshold"], decoder="provided")
    correctness = {item["id"]: int(set(item["predicted_indices"]) == {i for i, y in enumerate(item["target"]) if y})
                   for item in records}
    return {"id": run["id"], "arm": run["arm"], "seed": run["seed"],
            "teacher_checkpoint_id": run.get("teacher_checkpoint_id"), "checkpoint_id": checkpoint_id,
            "training_config": train_cfg, "evaluation_config": eval_cfg, "metrics": recomputed,
            "teacher_recipe": teacher_recipe,
            "correctness": correctness,
            "evidence": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in evidence.items()}}


def _paired(left, right):
    """Right minus left; negative BCE/count-NLL changes are improvements."""
    ids = sorted(left["correctness"])
    if ids != sorted(right["correctness"]):
        raise ValueError("paired correctness cohorts differ")
    wins = sum(not left["correctness"][key] and right["correctness"][key] for key in ids)
    losses = sum(left["correctness"][key] and not right["correctness"][key] for key in ids)
    return {"seed": right["seed"], "n_questions": len(ids), "improved_questions": wins,
            "worsened_questions": losses, "unchanged_questions": len(ids) - wins - losses,
            "delta_question_em": (wins - losses) / len(ids),
            "delta_bce": right["metrics"]["bce"] - left["metrics"]["bce"],
            "delta_count_nll": (None if left["metrics"]["count_nll"] is None or right["metrics"]["count_nll"] is None
                                else right["metrics"]["count_nll"] - left["metrics"]["count_nll"])}


def _aggregate(pairs):
    values = [item["delta_question_em"] for item in pairs]
    return {"n_seeds": len(pairs), "mean_delta_question_em": statistics.mean(values),
            "sample_sd_delta_question_em": statistics.stdev(values) if len(values) > 1 else None,
            "mean_delta_bce": statistics.mean(item["delta_bce"] for item in pairs),
            "paired_by_seed": pairs, "significance_test": "not performed; questions may share groups"}


def audit(manifest_path, *, allow_planned=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    if manifest.get("schema") != UTILITY_SCHEMA:
        raise ValueError("unsupported teaching utility manifest schema")
    runs, contract = manifest.get("runs"), manifest.get("contract", {})
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("manifest needs a gold reference and at least one teacher run")
    identifiers, pairs = set(), set()
    for run in runs:
        if (not isinstance(run, dict) or not isinstance(run.get("id"), str) or not run["id"]
                or run["id"] in identifiers or run.get("arm") not in ("gold", "teacher")
                or type(run.get("seed")) is not int or run["seed"] < 0):
            raise ValueError("invalid or duplicate run identity/arm/seed")
        identifiers.add(run["id"])
        key = (run["seed"], None if run["arm"] == "gold" else run.get("teacher_key"))
        if key in pairs or run["arm"] == "teacher" and not isinstance(run.get("teacher_key"), str):
            raise ValueError("duplicate/missing teacher treatment for a seed")
        pairs.add(key)
    base = manifest_path.parent
    expected = [_evidence_paths(base, run) for run in runs]
    found = sum(path.is_file() for evidence in expected for path in evidence.values())
    if found == 0 and allow_planned:
        return {"schema": "decision_teaching_utility_audit_v1", "status": "planned",
                "teaching_utility_measured": False, "n_planned_runs": len(runs),
                "reason": "no native training/evaluation evidence exists",
                "manifest_sha256": sha256_file(manifest_path)}
    missing = [str(path) for evidence in expected for path in evidence.values() if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete teaching-utility cohort; missing evidence: {missing[:5]}")
    input_path = _resolve(base, manifest.get("input"), "input")
    rows = read_dataset(input_path, "dev")
    loaded = [_load_run(base, run, rows, input_path) for run in runs]
    reference = loaded[0]
    for run in loaded[1:]:
        for field in MATCHED_FIELDS:
            if run["training_config"][field] != reference["training_config"][field]:
                raise ValueError(f"confounded comparison: {field} differs for {run['id']}")
        for field in EVAL_MATCHED_FIELDS:
            if run["evaluation_config"][field] != reference["evaluation_config"][field]:
                raise ValueError(f"confounded evaluation: {field} differs for {run['id']}")
    recipes = [run["teacher_recipe"] for run in loaded if run["arm"] == "teacher"]
    if recipes and any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError("confounded teacher treatment: views/weights/generation/readout settings differ")
    if contract:
        mapping = {"initialization_checkpoint_id": "init_checkpoint_id", "train_sha256": "train_sha256",
                   "dev_sha256": "dev_sha256", "modules_sha256": "modules_sha256", "planned_steps": "planned_steps",
                   "max_steps": "max_steps", "objective": "objective", "set_loss": "set_loss",
                   "count_weight": "count_weight", "gold_weight": "gold_weight"}
        for key, field in mapping.items():
            if key in contract and contract[key] != reference["training_config"][field]:
                raise ValueError(f"native evidence violates planned {key}")
    gold = {run["seed"]: run for run in loaded if run["arm"] == "gold"}
    teacher_ids = sorted({run["teacher_checkpoint_id"] for run in loaded if run["arm"] == "teacher"})
    if not gold or not teacher_ids:
        raise ValueError("need gold references and teacher treatments")
    teacher_runs = {ident: {run["seed"]: run for run in loaded if run["teacher_checkpoint_id"] == ident}
                    for ident in teacher_ids}
    if any(len(values) != sum(run["teacher_checkpoint_id"] == ident for run in loaded)
           for ident, values in teacher_runs.items()):
        raise ValueError("different teacher keys resolve to duplicate checkpoint treatments")
    seeds = sorted(gold)
    if any(sorted(values) != seeds for values in teacher_runs.values()):
        raise ValueError("teacher/gold arms must have identical seed coverage")
    if contract.get("probe_seeds") is not None and sorted(contract["probe_seeds"]) != seeds:
        raise ValueError("native seed coverage differs from plan")
    gains = {ident: _aggregate([_paired(gold[seed], teacher_runs[ident][seed]) for seed in seeds])
             for ident in teacher_ids}
    comparisons = [{"left_teacher": left, "right_teacher": right,
                    **_aggregate([_paired(teacher_runs[left][seed], teacher_runs[right][seed]) for seed in seeds])}
                   for left, right in itertools.combinations(teacher_ids, 2)]
    return {"schema": "decision_teaching_utility_audit_v1", "status": "measured_from_native_artifacts",
            "teaching_utility_measured": True, "manifest_sha256": sha256_file(manifest_path),
            "input_sha256": sha256_file(input_path), "matched_fields": list(MATCHED_FIELDS),
            "identity_policy": "pinned explicit checkpoint attestations; weight shard hashes not independently verified",
            "question_cohort": "identical full dev split for every run; no outcome-dependent filtering",
            "utility_definition": "next-student held-out accuracy change under matched initialization, seed, data, updates and decoder",
            "teacher_gain_over_gold": gains, "teacher_pairwise_comparisons": comparisons,
            "runs": [{key: value for key, value in run.items() if key not in ("training_config", "evaluation_config", "correctness")}
                     for run in loaded],
            "limitations": ["No claim of recursive degradation follows without paired teacher-generation evidence.",
                            "Exploratory dev outcomes; lock the protocol before a held-out test claim.",
                            "Seed-level variation is descriptive; no independent-question significance claim."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-planned", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit(args.manifest, allow_planned=args.allow_planned)
        if args.output:
            if args.output.resolve() == args.manifest.resolve():
                raise ValueError("output must not overwrite the input manifest")
            write_json_exclusive(args.output, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
