#!/usr/bin/env python3
"""Plan bounded fixed/refreshed self-distillation and matched teaching probes.

This tool writes an argv DAG and small manifests; it never launches training.
Future weight identities remain binding objects until a caller supplies pinned
identities with --bindings. Identities are operator attestations, not weight-file
hash verification. Every run starts via --model with a fresh optimizer/schedule.

Only train targets are built, with strict full coverage. The generation audit
reads raw dev membership files and can inspect existing partial outputs after a
readout exits nonzero because some rows have no valid chains. Such a diagnostic
does not mark the failed readout successful or unblock any training stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiselect_metrics import read_dataset
from multiselect_to_unified import overlap_audit
from set_distillation import input_digest

PLAN_SCHEMA = "decision_self_distillation_plan_v1"
UTILITY_SCHEMA = "decision_teaching_utility_manifest_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_identity(value):
    """Require a content-digest or repository revision shaped attestation."""
    if not isinstance(value, str) or not (
            re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            or re.fullmatch(r"(?:hf:)?[^\s@]+/[^\s@]+@[0-9a-f]{40,64}", value)):
        raise ValueError("checkpoint_id must be sha256:<64 lowercase hex> or repo/name@<40..64 hex revision>")
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _path(base, value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return (base / value).resolve()


def check_split_separation(train, dev):
    """Check IDs, inputs, and full connected groups (including legacy passages)."""
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("train/dev IDs overlap")
    if {input_digest(r) for r in train} & {input_digest(r) for r in dev}:
        raise ValueError("train/dev input content overlaps")
    groups = lambda rows: {(r["source"], str(r["group_id"])) for r in rows
                           if r.get("group_id") is not None}
    if groups(train) & groups(dev):
        raise ValueError("train/dev declared groups overlap")
    overlap = overlap_audit({"train": train, "dev": dev})
    if overlap["full_question"]["train_dev"] or overlap["group"]["train_dev"]:
        raise ValueError("train/dev full-question or connected passage/group overlap")


def validate_dag(stages):
    """Require a topological stage list with unique names and no future edges."""
    seen = set()
    for stage in stages:
        name = stage.get("id")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("missing or duplicate stage id")
        if any(parent not in seen for parent in stage.get("depends_on", [])):
            raise ValueError(f"{name}: cyclic, missing or non-topological dependency")
        if not isinstance(stage.get("argv"), list) or not stage["argv"]:
            raise ValueError(f"{name}: empty argv")
        seen.add(name)


def _resolve(value, bindings):
    if isinstance(value, dict) and set(value) == {"checkpoint_binding"}:
        key = value["checkpoint_binding"]
        return pinned_identity(bindings[key]) if key in bindings else value
    if isinstance(value, dict):
        return {key: _resolve(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, bindings) for item in value]
    return value


def build_plan(config_path, output_dir, bindings=None):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    name = config.get("experiment_id")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("experiment_id must be a safe nonempty identifier")
    root = _path(base_dir, config.get("repository", str(Path(__file__).resolve().parents[1])), "repository")
    run_root = _path(base_dir, config.get("run_root"), "run_root")
    train_path, dev_path = (_path(base_dir, config.get(key), key) for key in ("train", "dev"))
    train, dev = read_dataset(train_path, "train"), read_dataset(dev_path, "dev")
    check_split_separation(train, dev)
    initial = config.get("base_checkpoint", {})
    initial_path = _path(base_dir, initial.get("path"), "base_checkpoint.path")
    initial_identity = pinned_identity(initial.get("checkpoint_id"))
    modules = _path(base_dir, config.get("modules", str(root / "innerjev/configs/4B/modules.json")), "modules")
    modules_sha = sha256_file(modules)
    rounds = _integer(config.get("rounds", 2), "rounds")
    seeds = config.get("probe_seeds", [43, 44, 45])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe_seeds must be a nonempty unique list")
    seeds = [_integer(seed, "probe seed", 0) for seed in seeds]
    trajectory_seed = _integer(config.get("trajectory_seed", seeds[0]), "trajectory_seed", 0)
    train_cfg = config.get("training", {})
    max_steps = _integer(train_cfg.get("max_steps", 16), "max_steps")
    epochs = _integer(train_cfg.get("epochs", 1), "epochs")
    batch = _integer(train_cfg.get("global_batch", 64), "global_batch")
    world = _integer(train_cfg.get("world_size", 8), "world_size")
    if batch % world:
        raise ValueError("global_batch must be divisible by world_size")
    planned_steps = math.ceil(len(train) / batch) * epochs
    if max_steps > planned_steps:
        raise ValueError(f"max_steps={max_steps} exceeds available {planned_steps} optimizer updates")
    lr = _number(train_cfg.get("lr", 2e-6), "lr", minimum=1e-30)
    beta = _number(train_cfg.get("gold_weight", 0.25), "gold_weight", maximum=1)
    if beta == 1:
        raise ValueError("gold_weight=1 removes the teaching treatment")
    set_loss = train_cfg.get("set_loss", "set_ce")
    if set_loss not in ("set_ce", "independent_marginal", "induced_marginal"):
        raise ValueError("unsupported set_loss")
    count_weight = _number(train_cfg.get("count_weight", 1.0), "count_weight")
    if set_loss == "set_ce" and count_weight != 1:
        raise ValueError("set_ce requires count_weight=1")
    max_input = _integer(train_cfg.get("max_input_tokens", 8192), "max_input_tokens")
    max_batch = _integer(train_cfg.get("max_batch_tokens", 8192), "max_batch_tokens")
    teacher_cfg = config.get("teacher", {})
    views = teacher_cfg.get("views", ["direct", "verify"])
    if not isinstance(views, list) or not views or len(set(views)) != len(views) or any(v not in ("direct", "verify") for v in views):
        raise ValueError("teacher views must be a unique subset of direct,verify")
    samples = _integer(teacher_cfg.get("samples", 2), "teacher samples")
    teacher_seed = _integer(teacher_cfg.get("seed", 0), "teacher seed", 0)
    max_tokens = _integer(teacher_cfg.get("max_tokens", 1024), "teacher max_tokens")
    if max_tokens >= max_input:
        raise ValueError("teacher max_tokens must be below max_input_tokens")
    python = config.get("python", "python")
    generate_python = config.get("generate_python", python)
    torchrun = config.get("torchrun", "torchrun")
    if any(not isinstance(v, str) or not v for v in (python, generate_python, torchrun)):
        raise ValueError("interpreter commands must be nonempty strings")
    stages, manifests, checkpoints, targets, runs = [], {}, {}, {}, []
    checkpoints["base"] = {"path": str(initial_path), "checkpoint_id": initial_identity, "producer": None}

    def add_stage(ident, argv, dependencies=(), **extra):
        stages.append({"id": ident, "status": "planned", "depends_on": list(dependencies),
                       "cwd": str(root), "argv": argv, **extra})
        return ident

    def identity_for(key):
        return checkpoints[key]["checkpoint_id"]

    def add_teacher(key):
        if key in targets:
            return targets[key]
        cp = checkpoints[key]
        folder = run_root / "teachers" / key
        readout_deps = {"train": [], "dev": []}
        teacher_manifest = {split: {"teachers": []} for split in ("train", "dev")}
        for view in views:
            # View ID stays comparable across generations; checkpoint provenance
            # and cache paths carry the changing model identity separately.
            teacher_id = view
            for split, input_path in (("train", train_path), ("dev", dev_path)):
                reasoning = folder / view / f"{split}.reasoning.jsonl"
                readout = folder / view / f"{split}.membership.jsonl"
                common = ["--model", cp["path"], "--checkpoint-id", cp["checkpoint_id"],
                          "--teacher-id", teacher_id, "--teacher-view", view, "--input", str(input_path),
                          "--samples", str(samples), "--seed", str(teacher_seed), "--q-transform", "membership",
                          "--truncated-policy", "exclude", "--max-tokens", str(max_tokens),
                          "--max-model-len", str(max_input), "--max-input-tokens", str(max_input)]
                prefix = f"teacher.{key}.{view}.{split}"
                generate = add_stage(prefix + ".generate", [generate_python, "-I", str(root / "harness/sii_style_teacher.py"),
                                     "generate", *common, "--output", str(reasoning)],
                                     [cp["producer"]] if cp["producer"] else [])
                read = add_stage(prefix + ".readout", [python, "-I", str(root / "harness/sii_style_teacher.py"),
                                 "readout", *common, "--reasoning", str(reasoning), "--output", str(readout)], [generate])
                readout_deps[split].append(read)
                teacher_manifest[split]["teachers"].append({"id": teacher_id, "path": str(readout), "weight": 1.0})
        for split, input_path in (("train", train_path), ("dev", dev_path)):
            manifest_name = f"manifests/{key}.{split}.json"
            manifests[manifest_name] = teacher_manifest[split]
            if split == "dev":
                # The diagnostic must see missing-chain coverage. A strict dev
                # target build would fail on exactly that evidence, so there is
                # no dev target prerequisite (and no dev training-target file).
                targets[key]["dev_readout_producers"] = readout_deps[split]
                continue
            output = folder / f"{split}.targets.jsonl"
            build = add_stage(f"teacher.{key}.{split}.targets", [python, "-I", str(root / "harness/build_multiteacher_targets.py"),
                              "--input", str(input_path), "--manifest", str(output_dir / manifest_name),
                              "--split", split, "--missing", "error", "--output", str(output)], readout_deps[split],
                              purpose="training", requires_full_coverage=True)
            targets[key] = {"path": str(output), "producer": build}
        return targets[key]

    def add_student(key, init_key, teacher_key, seed, probe=False):
        cp = checkpoints[init_key]
        folder = run_root / ("probes" if probe else "trajectories") / key
        deps = [cp["producer"]] if cp["producer"] else []
        argv = [torchrun, "--standalone", "--nnodes", "1", "--nproc_per_node", str(world),
                str(root / "harness/train_multiselect.py"), "--train", str(train_path), "--dev", str(dev_path),
                "--model", cp["path"], "--init-checkpoint-id", cp["checkpoint_id"], "--modules", str(modules),
                "--output", str(folder), "--objective", "set", "--set-loss", set_loss,
                "--count-weight", str(count_weight), "--count-target", "teacher", "--gold-weight", str(beta),
                "--lr", str(lr), "--global-batch", str(batch), "--epochs", str(epochs),
                "--max-steps", str(max_steps), "--seed", str(seed), "--eval-every-frac", "1.0",
                "--max-input-tokens", str(max_input), "--max-batch-tokens", str(max_batch),
                "--lambda-g", "0", "--graph", "none"]
        if teacher_key is None:
            argv += ["--target", "gold"]
        else:
            target = add_teacher(teacher_key)
            deps.append(target["producer"])
            argv += ["--target", "file", "--target-file", target["path"], "--target-missing", "error"]
        stage = add_stage(f"student.{key}.train", argv, list(dict.fromkeys(deps)),
                          optimizer_state="fresh", initialization=init_key, teacher=teacher_key,
                          updates=max_steps, checkpoint_selection="terminal_only")
        checkpoints[key] = {"path": str(folder / "final"),
                            "checkpoint_id": {"checkpoint_binding": key}, "producer": stage}
        eval_folder = folder / "eval_dev"
        evaluate = add_stage(f"student.{key}.eval", [python, "-I", str(root / "harness/eval_multiselect.py"),
                  "--input", str(dev_path), "--expected-split", "dev", "--model", str(folder / "final"),
                  "--checkpoint-id", identity_for(key), "--decoder", "provided", "--output", str(eval_folder),
                  "--max-input-tokens", str(max_input), "--max-batch-tokens", str(max_batch)], [stage])
        checkpoints[key]["evaluation_producer"] = evaluate
        checkpoints[key]["direct_predictions"] = str(eval_folder / "predictions.jsonl")
        if probe:
            runs.append({"id": key, "arm": "gold" if teacher_key is None else "teacher", "seed": seed,
                         "teacher_key": teacher_key, "teacher_checkpoint_id": None if teacher_key is None else identity_for(teacher_key),
                         "checkpoint_id": identity_for(key), "training_dir": str(folder), "evaluation_dir": str(eval_folder),
                         "teacher_targets": None if teacher_key is None else targets[teacher_key]["path"]})

    base_eval_folder = run_root / "base" / "eval_dev"
    base_eval = add_stage("checkpoint.base.eval", [python, "-I", str(root / "harness/eval_multiselect.py"),
                         "--input", str(dev_path), "--expected-split", "dev", "--model", str(initial_path),
                         "--checkpoint-id", initial_identity, "--decoder", "auto", "--output", str(base_eval_folder),
                         "--max-input-tokens", str(max_input), "--max-batch-tokens", str(max_batch)])
    checkpoints["base"]["evaluation_producer"] = base_eval
    checkpoints["base"]["direct_predictions"] = str(base_eval_folder / "predictions.jsonl")
    add_teacher("base")
    for arm in ("fixed", "refreshed"):
        previous = "base"
        for number in range(1, rounds + 1):
            key = f"{arm}.r{number}"
            add_student(key, previous, "base" if arm == "fixed" else previous, trajectory_seed)
            previous = key
    # Audit/probe the last generation as a teacher too; no extra student trajectory is implied.
    teacher_keys = ["base"] + [f"refreshed.r{number}" for number in range(1, rounds + 1)]
    for key in teacher_keys:
        add_teacher(key)
    generations = {"max_map_candidates": 12,
                   "generations": [{"id": key, "checkpoint_id": identity_for(key),
                                    "direct_predictions": checkpoints[key]["direct_predictions"],
                                    "teachers": str(output_dir / f"manifests/{key}.dev.json")}
                                   for key in teacher_keys]}
    manifests["generation_audit_manifest.json"] = generations
    add_stage("audit.teacher_generations", [python, "-I", str(root / "harness/audit_teacher_generations.py"),
              "--input", str(dev_path), "--manifest", str(output_dir / "generation_audit_manifest.json"),
              "--output", str(run_root / "audits/teacher_generations.json")],
              [stage for key in teacher_keys for stage in
               [checkpoints[key]["evaluation_producer"], *targets[key]["dev_readout_producers"]]],
              checkpoint_bindings=[key for key in teacher_keys if key != "base"],
              diagnostic_accepts_partial_inputs=True,
              failed_dependency_handling=("After a dev readout attempt, the diagnostic may inspect existing raw output files "
                                          "even if that readout exited nonzero due to rows_zero_chain. Missing or malformed "
                                          "files still fail validation. This does not mark failed stages successful and "
                                          "does not unblock train target building or student training."))
    for seed in seeds:
        add_student(f"gold.s{seed}", "base", None, seed, probe=True)
        for teacher_key in teacher_keys:
            add_student(f"from.{teacher_key}.s{seed}", "base", teacher_key, seed, probe=True)
    contract = {"initialization_checkpoint_id": initial_identity,
                "train_sha256": sha256_file(train_path), "dev_sha256": sha256_file(dev_path),
                "modules_sha256": modules_sha, "planned_steps": planned_steps, "max_steps": max_steps,
                "probe_seeds": seeds, "objective": "set", "set_loss": set_loss,
                "count_weight": count_weight, "gold_weight": beta, "eval_decoder": "provided",
                "checkpoint_selection": "terminal_only", "optimizer_state": "fresh"}
    utility = {"schema": UTILITY_SCHEMA, "experiment_id": name, "input": str(dev_path),
               "contract": contract, "runs": runs}
    manifests["utility_results_manifest.json"] = utility
    add_stage("audit.teaching_utility", [python, "-I", str(root / "harness/audit_teaching_utility.py"),
              "--manifest", str(output_dir / "utility_results_manifest.json"),
              "--output", str(run_root / "audits/teaching_utility.json")],
              [checkpoints[run["id"]]["evaluation_producer"] for run in runs],
              checkpoint_bindings=[run["id"] for run in runs] + [key for key in teacher_keys if key != "base"])
    plan = {"schema": PLAN_SCHEMA, "experiment_id": name, "status": "planned", "executed": False,
            "config_sha256": sha256_file(config_path), "contract": contract,
            "checkpoints": checkpoints, "stages": stages, "manifests": sorted(manifests),
            "identity_policy": "operator-attested pinned content/revision; future identities require post-export bindings; no automatic weight hash verification",
            "interpretation": "trajectory comparison and common-initialization teaching-utility probes are separate experiments",
            "dataset_audit": {"train_rows": len(train), "dev_rows": len(dev),
                              "checks": ["explicit split", "IDs", "exact input digests", "source-scoped connected group aliases", "full normalized MultiRC passages"],
                              "semantic_near_duplicate_check": False}}
    validate_dag(stages)
    bindings = bindings or {}
    if not isinstance(bindings, dict) or set(bindings) - set(checkpoints):
        raise ValueError("bindings contains unknown checkpoint keys")
    if "base" in bindings and bindings["base"] != initial_identity:
        raise ValueError("cannot rebind the base checkpoint")
    for value in bindings.values():
        pinned_identity(value)
    plan, manifests = _resolve(plan, bindings), _resolve(manifests, bindings)
    for stage in plan["stages"]:
        stage["argv_resolved"] = all(isinstance(value, str) for value in stage["argv"])
        stage["manifest_bindings_resolved"] = all(key in bindings for key in stage.get("checkpoint_bindings", []))
    return plan, manifests


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, help="JSON mapping of produced checkpoint keys to pinned identities")
    parser.add_argument("--validate-only", action="store_true", help="print plan; create nothing")
    args = parser.parse_args(argv)
    try:
        bindings = json.loads(args.bindings.read_text()) if args.bindings else None
        plan, manifests = build_plan(args.config, args.output_dir, bindings)
        if not args.validate_only:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("output-dir must be new or empty")
            for name, value in manifests.items():
                write_json_exclusive(args.output_dir / name, value)
            write_json_exclusive(args.output_dir / "plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
