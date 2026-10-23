#!/usr/bin/env python3
"""Build coherent set targets from frozen candidate-membership teacher files.

Only the standard library is needed. Each valid chain defines an independent
Bernoulli product, conditioned on the task's allowed counts before mixing.
Manifest weights apply to teachers; chains share their own teacher's weight.
This is an explicit joint approximation, not an empirical teacher joint law.

Example::

    python -I harness/build_multiteacher_targets.py --input data/train.jsonl \
        --manifest teachers.json --output targets/train.jsonl

The manifest contains {"teachers": [{"id": "direct", "path": "direct.jsonl",
"weight": 1.0}, ...]}; relative teacher paths resolve beside the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__:
    from .set_distillation import (
        TARGET_SCHEMA, allowed_counts, conditioned_bernoulli_stats,
        input_digest, validate_set_targets,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from set_distillation import (
        TARGET_SCHEMA, allowed_counts, conditioned_bernoulli_stats,
        input_digest, validate_set_targets,
    )


class UnavailableTeacher(ValueError):
    """A missing teacher row or a well-formed row with no usable chains."""


def _nonfinite_json(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _loads(value):
    return json.loads(value, parse_constant=_nonfinite_json)


def _identifier(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: id must be a nonempty string")
    return value


def _chain_id(value):
    return (isinstance(value, int) and not isinstance(value, bool) and value >= 0
            or isinstance(value, str) and bool(value.strip()))


def read_jsonl(path):
    """Read each file once, hashing the exact bytes used for parsing."""
    path = Path(path)
    digest, records, seen = hashlib.sha256(), [], set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                record = _loads(raw)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            ident = _identifier(record.get("id"), f"{path}:{line_number}")
            if ident in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {ident!r}")
            seen.add(ident)
            records.append(record)
    return records, digest.hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = _loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("teachers"), list):
        raise ValueError("manifest must contain a teachers list")
    if not manifest["teachers"]:
        raise ValueError("manifest must name at least one teacher")
    teachers, seen = [], set()
    for item in manifest["teachers"]:
        if not isinstance(item, dict):
            raise ValueError("each manifest teacher must be an object")
        ident = _identifier(item.get("id"), "manifest teacher")
        if ident in seen:
            raise ValueError(f"duplicate manifest teacher id {ident!r}")
        seen.add(ident)
        location = item.get("path")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"{ident}: teacher path must be a nonempty string")
        weight = item.get("weight", 1.0)
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError(f"{ident}: weight must be finite and nonnegative")
        teachers.append({"id": ident, "path": (path.parent / location).resolve(),
                         "weight": float(weight)})
    scale = max(item["weight"] for item in teachers)
    if scale == 0:
        raise ValueError("teacher weights must have positive total mass")
    total = math.fsum(item["weight"] / scale for item in teachers)
    for item in teachers:
        item["normalized_weight"] = (item["weight"] / scale) / total
    return teachers, hashlib.sha256(raw).hexdigest()


def _probabilities(values, length, context):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{context}: expected {length} probabilities")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError(f"{context}: probabilities must be finite and in [0, 1]")
    return [float(v) for v in values]


def _validate_metadata(metadata, included, context, truncated_policy):
    if not isinstance(metadata, dict) or metadata.get("included") is not included:
        raise ValueError(f"{context}: malformed chain inclusion metadata")
    if not _chain_id(metadata.get("chain")):
        raise ValueError(f"{context}: invalid chain id in metadata")
    if not isinstance(metadata.get("truncated"), bool):
        raise ValueError(f"{context}: explicit boolean truncated metadata is required")
    if metadata.get("finish_reason") == "length" and not metadata["truncated"]:
        raise ValueError(f"{context}: length-limited chain must be marked truncated")
    if included and metadata["truncated"] and truncated_policy != "include":
        raise ValueError(f"{context}: truncated chain included without explicit include policy")
    if included and metadata.get("finish_reason") in {"missing", "error", "abort", "prompt_too_long"}:
        raise ValueError(f"{context}: unusable chain marked included")


def validate_teacher_record(record, row, teacher_id):
    """Return validated raw membership vectors and preserved provenance."""
    context = f"{row['id']} / {teacher_id}"
    if record.get("teacher_id") != teacher_id:
        raise ValueError(f"{context}: teacher_id differs from manifest")
    if (record.get("q_semantics") != "candidate_membership"
            or record.get("q_transform") != "membership"
            or record.get("readout") != "candidate_yes_no_v1"):
        raise ValueError(f"{context}: diagnostic or unsupported teacher targets are forbidden")
    for name in ("teacher_view", "checkpoint"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"{context}: missing {name}")
    keys = [option["key"] for option in row["options"]]
    if record.get("option_keys") != keys:
        raise ValueError(f"{context}: ordered option_keys mismatch")
    if "n_options" in record and (isinstance(record["n_options"], bool)
                                   or not isinstance(record["n_options"], int)
                                   or record["n_options"] != len(keys)):
        raise ValueError(f"{context}: n_options mismatch")
    for name in ("selection_mode", "allow_empty"):
        expected = row.get(name, "multi" if name == "selection_mode" else False)
        if name in record and record[name] != expected:
            raise ValueError(f"{context}: {name} mismatch")
    for name in ("split", "source_split"):
        if name in record and record[name] != row["split"]:
            raise ValueError(f"{context}: teacher {name} mismatch")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{context}: missing provenance object")
    expected_digest = input_digest(row)
    checked = 0
    for container, name in ((record, "input_digest"), (record, "input_sha256"),
                            (provenance, "input_digest"), (provenance, "input_sha256")):
        if name in container:
            if container[name] != expected_digest:
                raise ValueError(f"{context}: input digest mismatch in {name}")
            checked += 1
    if not checked:
        raise ValueError(f"{context}: a canonical input digest is required")
    policy = provenance.get("truncated_policy")
    if policy not in ("exclude", "include"):
        raise ValueError(f"{context}: explicit truncated_policy must be exclude or include")
    if provenance.get("legacy_reasoning_cache_allowed") is True:
        raise ValueError(f"{context}: unverified legacy reasoning is forbidden")
    if ("option_alignment" in provenance
            and provenance["option_alignment"] != "input_option_order"):
        raise ValueError(f"{context}: unsupported option alignment")
    vectors, ids, metadata = (record.get(name) for name in
                              ("per_chain_membership", "per_chain_ids", "per_chain_metadata"))
    if not all(isinstance(value, list) for value in (vectors, ids, metadata)):
        raise ValueError(f"{context}: missing per-chain arrays")
    if len(ids) != len(vectors) or len(metadata) != len(vectors):
        raise ValueError(f"{context}: per-chain arrays are not aligned")
    if any(not _chain_id(value) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{context}: invalid or duplicate chain ids")
    if "valid_chains" in record and (isinstance(record["valid_chains"], bool)
                                     or not isinstance(record["valid_chains"], int)
                                     or record["valid_chains"] != len(vectors)):
        raise ValueError(f"{context}: valid_chains differs from chain array length")
    requested = record.get("n_chains", len(vectors))
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < len(vectors):
        raise ValueError(f"{context}: invalid requested chain count")
    if "missing_chains" in record and (isinstance(record["missing_chains"], bool)
                                       or not isinstance(record["missing_chains"], int)
                                       or record["missing_chains"] != requested - len(vectors)):
        raise ValueError(f"{context}: missing_chains count mismatch")
    for chain, item in zip(ids, metadata):
        _validate_metadata(item, True, context, policy)
        if item["chain"] != chain:
            raise ValueError(f"{context}: per-chain metadata order mismatch")
    excluded = record.get("excluded_chain_metadata", [])
    if not isinstance(excluded, list):
        raise ValueError(f"{context}: invalid excluded_chain_metadata")
    seen = set(ids)
    for item in excluded:
        _validate_metadata(item, False, context, policy)
        if item["chain"] in seen:
            raise ValueError(f"{context}: duplicate included/excluded chain id")
        seen.add(item["chain"])
    if len(seen) > requested or any(isinstance(chain, int) and chain >= requested for chain in seen):
        raise ValueError(f"{context}: chain metadata exceeds requested count")
    if not vectors:
        if record.get("q") is not None:
            raise ValueError(f"{context}: no chains but non-null q")
        raise UnavailableTeacher(f"{context}: no valid chains")
    values = [_probabilities(vector, len(keys), context) for vector in vectors]
    q = _probabilities(record.get("q"), len(keys), context)
    mean = [math.fsum(column) / len(values) for column in zip(*values)]
    if any(abs(a - b) > 1e-7 for a, b in zip(q, mean)):
        raise ValueError(f"{context}: q is inconsistent with per-chain membership mean")
    detail = {
        "teacher_id": teacher_id, "teacher_view": record["teacher_view"],
        "checkpoint": record["checkpoint"], "readout": record["readout"],
        "valid_chains": len(values), "requested_chains": requested,
        "per_chain_ids": ids, "per_chain_metadata": metadata,
        "excluded_chain_metadata": excluded, "provenance": provenance,
        "input_digest_verified": bool(checked),
    }
    return values, detail


def aggregate_row(row, teacher_records, teachers):
    support = allowed_counts(row)
    components, details, unavailable = [], [], []
    for teacher in teachers:
        record = teacher_records[teacher["id"]].get(row["id"])
        if record is None:
            unavailable.append(f"{row['id']} / {teacher['id']}: missing teacher row")
            continue
        try:
            vectors, detail = validate_teacher_record(record, row, teacher["id"])
        except UnavailableTeacher as error:
            unavailable.append(str(error))
            continue
        per_chain_weight = teacher["normalized_weight"] / len(vectors)
        for vector in vectors:
            q, rho = conditioned_bernoulli_stats(vector, support)
            components.append((per_chain_weight, q, rho))
        detail.update(weight=teacher["weight"],
                      normalized_weight=teacher["normalized_weight"],
                      per_chain_weight=per_chain_weight,
                      path=str(teacher["path"]), file_sha256=teacher["file_sha256"])
        details.append(detail)
    if unavailable:
        raise UnavailableTeacher("; ".join(unavailable))
    k = len(row["options"])
    q = [min(1.0, max(0.0, math.fsum(weight * values[j] for weight, values, _ in components)))
         for j in range(k)]
    rho = [min(1.0, max(0.0, math.fsum(weight * values[m] for weight, _, values in components)))
           for m in range(k + 1)]
    validate_set_targets(q, rho, support)
    return {
        "schema": TARGET_SCHEMA, "id": row["id"],
        "q_semantics": "set_membership", "q": q, "count_probs": rho,
        "allowed_counts": support, "option_keys": [o["key"] for o in row["options"]],
        "input_digest": input_digest(row), "source_split": row["split"],
        "source": row.get("source"), "lang": row.get("lang"),
        "selection_mode": row.get("selection_mode", "multi"),
        "allow_empty": row.get("allow_empty", False),
        "joint_semantics": "mixture_of_per_chain_bernoulli_products_conditioned_before_mixing",
        "teacher_provenance": details,
    }


def _stage_json(path, records, jsonl):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if jsonl:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(records, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def _commit(temporary, destination, overwrite):
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Unlike exists()+replace(), link() cannot clobber a concurrent writer.
        os.link(temporary, destination)
        temporary.unlink()


def build_targets(input_path, manifest_path, output_path=None, *, split="train",
                  missing="error", overwrite=False, validate_only=False, report_path=None):
    """Validate the full selected split before publishing an atomic JSONL file.

    --missing drop drops an entire question, preserving the declared mixture;
    it never removes a teacher from one row or substitutes zeros. Malformed
    records remain fatal under this option. All manifest teachers, including
    zero-weight entries, must supply valid records. Returns the audit summary.
    """
    if split not in ("train", "dev"):
        raise ValueError("only train or explicitly requested dev is allowed; test is forbidden")
    if missing not in ("error", "drop"):
        raise ValueError("missing policy must be error or drop")
    if output_path is None and not validate_only:
        raise ValueError("output is required unless validate_only is enabled")
    if validate_only and report_path is not None:
        raise ValueError("validate_only writes no files; omit report_path")
    input_path, manifest_path = Path(input_path).resolve(), Path(manifest_path).resolve()
    teachers, manifest_sha = load_manifest(manifest_path)
    input_rows, input_sha = read_jsonl(input_path)
    for row in input_rows:
        if row.get("split") not in ("train", "dev", "test"):
            raise ValueError(f"{row['id']}: an explicit train/dev/test split is required")
    rows = [row for row in input_rows if row["split"] == split]
    if not rows:
        raise ValueError(f"input has no {split} rows")
    for row in rows:
        options = row.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"{row['id']}: options must be a nonempty list")
        keys = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError(f"{row['id']}: option must be an object")
            keys.append(_identifier(option.get("key"), row["id"]))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{row['id']}: duplicate option keys")
        for name in ("state", "question"):
            if name not in row:
                raise ValueError(f"{row['id']}: missing input {name}")
        allowed_counts(row)
    sources = {input_path, manifest_path} | {teacher["path"] for teacher in teachers}
    output = Path(output_path).resolve() if output_path is not None else None
    report = (Path(report_path).resolve() if report_path is not None
              else output.with_suffix(".summary.json") if output is not None else None)
    if not validate_only:
        if output == report or output in sources or report in sources:
            raise ValueError("output/report paths must be distinct from each other and all inputs")
        for destination in (output, report):
            if destination.exists() and (not overwrite or destination.is_dir()):
                raise FileExistsError(f"refusing to overwrite {destination}")
    selected_ids = {row["id"] for row in rows}
    teacher_records, teacher_audit = {}, []
    for teacher in teachers:
        records, teacher["file_sha256"] = read_jsonl(teacher["path"])
        teacher_records[teacher["id"]] = {record["id"]: record for record in records}
        teacher_audit.append({
            **{key: str(value) if isinstance(value, Path) else value for key, value in teacher.items()},
            "file_rows": len(records),
            "selected_ids_present": sum(record["id"] in selected_ids for record in records),
            "unused_file_rows": sum(record["id"] not in selected_ids for record in records),
        })
    outputs, dropped = [], []
    for row in rows:
        try:
            record = aggregate_row(row, teacher_records, teachers)
        except UnavailableTeacher as error:
            if missing == "error":
                raise
            dropped.append({"id": row["id"], "reason": str(error), "source": row.get("source")})
            continue
        record["input_file_sha256"] = input_sha
        record["manifest_sha256"] = manifest_sha
        outputs.append(record)
    if not outputs:
        raise ValueError("no covered rows remain; refusing to publish an empty target file")
    for audit in teacher_audit:
        details = [next(item for item in record["teacher_provenance"]
                        if item["teacher_id"] == audit["id"]) for record in outputs]
        audit.update({
            "retained_valid_chains": sum(item["valid_chains"] for item in details),
            "retained_requested_chains": sum(item["requested_chains"] for item in details),
            "included_truncated_chains": sum(meta["truncated"] for item in details
                                              for meta in item["per_chain_metadata"]),
            "excluded_truncated_chains": sum(meta["truncated"] for item in details
                                              for meta in item["excluded_chain_metadata"]),
            "retained_rows_without_input_digest": sum(not item["input_digest_verified"] for item in details),
        })
    by_source = {}
    for source in sorted({str(row.get("source")) for row in rows}):
        selected = sum(str(row.get("source")) == source for row in rows)
        retained = sum(str(row.get("source")) == source for row in outputs)
        by_source[source] = {"selected": selected, "retained": retained,
                             "dropped": selected - retained, "coverage": retained / selected}
    summary = {
        "schema": "decision_set_target_audit_v1", "target_schema": TARGET_SCHEMA,
        "source_split": split, "purpose": "training" if split == "train" else "teacher_quality_audit",
        "input": {"path": str(input_path), "sha256": input_sha,
                  "rows": len(input_rows), "rows_by_split": dict(Counter(row["split"] for row in input_rows))},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "weighting": "fixed normalized teacher weights; uniform valid-chain weights within each teacher",
        "conditioning": "each Bernoulli product conditioned on protocol counts before mixture",
        "joint_approximation": "conditional independence within each chain; not empirical teacher joint",
        "missing_policy": missing, "selected_rows": len(rows), "retained_rows": len(outputs),
        "dropped_rows": len(dropped), "coverage": len(outputs) / len(rows),
        "coverage_by_source": by_source, "dropped": dropped, "teachers": teacher_audit,
        "output": str(output) if output is not None and not validate_only else None,
        "validate_only": validate_only,
    }
    if not validate_only:
        staged = []
        report_committed = False
        try:
            staged.append(_stage_json(output, outputs, True))
            staged.append(_stage_json(report, summary, False))
            # Publish the data last; its existence marks a completed build.
            _commit(staged[1], report, overwrite)
            report_committed = True
            _commit(staged[0], output, overwrite)
        except BaseException:
            if report_committed and not overwrite:
                report.unlink(missing_ok=True)
            raise
        finally:
            for temporary in staged:
                temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="default: output stem + .summary.json")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--missing", choices=("error", "drop"), default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate and print audit; write no files")
    args = parser.parse_args(argv)
    try:
        summary = build_targets(args.input, args.manifest, args.output, split=args.split,
                                missing=args.missing, overwrite=args.overwrite,
                                validate_only=args.validate_only, report_path=args.report)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
