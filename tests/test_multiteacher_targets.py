"""CPU contracts for coherent mixtures, provenance and fail-closed publication."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_multiteacher_targets", ROOT / "harness/build_multiteacher_targets.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def row(ident="item-1", split="train", mode="multi", allow_empty=True):
    return {
        "id": ident, "source": "toy", "lang": "en", "split": split,
        "state": "Observed evidence", "question": "Which statements hold?",
        "options": [{"key": "a", "desc": "Alpha"}, {"key": "b", "desc": "Beta"}],
        "selection_mode": mode, "allow_empty": allow_empty, "target": [1, 0],
    }


def teacher_record(item, teacher_id, chains):
    digest = builder.input_digest(item)
    return {
        "id": item["id"], "teacher_id": teacher_id, "teacher_view": teacher_id,
        "checkpoint": "frozen-gold-checkpoint", "n_options": len(item["options"]),
        "readout": "candidate_yes_no_v1", "q_semantics": "candidate_membership",
        "q_transform": "membership", "option_keys": [o["key"] for o in item["options"]],
        "input_digest": digest, "n_chains": len(chains), "valid_chains": len(chains),
        "missing_chains": 0,
        "q": [math.fsum(column) / len(chains) for column in zip(*chains)] if chains else None,
        "per_chain_membership": chains, "per_chain_ids": list(range(len(chains))),
        "per_chain_metadata": [
            {"chain": i, "included": True, "truncated": False, "finish_reason": "stop",
             "reasoning_sha256": f"reasoning-{i}", "prompt_tokens": 20,
             "completion_tokens": 10, "generation": {"seed": i}}
            for i in range(len(chains))],
        "excluded_chain_metadata": [],
        "provenance": {"input_digest": digest, "input_sha256": digest,
                       "truncated_policy": "exclude", "option_alignment": "input_option_order",
                       "binary_tokens": {"yes_token": "Yes", "no_token": "No",
                                         "yes_token_id": 173, "no_token_id": 821}},
    }


class TargetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "input.jsonl"
        self.manifest = self.root / "teachers.json"
        self.output = self.root / "targets.jsonl"
        self.report = self.root / "targets.summary.json"
        self.rows = [row()]
        self.records = {
            "direct": [teacher_record(self.rows[0], "direct", [[0.8, 0.2]])],
            "verify": [teacher_record(self.rows[0], "verify", [[0.2, 0.8]] * 3)],
        }
        self.weights = {"direct": 1.0, "verify": 1.0}

    def write_jsonl(self, path, values):
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def prepare(self):
        self.write_jsonl(self.input, self.rows)
        manifest = []
        for ident, records in self.records.items():
            path = self.root / f"{ident}.jsonl"
            self.write_jsonl(path, records)
            manifest.append({"id": ident, "path": path.name, "weight": self.weights[ident]})
        self.manifest.write_text(json.dumps({"teachers": manifest}), encoding="utf-8")

    def build(self, **kwargs):
        self.prepare()
        return builder.build_targets(self.input, self.manifest, self.output, **kwargs)

    def output_rows(self):
        return [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]

    def add_row(self, item):
        self.rows.append(item)
        for ident in self.records:
            self.records[ident].append(teacher_record(item, ident, [[0.6, 0.4]]))

    def test_teacher_weights_do_not_scale_with_chain_count(self):
        summary = self.build()
        actual = self.output_rows()[0]
        for value in actual["q"]:
            self.assertAlmostEqual(value, 0.5)
        for expected, value in zip([0.16, 0.68, 0.16], actual["count_probs"]):
            self.assertAlmostEqual(value, expected)
        provenance = {item["teacher_id"]: item for item in actual["teacher_provenance"]}
        self.assertEqual(provenance["direct"]["per_chain_weight"], 0.5)
        self.assertAlmostEqual(provenance["verify"]["per_chain_weight"], 1 / 6)
        self.assertEqual(summary["coverage"], 1)
        self.assertEqual(actual["schema"], "decision_set_target_v1")
        self.assertEqual(actual["q_semantics"], "set_membership")
        self.assertEqual(actual["input_file_sha256"], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.report.read_text()), summary)
        builder.validate_set_targets(actual["q"], actual["count_probs"], actual["allowed_counts"])

    def test_nonuniform_fixed_teacher_weights(self):
        self.weights = {"direct": 3, "verify": 1}
        self.build()
        actual = self.output_rows()[0]
        self.assertAlmostEqual(actual["q"][0], 0.65)
        self.assertAlmostEqual(actual["q"][1], 0.35)

    def test_single_protocol_conditions_each_component_before_mixing(self):
        self.rows = [row(mode="single", allow_empty=False)]
        self.records = {
            "direct": [teacher_record(self.rows[0], "direct", [[0.8, 0.8]])],
            "verify": [teacher_record(self.rows[0], "verify", [[0.1, 0.2]])],
        }
        self.build()
        actual = self.output_rows()[0]
        expected = (0.5 + 0.08 / 0.26) / 2
        self.assertAlmostEqual(actual["q"][0], expected)
        self.assertNotAlmostEqual(actual["q"][0], (0.16 + 0.08) / (0.32 + 0.26))
        self.assertAlmostEqual(sum(actual["q"]), 1)
        self.assertEqual(actual["count_probs"], [0, 1, 0])
        self.assertEqual(actual["allowed_counts"], [1])

    def test_nonempty_protocol_and_count_identity(self):
        self.rows = [row(allow_empty=False)]
        self.records = {ident: [teacher_record(self.rows[0], ident, [[0.1, 0.2]])]
                        for ident in self.records}
        self.build()
        actual = self.output_rows()[0]
        self.assertEqual(actual["count_probs"][0], 0)
        self.assertAlmostEqual(actual["q"][0], 0.1 / 0.28)
        self.assertAlmostEqual(actual["q"][1], 0.2 / 0.28)
        self.assertAlmostEqual(sum(actual["q"]),
                               sum(m * mass for m, mass in enumerate(actual["count_probs"])))

    def test_diagnostic_targets_are_rejected_even_with_missing_drop(self):
        self.add_row(row("item-2"))
        self.records["direct"].pop()
        self.records["verify"][1]["q_semantics"] = "diagnostic_centered_sigmoid"
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            self.build(missing="drop")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_reordered_options_and_changed_input_are_rejected(self):
        self.records["direct"][0]["option_keys"].reverse()
        with self.assertRaisesRegex(ValueError, "option_keys"):
            self.build()
        self.records["direct"][0]["option_keys"].reverse()
        self.rows[0]["options"][0]["desc"] = "A different candidate"
        with self.assertRaisesRegex(ValueError, "digest"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_all_supplied_digests_are_checked_and_gold_is_excluded(self):
        self.rows[0]["target"] = [0, 1]
        self.rows[0]["gold_count"] = 99
        self.build(validate_only=True)
        self.records["direct"][0]["provenance"]["input_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "digest"):
            self.build(validate_only=True)

    def test_missing_digest_is_rejected_even_when_ids_and_options_match(self):
        record = self.records["direct"][0]
        record.pop("input_digest")
        record["provenance"].pop("input_digest")
        record["provenance"].pop("input_sha256")
        with self.assertRaisesRegex(ValueError, "canonical input digest is required"):
            self.build(validate_only=True)

    def test_missing_teacher_drops_whole_row_only_when_explicit(self):
        self.add_row(row("item-2"))
        self.records["verify"].pop()
        with self.assertRaisesRegex(builder.UnavailableTeacher, "missing teacher"):
            self.build()
        self.assertFalse(self.output.exists())
        summary = self.build(missing="drop")
        self.assertEqual([record["id"] for record in self.output_rows()], ["item-1"])
        self.assertEqual(summary["dropped_rows"], 1)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["dropped"][0]["id"], "item-2")
        self.assertEqual(summary["coverage_by_source"]["toy"]["coverage"], 0.5)
        self.assertAlmostEqual(self.output_rows()[0]["q"][0], 0.5)

    def test_no_valid_chains_are_missing_not_negative_labels(self):
        self.add_row(row("item-2"))
        self.records["verify"][1] = teacher_record(self.rows[1], "verify", [])
        with self.assertRaisesRegex(builder.UnavailableTeacher, "no valid chains"):
            self.build()
        summary = self.build(missing="drop")
        self.assertEqual(summary["retained_rows"], 1)
        self.assertIn("no valid chains", summary["dropped"][0]["reason"])

    def test_drop_refuses_empty_output(self):
        self.records["verify"] = []
        with self.assertRaisesRegex(ValueError, "empty target"):
            self.build(missing="drop")
        self.assertFalse(self.output.exists())

    def test_truncation_metadata_is_explicit_preserved_and_audited(self):
        record = self.records["direct"][0]
        record["per_chain_metadata"][0].update(truncated=True, finish_reason="length")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.build()
        record["provenance"]["truncated_policy"] = "include"
        record["n_chains"] = 2
        record["missing_chains"] = 1
        record["excluded_chain_metadata"] = [
            {"chain": 1, "included": False, "truncated": True,
             "finish_reason": "length", "exclusion_reason": "invalid_readout"}]
        summary = self.build()
        detail = self.output_rows()[0]["teacher_provenance"][0]
        self.assertEqual(detail["per_chain_metadata"], record["per_chain_metadata"])
        self.assertEqual(detail["excluded_chain_metadata"], record["excluded_chain_metadata"])
        self.assertEqual(summary["teachers"][0]["included_truncated_chains"], 1)
        self.assertEqual(summary["teachers"][0]["excluded_truncated_chains"], 1)

    def test_chain_alignment_and_probability_consistency_are_checked(self):
        original = copy.deepcopy(self.records["verify"][0])
        changes = [
            ("per_chain_ids", [0, 0, 2]),
            ("per_chain_metadata", original["per_chain_metadata"][:2]),
            ("q", [0.9, 0.1]),
            ("valid_chains", 2),
            ("per_chain_membership", [[0.2, 1.1]] * 3),
            ("per_chain_membership", [[False, 0.8]] * 3),
        ]
        for name, value in changes:
            with self.subTest(name=name, value=value):
                self.records["verify"][0] = copy.deepcopy(original)
                self.records["verify"][0][name] = value
                with self.assertRaises(ValueError):
                    self.build(validate_only=True)

    def test_duplicate_ids_are_rejected_in_inputs_teachers_and_manifest(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.build()
        self.rows.pop()
        self.records["direct"].append(copy.deepcopy(self.records["direct"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.build()
        self.records["direct"].pop()
        self.prepare()
        manifest = json.loads(self.manifest.read_text())
        manifest["teachers"].append(manifest["teachers"][0])
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate manifest"):
            builder.build_targets(self.input, self.manifest, self.output)

    def test_invalid_weights_are_rejected(self):
        for weights in ({"direct": -1, "verify": 1}, {"direct": 0, "verify": 0},
                        {"direct": math.inf, "verify": 1}, {"direct": True, "verify": 1}):
            with self.subTest(weights=weights):
                self.weights = weights
                with self.assertRaises(ValueError):
                    self.build(validate_only=True)

    def test_split_selection_never_builds_test_targets(self):
        self.add_row(row("dev-item", split="dev"))
        self.add_row(row("test-item", split="test"))
        self.build()
        self.assertEqual([item["id"] for item in self.output_rows()], ["item-1"])
        summary = self.build(split="dev", overwrite=True)
        self.assertEqual(summary["purpose"], "teacher_quality_audit")
        self.assertEqual([item["id"] for item in self.output_rows()], ["dev-item"])
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "test is forbidden"):
            self.build(split="test", overwrite=True)
        self.assertEqual(before, self.output.read_bytes())

    def test_refuses_overwrite_and_input_collisions(self):
        self.build()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, self.output.read_bytes())
        with self.assertRaisesRegex(ValueError, "distinct"):
            builder.build_targets(self.input, self.manifest, self.input, overwrite=True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            builder.build_targets(self.input, self.manifest, self.output,
                                  report_path=self.output, overwrite=True)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_validate_only_and_isolated_python_cli_need_no_model(self):
        self.prepare()
        process = subprocess.run(
            [sys.executable, "-I", str(ROOT / "harness/build_multiteacher_targets.py"),
             "--input", str(self.input), "--manifest", str(self.manifest), "--validate-only"],
            capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertTrue(report["validate_only"])
        self.assertIsNone(report["output"])
        self.assertEqual(report["retained_rows"], 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())


if __name__ == "__main__":
    unittest.main()
