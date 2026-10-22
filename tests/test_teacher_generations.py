"""CPU tests of matched teacher-quality audits; no neural-model claims."""
from __future__ import annotations

import copy
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
    "audit_teacher_generations", ROOT / "harness/audit_teacher_generations.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(ident="q1", target=(1, 0, 0)):
    return {"id": ident, "source": "toy", "lang": "en", "split": "dev",
            "state": "Evidence", "question": "Which statements are true?",
            "options": [{"key": chr(65 + i), "desc": f"Statement {i}"} for i in range(len(target))],
            "n_options": len(target), "target": list(target),
            "selection_mode": "multi", "allow_empty": False}


def teacher(item, checkpoint, chains, teacher_id="direct", requested=None):
    requested = len(chains) if requested is None else requested
    return {
        "id": item["id"], "teacher_id": teacher_id, "teacher_view": teacher_id,
        "checkpoint": checkpoint, "n_options": len(item["target"]),
        "input_digest": audit.input_digest(item),
        "option_keys": [o["key"] for o in item["options"]],
        "n_chains": requested, "valid_chains": len(chains), "missing_chains": requested - len(chains),
        "q_semantics": "candidate_membership", "q_transform": "membership", "readout": "candidate_yes_no_v1",
        "q": [math.fsum(column) / len(chains) for column in zip(*chains)] if chains else None,
        "per_chain_membership": chains, "per_chain_ids": list(range(len(chains))),
        "per_chain_metadata": [{"chain": i, "included": True, "truncated": False,
                                "finish_reason": "stop", "reasoning_sha256": f"reason-{i}",
                                "generation": {"seed": 42, "temperature": 0.6, "top_p": 0.95,
                                               "top_k": 20, "max_tokens": 2048, "max_model_len": 8192}}
                               for i in range(len(chains))],
        "excluded_chain_metadata": [],
        "provenance": {"checkpoint_identity_kind": "explicit", "input_digest": audit.input_digest(item),
                       "truncated_policy": "exclude", "reasoning_prompt_version": "test-v1",
                       "readout_prompt_version": "candidate_yes_no_v1", "readout_max_input_tokens": 8192,
                       "binary_tokens": {"yes_token": "Yes", "no_token": "No", "yes_token_id": 9175,
                                         "no_token_id": 2665},
                       "option_alignment": "input_option_order"},
    }


def direct(item, checkpoint, correct=True):
    target = item["target"] if correct else [1 - y for y in item["target"]]
    return {"id": item["id"], "p": [0.9 if y else 0.1 for y in target],
            "input_digest": audit.input_digest(item),
            "option_keys": [o["key"] for o in item["options"]],
            "checkpoint_id": checkpoint, "provenance": {"checkpoint_identity_kind": "explicit"}}


class TeacherGenerationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.rows = [row()]
        self.input = self.directory / "dev.jsonl"
        self.manifest = self.directory / "generations.json"
        self.output = self.directory / "audit.json"
        self.max_map = 12
        self.teacher_records = {
            "g0": {"direct": [teacher(self.rows[0], "checkpoint0", [[1, 0, 0], [1, 1, 1]])]},
            "g1": {"direct": [teacher(self.rows[0], "checkpoint1", [[1, 1, 0], [1, 0, 1]])]},
        }
        self.direct_records = {
            "g0": [direct(self.rows[0], "checkpoint0", False)],
            "g1": [direct(self.rows[0], "checkpoint1", True)],
        }
        self.weights = {"direct": 1.0, "verify": 1.0}

    def write_lines(self, path, records):
        path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")

    def prepare(self):
        self.write_lines(self.input, self.rows)
        generations = []
        for index, ident in enumerate(self.teacher_records):
            self.write_lines(self.directory / f"{ident}_direct.jsonl", self.direct_records[ident])
            teachers = []
            for teacher_id, records in self.teacher_records[ident].items():
                filename = f"{ident}_{teacher_id}_teacher.jsonl"
                self.write_lines(self.directory / filename, records)
                teachers.append({"id": teacher_id, "path": filename, "weight": self.weights[teacher_id]})
            teacher_manifest = f"{ident}_teachers.json"
            (self.directory / teacher_manifest).write_text(json.dumps({"teachers": teachers}), encoding="utf-8")
            generations.append({"id": ident, "checkpoint_id": f"checkpoint{index}",
                                "direct_predictions": f"{ident}_direct.jsonl", "teachers": teacher_manifest})
        self.manifest.write_text(json.dumps({"generations": generations, "max_map_candidates": self.max_map}), encoding="utf-8")

    def run_audit(self, **kwargs):
        self.prepare()
        return audit.audit_generations(self.input, self.manifest,
                                       None if kwargs.get("validate_only") else self.output, **kwargs)

    def test_same_memberships_can_hide_a_worse_count_law_and_teacher(self):
        result = self.run_audit()
        old, new = [g["per_question_teacher"]["q1"] for g in result["generations"]]
        self.assertEqual(old["q"], new["q"])
        self.assertEqual(old["q"], [1, 0.5, 0.5])
        self.assertEqual(old["count_probs"], [0, 0.5, 0, 0.5])
        self.assertEqual(new["count_probs"], [0, 0, 1, 0])
        self.assertEqual(old["gold_set_probability"], 0.5)
        self.assertEqual(new["gold_set_probability"], 0)
        self.assertEqual(old["map_predicted_indices"], [0])
        self.assertEqual(new["map_predicted_indices"], [0, 1])
        paired = result["paired"][0]
        self.assertEqual(paired["direct_em_delta"], 1)
        self.assertEqual(paired["teacher_map_em_delta"], -1)
        self.assertTrue(paired["matched_proxy_pattern"])
        self.assertFalse(paired["teaching_utility_measured"])
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_mixture_gold_probability_is_not_product_of_mean_memberships(self):
        result = self.run_audit(validate_only=True)
        record = result["generations"][0]["per_question_teacher"]["q1"]
        product_of_mean = record["q"][0] * (1 - record["q"][1]) * (1 - record["q"][2])
        self.assertEqual(product_of_mean, 0.25)
        self.assertEqual(record["gold_set_probability"], 0.5)
        self.assertFalse(self.output.exists())

    def test_teacher_view_weight_not_number_of_chains(self):
        for i, ident in enumerate(self.teacher_records):
            self.teacher_records[ident] = {
                "direct": [teacher(self.rows[0], f"checkpoint{i}", [[1, 0, 0]], "direct")],
                "verify": [teacher(self.rows[0], f"checkpoint{i}", [[1, 1, 1]] * 3, "verify")],
            }
        result = self.run_audit(validate_only=True)
        self.assertEqual(result["generations"][0]["per_question_teacher"]["q1"]["gold_set_probability"], 0.5)
        self.assertTrue(result["teacher_settings_matched"])

    def test_single_protocol_conditions_each_chain_before_mixing(self):
        self.rows = [row(target=(1, 0))]
        self.rows[0]["selection_mode"] = "single"
        for i, ident in enumerate(self.teacher_records):
            self.teacher_records[ident]["direct"] = [teacher(self.rows[0], f"checkpoint{i}", [[0.8, 0.8], [0.1, 0.2]])]
            self.direct_records[ident] = [direct(self.rows[0], f"checkpoint{i}")]
        result = self.run_audit(validate_only=True)
        record = result["generations"][0]["per_question_teacher"]["q1"]
        self.assertAlmostEqual(record["gold_set_probability"], (0.5 + 0.08 / 0.26) / 2)
        self.assertAlmostEqual(record["gold_set_probability"], record["q"][0])
        self.assertNotAlmostEqual(record["gold_set_probability"], (0.16 + 0.08) / (0.32 + 0.26))

    def test_common_cohort_prevents_coverage_induced_improvement(self):
        second = row("q2")
        self.rows.append(second)
        self.direct_records["g0"] = [direct(item, "checkpoint0", False) for item in self.rows]
        self.direct_records["g1"] = [direct(self.rows[0], "checkpoint1", False), direct(second, "checkpoint1", True)]
        self.teacher_records["g0"]["direct"].append(teacher(second, "checkpoint0", [[1, 0, 0]] * 2))
        # g1 has no q2 teacher: its direct improvement on q2 cannot enter a paired claim.
        result = self.run_audit(validate_only=True)
        self.assertEqual(result["common_cohort"]["ids"], ["q1"])
        self.assertEqual(result["common_cohort"]["coverage"], 0.5)
        self.assertEqual(result["generations"][1]["direct_on_available"]["question_em"], 0.5)
        self.assertEqual(result["generations"][1]["direct_on_common"]["question_em"], 0)
        self.assertEqual(result["paired"][0]["direct_em_delta"], 0)
        self.assertFalse(result["paired"][0]["matched_proxy_pattern"])
        self.assertEqual(result["generations"][1]["coverage"]["teacher_rows"], 1)

    def test_no_valid_chain_is_unavailable_not_negative_target(self):
        self.teacher_records["g1"]["direct"] = [teacher(self.rows[0], "checkpoint1", [], requested=2)]
        result = self.run_audit(validate_only=True)
        self.assertEqual(result["common_cohort"]["n_questions"], 0)
        self.assertEqual(result["generations"][1]["teacher_on_available"]["n_questions"], 0)
        self.assertIsNone(result["paired"][0]["direct_em_delta"])
        self.assertIsNone(result["paired"][0]["matched_proxy_pattern"])
        self.assertFalse(result["teacher_settings_matched"])

    def test_k_bound_skips_only_exact_map(self):
        self.max_map = 2
        result = self.run_audit(validate_only=True)
        record = result["generations"][0]["per_question_teacher"]["q1"]
        self.assertFalse(record["map_computed"])
        self.assertIsNone(record["map_question_em"])
        self.assertEqual(record["gold_set_probability"], 0.5)
        self.assertIsNone(result["paired"][0]["teacher_map_em_delta"])
        self.assertTrue(result["paired"][0]["matched_proxy_pattern"])

    def test_tiny_nonzero_set_probability_keeps_exact_log_nll(self):
        self.rows = [row(target=(1, 1, 1))]
        for i, ident in enumerate(self.teacher_records):
            self.teacher_records[ident]["direct"] = [teacher(self.rows[0], f"checkpoint{i}", [[1e-250] * 3])]
            self.direct_records[ident] = [direct(self.rows[0], f"checkpoint{i}")]
        result = self.run_audit(validate_only=True)
        record = result["generations"][0]["per_question_teacher"]["q1"]
        # exp(log Q) underflows, but an impossible event must remain distinct.
        self.assertEqual(record["gold_set_probability"], 0)
        self.assertFalse(record["gold_set_zero_probability"])
        self.assertTrue(math.isfinite(record["gold_set_nll"]))
        self.assertGreater(record["gold_set_nll"], 1000)
        self.assertAlmostEqual(record["gold_set_nll_clipped"], -math.log(audit.EPSILON))
        self.assertEqual(record["count_probs"][3], 0)
        self.assertFalse(record["count_zero_gold_probability"])
        self.assertAlmostEqual(record["count_nll"], record["gold_set_nll"])
        self.assertLess(record["count_gold_log_probability"], -1000)

    def test_exact_paired_nll_detects_worsening_beyond_clipping_cap(self):
        self.rows = [row(target=(1, 1, 1))]
        for i, ident in enumerate(self.teacher_records):
            self.teacher_records[ident]["direct"] = [teacher(
                self.rows[0], f"checkpoint{i}", [[1e-50 if i == 0 else 1e-100] * 3])]
            self.direct_records[ident] = [direct(self.rows[0], f"checkpoint{i}", bool(i))]
        result = self.run_audit(validate_only=True)
        paired = result["paired"][0]
        self.assertEqual(paired["teacher_gold_set_nll_clipped_delta"], 0)
        self.assertEqual(paired["teacher_count_nll_clipped_delta"], 0)
        self.assertGreater(paired["teacher_gold_set_nll_delta"], 200)
        self.assertGreater(paired["teacher_count_nll_delta"], 200)
        self.assertEqual(paired["teacher_count_nll_paired_finite_questions"], 1)
        self.assertEqual(paired["proxy_nll_comparison_basis"], "exact_full_cohort")
        self.assertTrue(paired["matched_proxy_pattern"])

    def test_count_nll_true_zero_is_distinct_from_underflow(self):
        result = self.run_audit(validate_only=True)
        record = result["generations"][1]["per_question_teacher"]["q1"]
        self.assertTrue(record["count_zero_gold_probability"])
        self.assertIsNone(record["count_nll"])
        self.assertIsNone(record["count_gold_log_probability"])
        self.assertEqual(result["paired"][0]["teacher_count_nll_paired_finite_questions"], 0)

    def test_stale_checkpoint_identity_is_rejected(self):
        self.teacher_records["g1"]["direct"][0]["checkpoint"] = "checkpoint0"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.run_audit(validate_only=True)
        self.teacher_records["g1"]["direct"][0]["checkpoint"] = "checkpoint1"
        self.direct_records["g1"][0]["checkpoint_id"] = "checkpoint0"
        with self.assertRaisesRegex(ValueError, "checkpoint_id"):
            self.run_audit(validate_only=True)

    def test_path_only_checkpoint_and_stale_digest_are_rejected(self):
        self.direct_records["g1"][0]["provenance"]["checkpoint_identity_kind"] = "local_path"
        with self.assertRaisesRegex(ValueError, "explicit checkpoint"):
            self.run_audit(validate_only=True)
        self.direct_records["g1"][0]["provenance"]["checkpoint_identity_kind"] = "explicit"
        self.direct_records["g1"][0]["input_digest"] = "stale"
        with self.assertRaisesRegex(ValueError, "input digest"):
            self.run_audit(validate_only=True)

    def test_option_order_mismatch_is_rejected(self):
        self.direct_records["g1"][0]["option_keys"].reverse()
        with self.assertRaisesRegex(ValueError, "option order"):
            self.run_audit(validate_only=True)

    def test_changed_sampling_budget_is_descriptive_not_matched(self):
        for metadata in self.teacher_records["g1"]["direct"][0]["per_chain_metadata"]:
            metadata["generation"]["max_tokens"] = 4096
        result = self.run_audit(validate_only=True)
        self.assertFalse(result["teacher_settings_matched"])
        self.assertTrue(result["paired"][0]["descriptive_direct_up_teacher_down_pattern"])
        self.assertIsNone(result["paired"][0]["matched_proxy_pattern"])

    def test_missing_sampling_budget_disables_matched_claim(self):
        for metadata in self.teacher_records["g1"]["direct"][0]["per_chain_metadata"]:
            metadata.pop("generation")
        result = self.run_audit(validate_only=True)
        self.assertFalse(result["teacher_settings_matched"])
        self.assertIsNone(result["paired"][0]["matched_proxy_pattern"])

    def test_missing_context_budget_or_binary_tokens_disables_matching(self):
        original = copy.deepcopy(self.teacher_records["g1"]["direct"][0])
        for field in ("binary_tokens", "readout_max_input_tokens", "max_model_len"):
            with self.subTest(field=field):
                record = copy.deepcopy(original)
                if field == "max_model_len":
                    for chain in record["per_chain_metadata"]:
                        chain["generation"].pop(field)
                else:
                    record["provenance"].pop(field)
                self.teacher_records["g1"]["direct"][0] = record
                result = self.run_audit(validate_only=True)
                self.assertFalse(result["teacher_settings_matched"])
                self.assertIsNone(result["paired"][0]["matched_proxy_pattern"])

    def test_changed_binary_token_ids_or_readout_context_breaks_match(self):
        original = copy.deepcopy(self.teacher_records["g1"]["direct"][0])
        for field in ("binary_tokens", "readout_max_input_tokens", "readout_prompt_version"):
            with self.subTest(field=field):
                record = copy.deepcopy(original)
                if field == "binary_tokens":
                    record["provenance"][field]["yes_token_id"] = 42
                elif field == "readout_max_input_tokens":
                    record["provenance"][field] = 4096
                else:
                    record["provenance"][field] = "changed-v2"
                self.teacher_records["g1"]["direct"][0] = record
                result = self.run_audit(validate_only=True)
                self.assertFalse(result["teacher_settings_matched"])

    def test_completion_rates_keep_failed_and_truncated_chains(self):
        for generation in self.teacher_records.values():
            record = generation["direct"][0]
            record["n_chains"] = 4
            record["missing_chains"] = 2
            record["excluded_chain_metadata"] = [
                {"chain": 2, "included": False, "truncated": True, "finish_reason": "length"},
                {"chain": 3, "included": False, "truncated": False, "finish_reason": "error"},
            ]
        result = self.run_audit(validate_only=True)
        completion = result["generations"][0]["completion"]["direct"]
        self.assertEqual(completion["completion_rate_in_present_rows"], 0.5)
        self.assertEqual(completion["usable_chain_rate_in_present_rows"], 0.5)
        self.assertEqual(completion["excluded_truncated_chains"], 1)

    def test_no_overwrite_and_no_input_clobber(self):
        self.run_audit()
        old = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            audit.audit_generations(self.input, self.manifest, self.output)
        self.assertEqual(old, self.output.read_bytes())
        with self.assertRaisesRegex(ValueError, "distinct"):
            audit.audit_generations(self.input, self.manifest, self.input, overwrite=True)

    def test_duplicate_and_extra_prediction_ids_fail(self):
        self.direct_records["g1"].append(copy.deepcopy(self.direct_records["g1"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.run_audit(validate_only=True)
        self.direct_records["g1"][-1]["id"] = "unseen"
        with self.assertRaisesRegex(ValueError, "outside the probe"):
            self.run_audit(validate_only=True)

    def test_training_or_test_rows_cannot_enter_exploratory_probe(self):
        for split in ("train", "test", None):
            with self.subTest(split=split):
                self.rows[0]["split"] = split
                with self.assertRaisesRegex(ValueError, "split"):
                    self.run_audit(validate_only=True)

    def test_cli_isolation(self):
        self.prepare()
        result = subprocess.run([sys.executable, "-I", str(ROOT / "harness/audit_teacher_generations.py"),
                                 "--input", str(self.input), "--manifest", str(self.manifest),
                                 "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema"], audit.SCHEMA)


if __name__ == "__main__":
    unittest.main()
