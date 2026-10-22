"""Dependency-free regression tests for grouped data and frozen-threshold scoring."""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_harness(name):
    path = ROOT / "harness" / (name + ".py")
    if not path.exists():
        raise AssertionError(f"missing implementation: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise AssertionError(f"{name} must import without model/parquet dependencies: {error}")
    return module


def row(ident, source="multirc", passage="passage", question=None,
        target=None, group_id=None, split="train", selection_mode="multi"):
    target = [1, 0] if target is None else target
    result = {
        "id": ident, "source": source, "lang": "en", "split": split,
        "state": passage, "question": question or ident,
        "options": [{"key": chr(65 + i), "desc": f"choice {i}"}
                    for i in range(len(target))],
        "target": target, "n_options": len(target), "gold_letters": None,
        "selection_mode": selection_mode, "allow_empty": False,
    }
    if group_id is not None:
        result["group_id"] = group_id
    return result


class GroupedDataTests(unittest.TestCase):
    def test_builder_import_is_dependency_free(self):
        self.assertTrue(callable(load_harness("multiselect_to_unified").stratified_dev))

    def test_multirc_preserves_group_and_original_question_identity(self):
        builder = load_harness("multiselect_to_unified")
        candidates = [
            {"idx": {"paragraph": 7, "question": q, "answer": a},
             "paragraph": "A long passage", "question": f"question {q}",
             "answer": f"answer {a}", "label": int(a == 0)}
            for q in (3, 9) for a in range(2)
        ]
        converted = builder.convert_multirc_rows(candidates, "train", Counter())
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["group_id"], converted[1]["group_id"])
        self.assertEqual([r["question_id"] for r in converted], [3, 9])
        self.assertEqual(converted[0]["original_idx"], {"paragraph": 7, "question": 3})
        self.assertTrue(all(r["selection_mode"] == "multi" for r in converted))
        self.assertTrue(all(r["allow_empty"] is False for r in converted))

    def test_multirc_supports_more_than_26_candidates(self):
        builder = load_harness("multiselect_to_unified")
        candidates = [{"idx": {"paragraph": 1, "question": 1, "answer": i},
                       "paragraph": "p", "question": "q", "answer": str(i),
                       "label": int(i == 0)} for i in range(27)]
        converted = builder.convert_multirc_rows(candidates, "train", Counter())
        self.assertEqual(converted[0]["options"][-1]["key"], "AA")

    def test_shared_passage_or_group_id_never_crosses_split(self):
        builder = load_harness("multiselect_to_unified")
        rows = [
            row("a", passage="same full passage", group_id="first"),
            row("b", passage=" same   full passage ", group_id="second"),
            row("c", passage="alternate passage spelling", group_id="second"),
            row("d", passage="separate 1", group_id="third"),
            row("e", passage="separate 2", group_id="fourth"),
            row("f", passage="separate 3", group_id="fifth"),
        ]
        train, dev, report = builder.stratified_dev(rows, 3, 42, return_report=True)
        sides = {r["id"]: "train" for r in train}
        sides.update({r["id"]: "dev" for r in dev})
        self.assertEqual(len({sides[k] for k in ("a", "b", "c")}), 1)
        self.assertEqual(report["actual_dev_rows"], len(dev))
        self.assertEqual(report["requested_dev_rows"], 3)
        audit = builder.overlap_audit({"train": train, "dev": dev, "test": []})
        self.assertEqual(audit["group"]["train_dev"], 0)
        self.assertEqual(audit["full_question"]["train_dev"], 0)

    def test_legacy_rows_use_full_passage_and_split_is_order_independent(self):
        builder = load_harness("multiselect_to_unified")
        rows = [row(f"q{i}", passage=f"paragraph {i // 2}") for i in range(12)]
        first = builder.stratified_dev(rows, 5, 31)
        second = builder.stratified_dev(list(reversed(rows)), 5, 31)
        self.assertEqual([{r["id"] for r in split} for split in first],
                         [{r["id"] for r in split} for split in second])
        train_passages = {builder.norm(r["state"]) for r in first[0]}
        dev_passages = {builder.norm(r["state"]) for r in first[1]}
        self.assertFalse(train_passages & dev_passages)

    def test_cmb_metadata_is_not_a_passage_group(self):
        builder = load_harness("multiselect_to_unified")
        rows = [row(f"q{i}", source="cmb_exam_multi", passage="same exam metadata")
                for i in range(10)]
        train, dev = builder.stratified_dev(rows, 3, 7)
        self.assertEqual((len(train), len(dev)), (7, 3))

    def test_full_keys_do_not_truncate_after_200_characters(self):
        builder = load_harness("multiselect_to_unified")
        prefix = "x" * 250
        a = row("a", passage=prefix + " first", question="same question")
        b = row("b", passage=prefix + " second", question="same question")
        self.assertNotEqual(builder.row_key(a), builder.row_key(b))
        audit = builder.overlap_audit({"train": [a], "dev": [b], "test": []})
        self.assertEqual(audit["full_question"]["train_dev"], 0)
        self.assertEqual(audit["group"]["train_dev"], 0)

    def test_group_audit_detects_different_questions_on_shared_passage(self):
        builder = load_harness("multiselect_to_unified")
        audit = builder.overlap_audit({"train": [row("a")], "dev": [row("b")], "test": []})
        self.assertEqual(audit["full_question"]["train_dev"], 0)
        self.assertEqual(audit["group"]["train_dev"], 1)

    def test_groups_are_scoped_by_source(self):
        builder = load_harness("multiselect_to_unified")
        a = row("a", source="source-a", group_id="1")
        b = row("b", source="source-b", group_id="1", question="a")
        audit = builder.overlap_audit({"train": [a], "dev": [b], "test": []})
        self.assertEqual(audit["group"]["train_dev"], 0)
        self.assertEqual(audit["full_question"]["train_dev"], 0)

    def test_holdout_overlap_removes_whole_training_group(self):
        builder = load_harness("multiselect_to_unified")
        pool = [row("leak-a", passage="test passage", group_id="a"),
                row("leak-b", passage="variant", group_id="a")]
        pool += [row(f"keep{i}", passage=f"p{i}") for i in range(8)]
        test = [row("official", passage="test passage", split="test")]
        train, dev, test_out, report = builder.build_splits(pool, [], test, 2, 42)
        self.assertFalse({"leak-a", "leak-b"} & {r["id"] for r in train + dev})
        self.assertEqual(test, test_out)
        self.assertEqual(report["excluded_train_holdout_rows"], 2)
        self.assertTrue(all(value == 0 for value in report["overlap"]["group"].values()))

    def test_deduplication_preserves_alias_bridges_to_heldout_groups(self):
        builder = load_harness("multiselect_to_unified")
        pool = [row("a", passage="heldout passage", question="duplicate", group_id="first"),
                row("b", passage="heldout passage", question="duplicate", group_id="second"),
                row("c", passage="variant passage", group_id="second")]
        pool += [row(f"keep{i}", passage=f"p{i}") for i in range(8)]
        heldout = [row("official", passage="heldout passage", split="test")]
        train, dev, _, report = builder.build_splits(pool, [], heldout, 2, 42)
        self.assertFalse({"a", "b", "c"} & {r["id"] for r in train + dev})
        self.assertEqual(report["deduplicated_train_rows"], 1)
        self.assertEqual(report["excluded_train_holdout_rows"], 2)

    def test_existing_output_is_never_overwritten(self):
        builder = load_harness("multiselect_to_unified")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "published"
            output.mkdir()
            saved = output / "train.jsonl"
            saved.write_text("published\n")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                builder.ensure_new_output(output)
            self.assertEqual(saved.read_text(), "published\n")

    def test_cmb_mixed_protocol_is_from_native_type(self):
        builder = load_harness("multiselect_to_unified")
        raw = [{"id": kind, "question_type": kind, "question": "q " + kind,
                "option": {"A": "a", "B": "b"}, "answer": "A"}
               for kind in ("单项选择题", "多项选择题")]
        multi = builder.convert_cmb(raw, "cmb_exam_multi", "train", "cmb", Counter())
        mixed = builder.convert_cmb(raw, "cmb_exam_multi", "train", "cmb", Counter(),
                                    question_mode="mixed")
        self.assertEqual([r["selection_mode"] for r in multi], ["multi"])
        self.assertEqual([r["selection_mode"] for r in mixed], ["single", "multi"])


class MetricTests(unittest.TestCase):
    def test_count_metrics_and_oracle_ranking_are_distinct(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 0, 1], source="s1"),
                row("b", target=[0, 1], source="s2")]
        predictions = [{"id": "a", "p": [.9, .7, .8]},
                       {"id": "b", "p": [.4, .3]}]
        result = metrics.compute_metrics(rows, predictions)
        self.assertEqual(result["question_em"], 0.0)
        self.assertEqual(result["count_accuracy"], 0.0)
        self.assertEqual(result["count_mae"], 1.0)
        self.assertEqual(result["oracle_gold_count"]["question_em"], .5)
        self.assertTrue(result["oracle_gold_count"]["uses_gold_count"])
        self.assertEqual(result["groups"]["source"]["s1"]["count_mae"], 1.0)
        self.assertEqual(result["groups"]["source"]["s2"]["oracle_gold_count"]["question_em"], 0.0)

    def test_oracle_ties_never_use_gold_to_break_the_tie(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[0, 1])]
        result = metrics.compute_metrics(rows, [{"id": "a", "p": [.5, .5]}])
        oracle = result["oracle_gold_count"]
        self.assertEqual(oracle["question_em"], 0.0)
        self.assertEqual(oracle["cutoff_tie_questions"], 1)
        self.assertEqual(oracle["question_em_expected_random_ties"], .5)
        self.assertEqual(oracle["strict_separation_question_em"], 0.0)
        self.assertEqual(result["tp"], 1)  # >= includes the threshold boundary.
        self.assertEqual(result["fp"], 1)

    def test_single_protocol_is_explicit_and_never_inferred_from_gold_count(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("single", target=[1, 0], selection_mode="single"),
                row("multi", target=[1, 0], selection_mode="multi")]
        predictions = [{"id": ident, "p": [.4, .3]} for ident in ("single", "multi")]
        exported = metrics.prediction_records(rows, predictions)
        self.assertEqual(exported[0]["predicted_indices"], [0])
        self.assertEqual(exported[1]["predicted_indices"], [])
        self.assertEqual(metrics.compute_metrics(rows, predictions)["count_accuracy"], .5)

    def test_explicit_single_argmax_preserves_ranking_when_sigmoid_saturates(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[0, 1], selection_mode="single")]
        exported = metrics.prediction_records(rows, [{"id": "a", "p": [1.0, 1.0],
                                                       "logits": [30.0, 100.0]}])
        self.assertEqual(exported[0]["predicted_indices"], [1])

    def test_duplicate_missing_extra_and_invalid_predictions_fail(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a")]
        good = {"id": "a", "p": [.8, .1]}
        for predictions in ([good, good], [], [good, {"id": "extra", "p": [.2, .3]}],
                            [{"id": "a", "p": [float("nan"), .2]}],
                            [{"id": "a", "p": [1.1, .2]}],
                            [{"id": "a", "p": [.2]}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                metrics.compute_metrics(rows, predictions)

    def test_logits_preserve_bce_when_sigmoid_saturates(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 0])]
        prediction = {"id": "a", "p": [1.0, 1.0], "logits": [100.0, 100.0]}
        result = metrics.compute_metrics(rows, [prediction])
        self.assertAlmostEqual(result["bce"], 50.0)

    def test_provided_sets_support_external_decoders_without_gold_count(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 0, 1])]
        prediction = {"id": "a", "p": [.8, .7, .9], "predicted_indices": [0, 2]}
        result = metrics.compute_metrics(rows, [prediction], decoder="provided")
        self.assertEqual(result["question_em"], 1.0)
        self.assertEqual(result["count_accuracy"], 1.0)

    def test_count_distribution_scoring_uses_native_probability_vector(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 0, 1])]
        rows[0]["allow_empty"] = True
        prediction = {"id": "a", "p": [.6, .5, .5], "count_probs": [.1, .2, .7, 0.0],
                      "predicted_indices": [0, 2], "decoder": "set_map"}
        result = metrics.compute_metrics(rows, [prediction], decoder="auto")
        self.assertEqual(result["question_em"], 1.0)
        self.assertAlmostEqual(result["count_nll"], -math.log(.7))
        self.assertAlmostEqual(result["count_brier"], .14)
        self.assertEqual(result["count_mode_accuracy"], 1.0)
        self.assertAlmostEqual(result["count_distribution_mean_mae"], .4)
        self.assertEqual(result["count_distribution_n_questions"], 1)

    def test_structured_predictions_reject_incoherent_membership_and_count_vectors(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 0])]
        base = {"id": "a", "decoder": "set_map", "predicted_indices": [0]}
        for p, counts, message in (([.8, .8], [0, 1, 0], "expectations"),
                                    ([.5, .5], [.5, 0, .5], "protocol")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                metrics.compute_metrics(rows, [{**base, "p": p, "count_probs": counts}], decoder="auto")

    def test_count_distribution_rejects_invalid_vectors_and_ties_do_not_use_gold(self):
        metrics = load_harness("multiselect_metrics")
        rows = [row("a", target=[1, 1])]
        item = {"id": "a", "p": [.75, .75], "count_probs": [0, .5, .5]}
        self.assertEqual(metrics.compute_metrics(rows, [item])["count_mode_accuracy"], 0.0)
        for bad in ([0, .4, .4], [0, float("nan"), 1], [.2, .8], [0, -1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                metrics.compute_metrics(rows, [{**item, "count_probs": bad}])


class EvaluationCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", str(ROOT / "harness" / "eval_multiselect.py"),
                               *map(str, args)], text=True, capture_output=True)

    def test_validate_only_accepts_test_rows_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "test.jsonl"
            data.write_text(json.dumps(row("a", split="test")) + "\n")
            completed = self.run_cli("--input", data, "--validate-only")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["expected_split"], "test")
            self.assertFalse(report["threshold_fitted_on_input"])
            self.assertFalse(report["prompt_lengths_checked"])

    def test_score_cli_freezes_threshold_and_exports_predicted_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions, output = root / "test.jsonl", root / "p.jsonl", root / "score"
            data.write_text(json.dumps(row("a", split="test")) + "\n")
            predictions.write_text(json.dumps({"id": "a", "p": [.6, .4]}) + "\n")
            completed = self.run_cli("--input", data, "--score-predictions", predictions,
                                     "--threshold", ".7", "--output", output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "metrics.json").read_text())
            exported = json.loads((output / "predictions.jsonl").read_text())
            self.assertEqual(report["threshold"], .7)
            self.assertEqual(report["question_em"], 0.0)
            self.assertFalse(report["threshold_fitted_on_input"])
            self.assertEqual(exported["predicted_indices"], [])
            self.assertEqual(exported["p"], [.6, .4])

    def test_token_budget_rejects_overlong_inputs_instead_of_truncating(self):
        evaluator = load_harness("eval_multiselect")
        with self.assertRaisesRegex(ValueError, "never truncated"):
            evaluator.make_batches(["a"], [9], batch_size=2,
                                   max_input_tokens=8, max_batch_tokens=16)
        with self.assertRaisesRegex(ValueError, "budget"):
            evaluator.make_batches(["a"], [9], batch_size=2,
                                   max_input_tokens=16, max_batch_tokens=8)

    def test_right_padding_is_required_and_batch_plan_covers_every_row(self):
        evaluator = load_harness("eval_multiselect")
        evaluator.validate_right_padding([[1, 1, 0], [1, 1, 1]], [2, 3])
        with self.assertRaisesRegex(ValueError, "right padding"):
            evaluator.validate_right_padding([[0, 1, 1]], [2])
        with self.assertRaisesRegex(ValueError, "untruncated plan"):
            evaluator.validate_right_padding([[1, 1, 0]], [3])
        lengths = [9, 3, 4, 10]
        batches = evaluator.make_batches(["a", "b", "c", "d"], lengths,
                                         batch_size=3, max_input_tokens=12, max_batch_tokens=20)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(4)))
        self.assertTrue(all(len(batch) <= 3 and len(batch) * max(lengths[i] for i in batch) <= 20
                            for batch in batches))

    def test_set_sidecar_cannot_silently_fall_back_to_bce(self):
        evaluator = load_harness("eval_multiselect")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "decision_readout.json").write_text(json.dumps({
                "schema": "decision_set_readout_v1", "objective": "set",
                "max_candidates": 255, "hidden_size": 4}))
            with self.assertRaisesRegex(ValueError, "decision_count_head"):
                evaluator.checkpoint_objective(model)
            (model / "decision_count_head.pt").write_bytes(b"test fixture; never loaded")
            self.assertEqual(evaluator.checkpoint_objective(model), "set")


if __name__ == "__main__":
    unittest.main()
