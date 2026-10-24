"""Dependency-free tests for the Gate 1 readout-audit comparison."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from readout_audit import _subset, audit_report


def question(ident, target):
    return {
        "id": ident,
        "source": "s",
        "lang": "zh",
        "split": "dev",
        "options": [
            {"key": chr(65 + i), "desc": f"opt{i}"} for i in range(len(target))
        ],
        "target": target,
        "selection_mode": "multi",
        "allow_empty": False,
    }


def prediction(ident, probs):
    return {
        "id": ident,
        "p": probs,
        "logits": [
            math.log(p / (1 - p)) if 0 < p < 1 else (10.0 if p == 1 else -10.0)
            for p in probs
        ],
    }


ROWS = [
    question("q1", [1, 0, 1, 0, 0]),
    question("q2", [1, 0, 0, 1, 0]),
    question("q3", [1, 1, 0, 0, 0]),
]


class ReadoutAuditTests(unittest.TestCase):
    def test_identical_readouts_have_zero_delta(self):
        probs = [
            [0.9, 0.1, 0.8, 0.1, 0.1],
            [0.8, 0.1, 0.2, 0.9, 0.1],
            [0.9, 0.8, 0.1, 0.2, 0.1],
        ]
        predictions = [prediction(row["id"], p) for row, p in zip(ROWS, probs)]
        report = audit_report(ROWS, predictions, predictions)
        self.assertEqual(report["schema"], "readout_audit_v1")
        self.assertEqual(report["n_questions"], 3)
        for name, value in report["comparison"].items():
            delta = value["delta_yesno_minus_native"]
            if isinstance(delta, float):
                self.assertAlmostEqual(delta, 0.0, places=12, msg=name)
            self.assertEqual(value["native"], value["yesno"], msg=name)

    def test_overpredicting_yesno_readout_degrades_em(self):
        native = [
            prediction("q1", [0.9, 0.1, 0.8, 0.1, 0.1]),
            prediction("q2", [0.8, 0.1, 0.2, 0.9, 0.1]),
            prediction("q3", [0.9, 0.8, 0.1, 0.2, 0.1]),
        ]
        yesno = [
            prediction("q1", [0.6, 0.6, 0.6, 0.6, 0.6]),
            prediction("q2", [0.6, 0.6, 0.6, 0.6, 0.6]),
            prediction("q3", [0.6, 0.6, 0.6, 0.6, 0.6]),
        ]
        report = audit_report(ROWS, native, yesno)
        comparison = report["comparison"]
        self.assertGreater(
            comparison["question_em"]["native"], comparison["question_em"]["yesno"]
        )
        self.assertLess(comparison["question_em"]["delta_yesno_minus_native"], 0.0)
        self.assertLess(comparison["precision"]["delta_yesno_minus_native"], 0.0)
        self.assertGreaterEqual(comparison["recall"]["delta_yesno_minus_native"], 0.0)


    def test_stride_means_n_evenly_spaced_rows_not_every_nth_row(self):
        rows = [question(f"q{i}", [1, 0, 1, 0, 0]) for i in range(100)]
        sample = _subset(rows, 13, "stride")
        self.assertEqual(len(sample), 13)
        self.assertEqual(len({row["id"] for row in sample}), 13)
        self.assertEqual(sample[0]["id"], "q0")
        self.assertEqual(sample[-1]["id"], "q92")

    def test_stratified_sampler_covers_sources_and_gold_counts(self):
        rows = []
        for source in ("cmb_exam_multi", "multirc"):
            for gold_count in (1, 2, 3, 4):
                for i in range(18):
                    row = question(f"{source}-{gold_count}-{i}",
                                   [1] * gold_count + [0] * (5 - gold_count))
                    row["source"] = source
                    rows.append(row)
        sample = _subset(rows, 30, "stratified", seed=43)
        same = _subset(rows, 30, "stratified", seed=43)
        self.assertEqual([row["id"] for row in sample],
                         [row["id"] for row in same])
        self.assertEqual(len(sample), 30)
        self.assertEqual(len({row["id"] for row in sample}), 30)
        seen = {(r["source"], sum(r["target"])) for r in sample}
        self.assertEqual(len(seen), 8)
        self.assertTrue(all((source, c) in seen
                            for source in ("cmb_exam_multi", "multirc")
                            for c in (1, 2, 3, 4)))

    def test_stratified_sample_requires_one_row_per_nonempty_stratum(self):
        rows = [question("a", [1, 0, 0]),
                question("b", [1, 1, 0]),
                question("c", [1, 1, 1])]
        with self.assertRaisesRegex(ValueError, "smaller than"):
            _subset(rows, 2, "stratified")

    def test_key_metrics_present_in_comparison(self):
        report = audit_report(
            ROWS,
            [prediction(r["id"], [0.9] * len(r["options"])) for r in ROWS],
            [prediction(r["id"], [0.5] * len(r["options"])) for r in ROWS],
        )
        for name in (
            "question_em",
            "f1",
            "precision",
            "recall",
            "bce",
            "membership_brier",
            "count_mae",
            "count_accuracy",
            "oracle_gold_count_em",
        ):
            self.assertIn(name, report["comparison"])
        self.assertIn("native", report)
        self.assertIn("yesno", report)


if __name__ == "__main__":
    unittest.main()
