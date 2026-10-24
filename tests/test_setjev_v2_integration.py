"""Regression guards for the SetJev v2 checkpoint and stage-gating fixes."""

import ast
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SetJevV2IntegrationTests(unittest.TestCase):
    def test_in_training_dev_uses_same_count_anchor_as_loss(self):
        tree = ast.parse((ROOT / "harness/train_multiselect.py").read_text())
        evaluate = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
        ]
        self.assertEqual(len(evaluate), 1)
        calls = [
            node for node in ast.walk(evaluate[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "structured_prediction"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertIn("count_anchor", keywords)
        value = keywords["count_anchor"]
        self.assertIsInstance(value, ast.Attribute)
        self.assertIsInstance(value.value, ast.Name)
        self.assertEqual(value.value.id, "args")
        self.assertEqual(value.attr, "count_anchor")

    def test_anchor_is_recorded_and_resume_guard_exists(self):
        source = (ROOT / "harness/train_multiselect.py").read_text()
        self.assertGreaterEqual(source.count('"count_anchor": args.count_anchor'), 4)
        self.assertIn("resume checkpoint count-anchor mode disagrees", source)
        self.assertIn("--count-anchor requires --objective set", source)

    def test_gate1_payload_uses_stratified_sample_not_head(self):
        payload = json.loads(
            (ROOT / "qz_setjev_v2/readout-audit-gate1-bce-s44.json").read_text()
        )
        self.assertIn("--limit-mode stratified", payload["command"])
        self.assertIn("--sample-seed 43", payload["command"])
        self.assertNotIn("--limit-mode head", payload["command"])

    @unittest.skipUnless(shutil.which("bash"), "bash is not installed")
    def test_submit_script_validates_all_without_job_submission(self):
        script = str(ROOT / "qz/submit_setjev_v2.sh")
        result = subprocess.run(
            ["bash", script, "validate", "all"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing submitted", result.stdout.replace(
            "no GPU jobs submitted", "nothing submitted"
        ))

    @unittest.skipUnless(shutil.which("bash"), "bash is not installed")
    def test_submit_all_is_refused_before_any_qz_call(self):
        script = str(ROOT / "qz/submit_setjev_v2.sh")
        result = subprocess.run(
            ["bash", script, "--submit", "all"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing combined GPU submission", result.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is not installed")
    def test_gate3_requires_real_report_and_explicit_review(self):
        script = str(ROOT / "qz/submit_setjev_v2.sh")
        env = dict(os.environ)
        env["CONFIRM_SUBMIT"] = "1"
        env.pop("GATE1_REVIEWED", None)
        env.pop("GATE1_AUDIT_JSON", None)
        result = subprocess.run(
            ["bash", script, "--submit", "gate3"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Gate 3 requires", result.stderr)


if __name__ == "__main__":
    unittest.main()
