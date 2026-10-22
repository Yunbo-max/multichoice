"""CPU contract tests; real model/tokenizer execution is a separate GPU check."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sii_style_teacher", ROOT / "harness/sii_style_teacher.py")
teacher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher)


class FakeTokenizer:
    """No model dependency; IDs deliberately differ from common real token IDs."""

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith("Yes"):
                result.append(173)
                text = text[3:]
            elif text.startswith("No"):
                result.append(821)
                text = text[2:]
            else:
                result.append(1000 + ord(text[0]))
                text = text[1:]
        return result

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True):
        content = "\n".join(message["content"] for message in messages)
        suffix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return "<user>\n" + content + "\n</user>\n<assistant>\n" + suffix


class FakeEngine:
    """Emit known yes/no evidence for each named candidate, or fail one prompt."""

    def __init__(self, fail_candidate=None, invalid_candidate=None):
        self.prompts = []
        self.fail_candidate = fail_candidate
        self.invalid_candidate = invalid_candidate

    def logits(self, prompts):
        self.prompts.extend(prompts)
        values = []
        for prompt in prompts:
            candidate = "b" if "Candidate under evaluation:\n(B) b:" in prompt else "a"
            if candidate == self.fail_candidate:
                raise RuntimeError("synthetic candidate failure")
            if candidate == self.invalid_candidate:
                values.append([float("nan"), 0.0])
            elif "chain two" in prompt:
                values.append([math.log(3), 0.0] if candidate == "a" else [0.0, math.log(3)])
            else:
                values.append([math.log(9), 0.0] if candidate == "a" else [math.log(4), 0.0])
        return values


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.labels = ["A", "B"]
        self.row = {
            "id": "toy-1", "source": "toy", "lang": "en",
            "state": "state text", "question": "Which statements are supported?",
            "options": [{"key": "a", "desc": "alpha description"},
                        {"key": "b", "desc": "beta description"}],
            "selection_mode": "multi", "allow_empty": False,
            "target": [1, 1], "gold_count": 2, "answer": "DO_NOT_LEAK_GOLD",
        }

    def require_membership(self):
        self.assertTrue(callable(getattr(teacher, "membership_probability", None)),
                        "candidate yes/no membership probability is not implemented")

    def args(self, **updates):
        fields = dict(
            model=Path("/models/fake"), teacher_id="toy-teacher",
            teacher_view="direct", checkpoint_id="sha:toy-weights",
            yes_token="Yes", no_token="No", q_transform="membership",
            truncated_policy="exclude", allow_legacy_reasoning_cache=False,
            samples=2, shard=0, shards=1, limit=0, resume=False,
            readout_batch_size=3, batch_size=2, store_logits=True,
            progress_every=0, seed=0, temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=100, max_model_len=8192, max_input_tokens=8192,
        )
        fields.update(updates)
        return SimpleNamespace(**fields)

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def chain(self, args, index, finish="stop"):
        return teacher.chain_record(
            self.row, index, "chain one" if index == 0 else "chain two", finish,
            10, 20, provenance=teacher.chain_provenance(args, self.row))

    def run_readout(self, args=None, engine=None, chains=None):
        self.require_membership()
        args = args or self.args()
        engine = engine or FakeEngine()
        context = tempfile.TemporaryDirectory()
        self.addCleanup(context.cleanup)
        root = Path(context.name)
        input_path = root / "input.jsonl"
        reasoning_path = root / "reasoning.jsonl"
        output_path = root / "part-00000.jsonl"
        self.write_jsonl(input_path, [self.row])
        self.write_jsonl(reasoning_path, chains if chains is not None else [
            self.chain(args, 0), self.chain(args, 1)])
        profile = teacher.readout_split(engine, self.tokenizer, self.labels, args,
                                        input_path, [reasoning_path], output_path)
        records = teacher.read_jsonl(output_path)
        self.assertEqual(len(records), 1, "write one result per input, including failures")
        return records[0], profile, engine, output_path

    def test_cli_defaults_to_membership(self):
        args = teacher.parse_args([])
        self.assertEqual(args.q_transform, "membership")

    def test_native_readout_and_target_provenance_support_generation_audit(self):
        # Actual harness schema through a fake engine; no neural execution.
        from audit_teacher_generations import _signature
        from build_multiteacher_targets import aggregate_row
        self.row["split"] = "train"
        args = self.args()
        record, _, _, path = self.run_readout(args=args)
        signature, known = _signature(record)
        self.assertTrue(known)
        self.assertEqual(signature["generation"]["max_model_len"], args.max_model_len)
        teachers = [{"id": args.teacher_id, "weight": 1., "normalized_weight": 1.,
                     "path": path, "file_sha256": teacher.file_sha256(path)}]
        target = aggregate_row(self.row, {args.teacher_id: {self.row["id"]: record}}, teachers)
        component = target["teacher_provenance"][0]
        from_target, target_known = _signature({**component, "id": target["id"],
                                              "n_chains": component["requested_chains"]})
        self.assertTrue(target_known)
        self.assertEqual(signature, from_target)

    def test_resume_rejects_changed_generation_or_readout_context_budget(self):
        args = self.args()
        record, _, _, path = self.run_readout(args=args)
        with self.assertRaisesRegex(ValueError, "readout token budget"):
            teacher.validate_readout_record(record, args=self.args(max_input_tokens=9999))
        reasoning = path.parent / "budget_chains.jsonl"
        self.write_jsonl(reasoning, [self.chain(args, 0)])
        with self.assertRaisesRegex(ValueError, "max_model_len mismatch"):
            teacher.load_done_chains(reasoning, expected_identity=teacher.teacher_identity(args),
                                     input_rows={self.row["id"]: self.row},
                                     args=self.args(max_model_len=9999))

    def test_default_teacher_id_distinguishes_views_of_one_checkpoint(self):
        direct = teacher.parse_args(["--model", "/models/fake", "--teacher-view", "direct"])
        verify = teacher.parse_args(["--model", "/models/fake", "--teacher-view", "verify"])
        self.assertEqual(teacher.teacher_identity(direct)["teacher_id"], "fake:direct")
        self.assertEqual(teacher.teacher_identity(verify)["teacher_id"], "fake:verify")

    def test_probability_is_yes_softmax_not_label_sigmoid(self):
        self.require_membership()
        self.assertAlmostEqual(teacher.membership_probability([math.log(9), 0]), 0.9)
        self.assertAlmostEqual(teacher.membership_probability([1003, 1000]),
                               teacher.membership_probability([3, 0]))
        self.assertAlmostEqual(teacher.membership_probability([-1000, 1000]), 0.0)
        for values in ([math.nan, 0], [0, math.inf], [0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                teacher.membership_probability(values)

    def test_binary_token_ids_are_resolved_and_checked(self):
        self.require_membership()
        info = teacher.binary_token_info(self.tokenizer, "Yes", "No")
        self.assertEqual(info, {"yes_token": "Yes", "no_token": "No",
                                "yes_token_id": 173, "no_token_id": 821})
        with self.assertRaises(ValueError):
            teacher.binary_token_info(self.tokenizer, "Yes yes", "No")
        with self.assertRaises(ValueError):
            teacher.binary_token_info(self.tokenizer, "Yes", "Yes")

    def test_full_context_candidate_prompt_excludes_gold(self):
        self.require_membership()
        prompt = teacher.make_membership_prompt(
            self.tokenizer, self.row, self.labels, "reasoning evidence", 1,
            teacher_view="verify")
        for item in ("state text", self.row["question"], "alpha description",
                     "beta description", "reasoning evidence",
                     "Candidate under evaluation:\n(B) b: beta description"):
            self.assertIn(item, prompt)
        self.assertIn("multiple", prompt.lower())
        self.assertIn("at least one", prompt.lower())
        self.assertNotIn("DO_NOT_LEAK_GOLD", prompt)
        self.assertNotIn("gold_count", prompt)
        self.assertTrue(prompt.endswith("Membership answer:\n"))
        with self.assertRaises(ValueError):
            teacher.make_membership_prompt(self.tokenizer, self.row, self.labels, "", 2)

    def test_views_use_different_reasoning_instructions(self):
        self.require_membership()
        direct = teacher.make_think_prefix(self.tokenizer, self.row, self.labels,
                                           teacher_view="direct")
        verify = teacher.make_think_prefix(self.tokenizer, self.row, self.labels,
                                           teacher_view="verify")
        self.assertNotEqual(direct, verify)
        self.assertIn("counterexample", verify.lower())
        for prompt in (direct, verify):
            self.assertIn(self.row["question"], prompt)
            self.assertNotIn("DO_NOT_LEAK_GOLD", prompt)

    def test_chain_membership_preserves_candidate_order_and_allows_multiple_yes(self):
        record, profile, engine, _ = self.run_readout()
        self.assertEqual(record["q_semantics"], "candidate_membership")
        self.assertEqual(record["option_keys"], ["a", "b"])
        self.assertEqual(record["per_chain_ids"], [0, 1])
        self.assertEqual(record["per_chain_membership"], [[0.9, 0.8], [0.75, 0.25]])
        self.assertEqual(record["q"], [0.825, 0.525])
        self.assertGreater(sum(record["per_chain_membership"][0]), 1.0)
        self.assertEqual(record["teacher_id"], "toy-teacher")
        self.assertEqual(record["teacher_view"], "direct")
        self.assertEqual(record["checkpoint"], "sha:toy-weights")
        self.assertEqual(record["readout"], "candidate_yes_no_v1")
        self.assertEqual(len(record["per_chain_metadata"]), 2)
        self.assertTrue(all(item["included"] for item in record["per_chain_metadata"]))
        self.assertEqual(profile["rows_with_q"], 1)
        self.assertEqual(len(engine.prompts), 4)

    def test_missing_or_failed_candidate_does_not_become_a_negative_target(self):
        for engine in (FakeEngine(fail_candidate="b"), FakeEngine(invalid_candidate="b")):
            with self.subTest(engine=engine.__dict__):
                record, profile, _, _ = self.run_readout(engine=engine)
                self.assertIsNone(record["q"])
                self.assertEqual(record["per_chain_membership"], [])
                self.assertEqual(record["valid_chains"], 0)
                self.assertEqual(profile["rows_zero_chain"], 1)
                self.assertEqual(len(record["excluded_chain_metadata"]), 2)
        record, _, _, _ = self.run_readout(chains=[])
        self.assertIsNone(record["q"])
        self.assertEqual(record["missing_chains"], 2)

    def test_truncated_policy_is_explicit_and_counted(self):
        self.require_membership()
        args = self.args()
        chains = [self.chain(args, 0, "length"), self.chain(args, 1)]
        excluded, profile, _, _ = self.run_readout(args=args, chains=chains)
        self.assertEqual(excluded["per_chain_ids"], [1])
        self.assertEqual(excluded["q"], [0.75, 0.25])
        self.assertEqual(profile["excluded_truncated_chains"], 1)
        self.assertTrue(excluded["excluded_chain_metadata"][0]["truncated"])
        included, profile, _, _ = self.run_readout(
            args=self.args(truncated_policy="include"), chains=chains)
        self.assertEqual(included["per_chain_ids"], [0, 1])
        self.assertEqual(profile["included_truncated_chains"], 1)

    def test_cache_rejects_wrong_view_and_duplicate_chain(self):
        self.require_membership()
        args = self.args()
        wrong = self.chain(self.args(teacher_view="verify"), 0)
        with self.assertRaisesRegex(ValueError, "teacher_view"):
            self.run_readout(args=args, chains=[wrong])
        duplicate = self.chain(args, 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_readout(args=args, chains=[duplicate, duplicate])

    def test_resume_rejects_invalid_q_and_stale_input(self):
        record, _, _, path = self.run_readout()
        args = self.args()
        expected = teacher.teacher_identity(args)
        record["q"][0] = math.nan
        self.write_jsonl(path, [record])
        with self.assertRaises(ValueError):
            teacher.load_done_ids(path, expected_identity=expected,
                                  input_rows={self.row["id"]: self.row}, args=args)
        record["q"] = [0.825, 0.525]
        self.write_jsonl(path, [record])
        changed = dict(self.row, question="Changed content with the same id")
        with self.assertRaisesRegex(ValueError, "input"):
            teacher.load_done_ids(path, expected_identity=expected,
                                  input_rows={changed["id"]: changed}, args=args)

    def test_resume_rejects_replaced_reasoning_cache(self):
        record, _, _, output = self.run_readout()
        reasoning_path = Path(record["provenance"]["reasoning_paths"][0])
        chains = teacher.read_jsonl(reasoning_path)
        chains[0]["reasoning"] = "new reasoning with the same input and chain id"
        chains[0]["reasoning_sha256"] = hashlib.sha256(chains[0]["reasoning"].encode()).hexdigest()
        self.write_jsonl(reasoning_path, chains)
        with self.assertRaisesRegex(ValueError, "reasoning"):
            teacher.readout_split(FakeEngine(), self.tokenizer, self.labels,
                                  self.args(resume=True), output.parent / "input.jsonl",
                                  [reasoning_path], output)

    def test_legacy_centered_output_is_only_diagnostic(self):
        self.require_membership()
        raw, centered, softmax = teacher.chain_probabilities([1, 1, 1], 3)
        self.assertTrue(all(0 < value < 1 for value in raw))
        self.assertAlmostEqual(sum(softmax), 1.0)
        self.assertAlmostEqual(sum(centered), 0.75)
        self.assertEqual(teacher.q_semantics("centered_sigmoid"),
                         "diagnostic_centered_sigmoid")

    def test_summary_reports_membership_and_excluded_truncation(self):
        args = self.args()
        record, profile, _, _ = self.run_readout(
            args=args, chains=[self.chain(args, 0, "length"), self.chain(args, 1)])
        report = teacher._summarize([record], [self.row], [profile])
        self.assertEqual(report.get("q_semantics"), "candidate_membership")
        self.assertEqual(report.get("truncation", {}).get("excluded"), 1)
        self.assertIn("candidate", report["method"].lower())

    def test_input_digest_matches_shared_contract_and_ignores_gold(self):
        self.require_membership()
        self.assertTrue(callable(getattr(teacher, "input_digest", None)),
                        "teacher must use the shared input digest")
        altered = dict(self.row, target=[0, 0], gold_count=0, id="other-id")
        self.assertEqual(teacher.input_fingerprint(self.row), teacher.input_digest(self.row))
        self.assertEqual(teacher.input_fingerprint(self.row), teacher.input_fingerprint(altered))
        record, _, _, _ = self.run_readout()
        self.assertEqual(record.get("input_digest"), teacher.input_digest(self.row))
        self.assertEqual(record["provenance"].get("input_digest"), record["input_digest"])

    def test_merge_rejects_duplicate_ids_instead_of_last_writer_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl"
            self.write_jsonl(path, [{"id": "same", "q": [0.1, 0.9]},
                                    {"id": "same", "q": [0.9, 0.1]}])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                teacher._merge_records([path], lambda row: row["id"])

    def test_merge_readout_writes_explicit_output_directory(self):
        record, _, _, _ = self.run_readout()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts, data, output = root / "parts", root / "data", root / "out"
            (parts / "train").mkdir(parents=True)
            data.mkdir()
            self.write_jsonl(data / "train.jsonl", [self.row])
            self.write_jsonl(parts / "train/part-00000.jsonl", [record])
            args = self.args(parts_root=parts, data_dir=data, output_dir=output,
                             summary=root / "summary.json")
            result = teacher.merge_readout(args)
            self.assertEqual(result, 0)
            self.assertTrue((output / "sii_teacher_train.jsonl").exists())
            self.assertFalse((data / "sii_teacher_train.jsonl").exists())

    def test_merge_reasoning_rejects_view_mixture(self):
        self.require_membership()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts, data = root / "parts", root / "data"
            (parts / "train").mkdir(parents=True)
            data.mkdir()
            self.write_jsonl(data / "train.jsonl", [self.row])
            self.write_jsonl(parts / "train/part-00000.jsonl",
                             [self.chain(self.args(teacher_view="verify"), 0)])
            with self.assertRaisesRegex(ValueError, "teacher_view"):
                teacher.merge_reasoning(self.args(parts_root=parts, data_dir=data,
                                                   output_dir=root / "out"))

    def test_runner_dry_run_uses_native_modes_and_isolated_view_paths(self):
        script = ROOT / "harness/run_sii_style_teacher.sh"
        self.assertTrue(script.exists(), "finite teacher shell runner is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_jsonl(root / "train.jsonl", [self.row])
            env = dict(os.environ, DRY_RUN="1", SHARDS="2", SAMPLES="2", SPLITS="train",
                       DATA_DIR=str(root), RUN_ROOT=str(root / "run"), LOG_DIR=str(root / "log"),
                       TEACHER_VIEW="verify", REASONING="1", READOUT="1", Q_TRANSFORM="membership",
                       CUDA_VISIBLE_DEVICES="0,1")
            result = subprocess.run(["bash", str(script)], env=env, text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line[len("DRY_RUN "):]) for line in result.stdout.splitlines()
                        if line.startswith("DRY_RUN ")]
            modes = []
            for command in commands:
                script_pos = next(index for index, value in enumerate(command)
                                  if value.endswith("sii_style_teacher.py"))
                parsed = teacher.parse_args(command[script_pos + 1:])
                modes.append(parsed.mode)
                self.assertEqual(parsed.teacher_view, "verify")
                self.assertEqual(parsed.q_transform, "membership")
            self.assertEqual(modes.count("generate"), 2)
            self.assertEqual(modes.count("readout"), 2)
            self.assertEqual(modes.count("merge-reasoning"), 1)
            self.assertEqual(modes.count("merge-readout"), 1)
            self.assertIn("/verify/q/membership/exclude/merged", result.stdout)

    def test_validate_only_checks_binary_tokens_and_membership_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_jsonl(root / "input.jsonl", [self.row])
            fake_transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda _path: self.tokenizer))
            args = self.args(input=[root / "input.jsonl"], report=root / "report.json")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}), \
                    mock.patch.object(teacher, "option_labels", return_value=self.labels):
                status = teacher.validate_only(args)
            self.assertEqual(status, 0)
            report = json.loads(args.report.read_text())
            self.assertEqual(report.get("q_semantics"), "candidate_membership")
            self.assertEqual(report.get("binary_tokens", {}).get("yes_token_id"), 173)
            sample = report["inputs"][str(root / "input.jsonl")]["prompt_sample"]
            self.assertIn("Candidate under evaluation", sample["read_prompt_with_reasoning"])
            self.assertEqual(report.get("model_forward_executed"), False)

    def test_fake_generation_records_the_view_and_chain_provenance(self):
        args = self.args(teacher_view="verify")
        prompts_seen = []
        class FakeLLM:
            def generate(self, prompts, sampling, **_kwargs):
                prompts_seen.extend(prompts)
                return [SimpleNamespace(outputs=[
                    SimpleNamespace(text="chain one</think>", finish_reason="stop", token_ids=[1, 2]),
                    SimpleNamespace(text="chain two", finish_reason="length", token_ids=[3, 4, 5]),
                ]) for _prompt in prompts]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_jsonl(root / "input.jsonl", [self.row])
            teacher.generate_split(FakeLLM(), object(), self.tokenizer, self.labels,
                                   args, root / "input.jsonl", root / "part.jsonl")
            rows = teacher.read_jsonl(root / "part.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["truncated"] for row in rows], [False, True])
            self.assertTrue(all(row["teacher_view"] == "verify" for row in rows))
            self.assertEqual(rows[0]["input_digest"], teacher.input_digest(self.row))
            self.assertEqual(rows[0]["reasoning"], "chain one")
            self.assertIn("counterexample", prompts_seen[0])


if __name__ == "__main__":
    unittest.main()
