"""CPU fixture checks for planning/evidence auditing; no neural training runs."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import plan_self_distillation_rounds as planner
import audit_teaching_utility as utility
from multiselect_metrics import compute_metrics
from set_distillation import TARGET_SCHEMA, input_digest


def identity(number):
    return "sha256:" + f"{number:064x}"


def write(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value))
    return path


def row(ident, split, answer=0):
    return {"id": ident, "split": split, "state": "passage " + ident,
            "question": "Choose the right item", "source": "fixture", "lang": "en",
            "selection_mode": "single", "allow_empty": False,
            "options": [{"key": key, "desc": key + " description"} for key in ("A", "B", "C")],
            "target": [int(i == answer) for i in range(3)], "n_options": 3,
            "group_id": "group-" + ident}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.train = [row("t0", "train"), row("t1", "train", 1), row("t2", "train", 2)]
        self.dev = [row("d0", "dev"), row("d1", "dev", 1)]
        write(self.base / "train.jsonl", self.train, True)
        write(self.base / "dev.jsonl", self.dev, True)
        write(self.base / "modules.json", ["linear"])
        self.config = {"experiment_id": "fixture", "repository": str(ROOT), "run_root": "runs",
                       "train": "train.jsonl", "dev": "dev.jsonl", "modules": "modules.json",
                       "base_checkpoint": {"path": "model", "checkpoint_id": identity(0)},
                       "rounds": 2, "probe_seeds": [43, 44],
                       "training": {"global_batch": 2, "world_size": 1, "epochs": 2, "max_steps": 2}}
        self.path = self.base / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, bindings=None):
        write(self.path, self.config)
        return planner.build_plan(self.path, self.base / "plan", bindings)

    def test_fixed_and_refresh_lineage_and_common_probe_initialization(self):
        plan, manifests = self.plan()
        stages = {stage["id"]: stage for stage in plan["stages"]}
        fixed = stages["student.fixed.r2.train"]
        refresh = stages["student.refreshed.r2.train"]
        self.assertEqual((fixed["initialization"], fixed["teacher"]), ("fixed.r1", "base"))
        self.assertEqual((refresh["initialization"], refresh["teacher"]), ("refreshed.r1", "refreshed.r1"))
        for stage in stages.values():
            if stage["id"].startswith("student.") and stage["id"].endswith(".train"):
                self.assertNotIn("--resume", stage["argv"])
                self.assertEqual(stage["argv"][stage["argv"].index("--eval-every-frac") + 1], "1.0")
                if stage["id"].startswith(("student.from.", "student.gold.")):
                    self.assertEqual(stage["initialization"], "base")
                    self.assertEqual(stage["argv"][stage["argv"].index("--init-checkpoint-id") + 1], identity(0))
        self.assertEqual(len(manifests["utility_results_manifest.json"]["runs"]), 8)
        self.assertEqual([g["id"] for g in manifests["generation_audit_manifest.json"]["generations"]],
                         ["base", "refreshed.r1", "refreshed.r2"])
        self.assertEqual([t["id"] for t in manifests["manifests/base.dev.json"]["teachers"]], ["direct", "verify"])
        self.assertEqual([t["id"] for t in manifests["manifests/refreshed.r1.dev.json"]["teachers"]], ["direct", "verify"])
        self.assertFalse(stages["audit.teacher_generations"]["manifest_bindings_resolved"])
        self.assertFalse(plan["executed"])
        planner.validate_dag(plan["stages"])

    def test_future_identities_are_not_fabricated(self):
        plan, _ = self.plan()
        generated = next(s for s in plan["stages"] if s["id"] == "teacher.refreshed.r1.direct.train.generate")
        self.assertFalse(generated["argv_resolved"])
        plan2, _ = self.plan({"refreshed.r1": identity(1)})
        generated = next(s for s in plan2["stages"] if s["id"] == generated["id"])
        self.assertTrue(generated["argv_resolved"])
        self.assertIn(identity(1), generated["argv"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.plan({"nonexistent": identity(9)})

    def test_dev_diagnostic_has_no_strict_target_prerequisite(self):
        plan, manifests = self.plan()
        stages = {stage["id"]: stage for stage in plan["stages"]}
        audit = stages["audit.teacher_generations"]
        self.assertTrue(audit["diagnostic_accepts_partial_inputs"])
        self.assertIn("does not mark failed stages successful", audit["failed_dependency_handling"])
        self.assertFalse(any(name.endswith(".dev.targets") for name in stages))
        for key in ("base", "refreshed.r1", "refreshed.r2"):
            self.assertIn(f"manifests/{key}.dev.json", manifests)
            for view in ("direct", "verify"):
                self.assertIn(f"teacher.{key}.{view}.dev.readout", audit["depends_on"])
            self.assertFalse(any("targets" in dependency for dependency in audit["depends_on"]))
        for stage in stages.values():
            if stage["id"].endswith(".train.targets"):
                self.assertTrue(stage["requires_full_coverage"])
                self.assertEqual(stage["argv"][stage["argv"].index("--missing") + 1], "error")
            if stage["id"].startswith("student.") and stage["id"].endswith(".train") and stage["teacher"] is not None:
                self.assertEqual(stage["argv"][stage["argv"].index("--target-missing") + 1], "error")
                self.assertIn(f"teacher.{stage['teacher']}.train.targets", stage["depends_on"])

    def test_mutable_alias_is_rejected(self):
        self.config["base_checkpoint"]["checkpoint_id"] = "latest"
        with self.assertRaisesRegex(ValueError, "checkpoint_id"):
            self.plan()

    def test_unavailable_update_budget_is_rejected(self):
        self.config["training"]["max_steps"] = 5
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.plan()

    def test_input_and_group_leakage_are_rejected(self):
        self.dev[0]["group_id"] = self.train[0]["group_id"]
        write(self.base / "dev.jsonl", self.dev, True)
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            self.plan()
        self.dev[0] = {**self.train[0], "id": "d0", "split": "dev", "group_id": "devgroup"}
        write(self.base / "dev.jsonl", self.dev, True)
        with self.assertRaisesRegex(ValueError, "content overlaps"):
            self.plan()

    def test_mislabeled_training_split_is_rejected(self):
        self.train[0]["split"] = "test"
        write(self.base / "train.jsonl", self.train, True)
        with self.assertRaisesRegex(ValueError, "split"):
            self.plan()

    def test_legacy_multirc_passage_overlap_is_rejected(self):
        self.train[0].update(source="multirc", state="A shared long passage", question="First question")
        self.dev[0].update(source="multirc", state="A  shared long passage", question="Different question")
        self.train[0].pop("group_id")
        self.dev[0].pop("group_id")
        write(self.base / "train.jsonl", self.train, True)
        write(self.base / "dev.jsonl", self.dev, True)
        with self.assertRaisesRegex(ValueError, "passage/group overlap"):
            self.plan()

    def test_bad_dag_rejected(self):
        with self.assertRaisesRegex(ValueError, "dependency"):
            planner.validate_dag([{"id": "a", "depends_on": ["b"], "argv": ["python"]},
                                  {"id": "b", "depends_on": ["a"], "argv": ["python"]}])


class TeachingUtilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.train = [row("t0", "train"), row("t1", "train", 1)]
        self.dev = [row("d0", "dev"), row("d1", "dev", 1)]
        self.train_path = write(self.base / "train.jsonl", self.train, True)
        self.dev_path = write(self.base / "dev.jsonl", self.dev, True)
        self.modules = write(self.base / "modules.json", ["linear"])
        self.manifest = {"schema": planner.UTILITY_SCHEMA, "input": str(self.dev_path), "runs": []}
        self.manifest_path = self.base / "utility.json"

    def tearDown(self):
        self.tmp.cleanup()

    def add_run(self, name, teacher=None, seed=43, correct=(True, True)):
        folder = self.base / name
        evaluation = folder / "eval_dev"
        checkpoint = identity(100 + len(self.manifest["runs"]))
        teacher_id = None if teacher is None else identity(teacher)
        config = dict.fromkeys(utility.MATCHED_FIELDS)
        config.update({"train": str(self.train_path), "dev": str(self.dev_path), "modules": str(self.modules),
                       "train_sha256": planner.sha256_file(self.train_path), "dev_sha256": planner.sha256_file(self.dev_path),
                       "modules_sha256": planner.sha256_file(self.modules), "model": str(self.base / "initial"),
                       "loaded_model": str(self.base / "initial"), "resume": None, "init_checkpoint_id": identity(0),
                       "checkpoint_identity_kind": "explicit", "objective": "set", "set_loss": "set_ce",
                       "count_weight": 1.0, "count_target": "teacher", "lambda_g": 0, "graph": "none", "lr": 2e-6,
                       "global_batch": 2, "epochs": 2, "max_input_tokens": 8192, "max_batch_tokens": 8192,
                       "eval_every_frac": 1.0, "max_steps": 2, "limit": 0, "limit_dev": 0, "world_size": 1,
                       "planned_steps": 2, "stop_step": 2, "start_step": 0, "grad_clip": 1.0,
                       "lora": False, "lora_r": 64, "lora_alpha": 128, "gold_weight": .25, "seed": seed,
                       "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
                       "schedule": "lr_factor(step, planned_steps); fresh optimizer per invocation",
                       "target_mode": "gold" if teacher is None else "file", "target_file": None,
                       "targets": {"missing": 0}})
        target_path = None
        if teacher is not None:
            target_path = self.base / f"teacher{teacher}.targets.jsonl"
            targets = [{"schema": TARGET_SCHEMA, "id": item["id"], "q_semantics": "set_membership",
                        "source_split": "train",
                        "q": list(item["target"]), "count_probs": [0., 1., 0., 0.], "allowed_counts": [1],
                        "option_keys": ["A", "B", "C"], "input_digest": input_digest(item),
                        "teacher_provenance": [{"checkpoint": teacher_id, "teacher_id": "direct",
                                                "teacher_view": "direct", "readout": "candidate_yes_no_v1",
                                                "requested_chains": 2, "normalized_weight": 1.0,
                                                "per_chain_metadata": [{"included": True,
                                                    "generation": {"seed": 0, "temperature": .6, "top_p": .95,
                                                                   "top_k": 20, "max_tokens": 1024, "max_model_len": 8192}}],
                                                "provenance": {"checkpoint_identity_kind": "explicit",
                                                    "reasoning_prompt_version": "fixture_v1", "readout_prompt_version": "candidate_yes_no_v1",
                                                    "readout_max_input_tokens": 8192, "truncated_policy": "exclude",
                                                    "binary_tokens": {"yes_token": "Yes", "no_token": "No", "yes_token_id": 5, "no_token_id": 6}}}]}
                       for item in self.train]
            write(target_path, targets, True)
            config.update(target_file=str(target_path), target_file_sha256=planner.sha256_file(target_path))
        complete = {"steps": 2, "start_step": 0, "selected": {"step": 2}}
        eval_config = {"input_sha256": planner.sha256_file(self.dev_path), "expected_split": "dev", "decoder": "provided",
                       "threshold": .5, "max_input_tokens": 8192, "max_batch_tokens": 8192, "batch_size": 8,
                       "mode": "checkpoint-evaluation", "checkpoint_id": checkpoint, "checkpoint_identity_kind": "explicit",
                       "model": str(folder / "final"), "threshold_fitted_on_input": False}
        predictions = []
        for item, is_correct in zip(self.dev, correct):
            gold = item["target"].index(1)
            choice = gold if is_correct else (gold + 1) % 3
            predictions.append({"id": item["id"], "p": [.8 if i == choice else .1 for i in range(3)],
                                "predicted_indices": [choice], "count_probs": [0., 1., 0., 0.], "decoder": "set_map",
                                "input_digest": input_digest(item), "option_keys": ["A", "B", "C"],
                                "checkpoint_id": checkpoint, "provenance": {"checkpoint_identity_kind": "explicit"}})
        write(folder / "training_config.json", config)
        write(folder / "training_complete.json", complete)
        write(evaluation / "evaluation_config.json", eval_config)
        write(evaluation / "predictions.jsonl", predictions, True)
        write(evaluation / "metrics.json", compute_metrics(self.dev, predictions, decoder="provided"))
        run = {"id": name, "arm": "gold" if teacher is None else "teacher", "seed": seed,
               "teacher_key": None if teacher is None else f"teacher{teacher}", "teacher_checkpoint_id": teacher_id,
               "checkpoint_id": checkpoint, "training_dir": str(folder), "evaluation_dir": str(evaluation),
               "teacher_targets": str(target_path) if target_path else None}
        self.manifest["runs"].append(run)
        return folder

    def audit(self, allow_planned=False):
        write(self.manifest_path, self.manifest)
        return utility.audit(self.manifest_path, allow_planned=allow_planned)

    def alter(self, path, **values):
        content = json.loads(path.read_text())
        content.update(values)
        write(path, content)

    def test_teaching_utility_is_next_student_gain_not_teacher_score(self):
        for seed in (43, 44):
            self.add_run(f"gold{seed}", seed=seed, correct=(True, False))
            self.add_run(f"good{seed}", teacher=1, seed=seed, correct=(True, True))
            self.add_run(f"bad{seed}", teacher=2, seed=seed, correct=(False, False))
        result = self.audit()
        self.assertTrue(result["teaching_utility_measured"])
        self.assertEqual(result["teacher_gain_over_gold"][identity(1)]["mean_delta_question_em"], .5)
        self.assertEqual(result["teacher_gain_over_gold"][identity(2)]["mean_delta_question_em"], -.5)
        pair = result["teacher_pairwise_comparisons"][0]
        self.assertEqual(pair["mean_delta_question_em"], -1.)
        self.assertEqual(pair["n_seeds"], 2)

    def test_different_initialization_is_confounded(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        self.alter(folder / "training_config.json", init_checkpoint_id=identity(7))
        with self.assertRaisesRegex(ValueError, "confounded.*init_checkpoint_id"):
            self.audit()

    def test_early_selected_checkpoint_is_not_matched_budget(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        self.alter(folder / "training_complete.json", selected={"step": 1})
        with self.assertRaisesRegex(ValueError, "terminal checkpoint"):
            self.audit()

    def test_resume_not_allowed(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        self.alter(folder / "training_config.json", resume="some_checkpoint")
        with self.assertRaisesRegex(ValueError, "fresh optimizer"):
            self.audit()

    def test_optimizer_change_is_confounded(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        self.alter(folder / "training_config.json", optimizer={"name": "SGD"})
        with self.assertRaisesRegex(ValueError, "confounded.*optimizer"):
            self.audit()

    def test_generation_budget_change_is_confounded(self):
        self.add_run("gold")
        self.add_run("teacher1", teacher=1)
        folder = self.add_run("teacher2", teacher=2)
        path = Path(self.manifest["runs"][-1]["teacher_targets"])
        targets = [json.loads(line) for line in path.read_text().splitlines()]
        for target in targets:
            target["teacher_provenance"][0]["per_chain_metadata"][0]["generation"]["max_tokens"] = 2048
        write(path, targets, True)
        self.alter(folder / "training_config.json", target_file_sha256=planner.sha256_file(path))
        with self.assertRaisesRegex(ValueError, "confounded teacher treatment"):
            self.audit()

    def test_teacher_checkpoint_contamination_rejected(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        path = Path(self.manifest["runs"][-1]["teacher_targets"])
        targets = [json.loads(line) for line in path.read_text().splitlines()]
        targets[0]["teacher_provenance"][0]["checkpoint"] = identity(2)
        write(path, targets, True)
        self.alter(folder / "training_config.json", target_file_sha256=planner.sha256_file(path))
        with self.assertRaisesRegex(ValueError, "contamination"):
            self.audit()

    def test_prediction_input_digest_rejected(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        path = folder / "eval_dev/predictions.jsonl"
        items = [json.loads(line) for line in path.read_text().splitlines()]
        items[0]["input_digest"] = "wrong"
        write(path, items, True)
        with self.assertRaisesRegex(ValueError, "input identity"):
            self.audit()

    def test_cherry_picked_summary_rejected(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        self.alter(folder / "eval_dev/metrics.json", question_em=.5)
        with self.assertRaisesRegex(ValueError, "saved metrics disagree"):
            self.audit()

    def test_missing_seed_arm_rejected(self):
        self.add_run("gold43")
        self.add_run("teacher43", teacher=1)
        self.add_run("gold44", seed=44)
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            self.audit()

    def test_incomplete_evidence_rejected_even_allow_planned(self):
        self.add_run("gold")
        folder = self.add_run("teacher", teacher=1)
        (folder / "training_complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            self.audit(allow_planned=True)

    def test_unexecuted_plan_is_unmeasured_only_with_explicit_flag(self):
        self.manifest["runs"] = [
            {"id": "gold", "arm": "gold", "seed": 43, "training_dir": "future/gold", "evaluation_dir": "future/gold/eval"},
            {"id": "teacher", "arm": "teacher", "seed": 43, "teacher_key": "t0",
             "training_dir": "future/student", "evaluation_dir": "future/student/eval"}]
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            self.audit()
        report = self.audit(allow_planned=True)
        self.assertFalse(report["teaching_utility_measured"])
        self.assertEqual(report["status"], "planned")


if __name__ == "__main__":
    unittest.main()
