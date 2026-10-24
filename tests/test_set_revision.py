"""Unit tests for set-revision targets and metrics (dependency-free)."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from set_revision import (
    revision_count_target,
    revision_membership_target,
    revision_metrics,
    revision_target,
)


class RevisionTargetTests(unittest.TestCase):
    def test_add_remove_keep_decomposition(self):
        prev, new = [1, 1, 0, 0], [1, 0, 1, 0]
        target = revision_target(prev, new)
        self.assertEqual(target["additions"], [0, 0, 1, 0])
        self.assertEqual(target["removals"], [0, 1, 0, 0])
        self.assertEqual(target["unchanged"], [1, 0, 0, 1])
        self.assertEqual(target["n_added"], 1)
        self.assertEqual(target["n_removed"], 1)

    def test_identity_transition_has_no_revision(self):
        prev = [1, 0, 1]
        target = revision_target(prev, prev)
        self.assertEqual(target["additions"], [0, 0, 0])
        self.assertEqual(target["removals"], [0, 0, 0])
        self.assertEqual(target["n_added"], 0)

    def test_membership_target_signs(self):
        prev, new = [1, 0, 1], [1, 1, 0]
        self.assertEqual(revision_membership_target(prev, new), [0, 1, -1])
        self.assertEqual(revision_count_target(prev, new), 2)

    def test_rejects_nonbinary_or_mismatched(self):
        with self.assertRaises(ValueError):
            revision_target([1, 0], [1, 2])
        with self.assertRaises(ValueError):
            revision_target([1, 0], [1])


class RevisionMetricsTests(unittest.TestCase):
    def test_perfect_revision(self):
        prev = [[1, 0], [1, 0]]
        new = [[1, 1], [0, 0]]
        pred = new
        metrics = revision_metrics(prev, new, pred, trajectory_ids=[0, 1])
        self.assertEqual(metrics["repair_rate"], 1.0)
        self.assertEqual(metrics["broken_rate"], 0.0)
        self.assertEqual(metrics["trajectory_success"], 1.0)
        self.assertEqual(metrics["additions"]["precision"], 1.0)
        self.assertEqual(metrics["removals"]["precision"], 1.0)

    def test_broken_and_repair_counts(self):
        prev = [[1, 0], [1, 1], [0, 0]]  # step0 correct, step1 correct, step2 wrong
        new = [[1, 0], [1, 1], [1, 0]]  # gold
        pred = [[0, 0], [1, 1], [1, 0]]  # step0 broken, step1 held, step2 repaired
        metrics = revision_metrics(prev, new, pred, trajectory_ids=[0, 0, 1])
        self.assertEqual(metrics["broken_rate"], 0.5)  # 1 broken of 2 held
        self.assertEqual(metrics["repair_rate"], 1.0)  # 1 repaired of 1 revisable
        self.assertEqual(metrics["trajectory_success"], 0.5)  # traj0 broken, traj1 ok

    def test_addition_removal_prf(self):
        prev = [[0, 0], [1, 1]]
        new = [[1, 0], [0, 1]]  # add opt0, then swap
        pred = [[1, 1], [0, 1]]  # over-adds in step0, perfect step1
        metrics = revision_metrics(prev, new, pred)
        # step0: true add {0}, pred add {0,1} -> 1 tp, 1 fp
        self.assertEqual(metrics["additions"]["precision"], 0.5)
        self.assertEqual(metrics["additions"]["recall"], 1.0)
        # removals: step0 none, step1 true remove {0}, pred remove {0} -> 1 tp
        self.assertEqual(metrics["removals"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
