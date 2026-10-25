"""Identifiability and causality tests for the shift-by-k (v4) generator."""

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import gen_text_episodes_shift as g


class ShiftIdentifiabilityTests(unittest.TestCase):
    def test_every_two_active_state_distinguishes_every_shift_pair(self):
        n = len(g.CELLS)
        self.assertEqual(n % 2, 1, "odd cell count required")
        for pair in itertools.combinations(range(n), g.ACTIVE_COUNT):
            active = [int(i in pair) for i in range(n)]
            for k in range(1, n):
                for kp in range(1, n):
                    if k == kp:
                        continue
                    self.assertNotEqual(
                        g.shift_set(active, k),
                        g.shift_set(active, kp),
                        f"ambiguous shift {k} vs {kp} on state {active}",
                    )

    def test_no_absorbing_full_or_empty_set(self):
        n = len(g.CELLS)
        active = [1] * n
        for k in range(1, n):
            self.assertEqual(g.shift_set(active, k), active)  # full set is invariant
            self.assertNotEqual(sum(g.shift_set(active, k)), g.ACTIVE_COUNT)
        # The generator never emits a full/empty state because it keeps
        # ACTIVE_COUNT cells active throughout.

    def test_generate_and_validate_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "episodes.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(g.__file__)),
                    "--output",
                    str(out),
                    "--episodes",
                    "16",
                    "--horizon",
                    "5",
                    "--rule-switch-prob",
                    "0.5",
                    "--seed",
                    "7",
                ],
                check=True,
                capture_output=True,
            )
            episodes = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(episodes), 16)
            for episode in episodes:
                g.validate_episode(episode)

    def test_rule_change_uses_a_distinct_shift(self):
        import random

        rng = random.Random(0)
        episode = g.gen_episode(rng, horizon=12, rule_switch_prob=1.0)
        rules = [step["oracle_rule_before_decision"] for step in episode["steps"]]
        for a, b in zip(rules, rules[1:]):
            self.assertNotEqual(a, b, "rule change must switch to a different shift")


if __name__ == "__main__":
    unittest.main()
