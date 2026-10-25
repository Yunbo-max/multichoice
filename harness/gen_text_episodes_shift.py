#!/usr/bin/env python3
"""Generate shift-by-k rule episodes (harder than left/right, v4).

The v3 environment only had two rules (left/right), so memory only had to store
one bit. This harder environment uses a cyclic shift by k over N=7 cells, giving
N-1 = 6 distinct rules; a demonstration shows one input->output pair (one-hot at
position 0 -> position k) that uniquely determines k, and the agent must retain
the SPECIFIC k (log2 6 bits) across a long horizon and update it when the rule
changes to a different shift.

N=7 is prime, so a shift by any nonzero k is never the identity on a 2-element
active set: every reachable state distinguishes every pair of rules (no symmetric
or absorbing shortcuts). ACTIVE_COUNT stays 2 but the candidate space grows to
C(7,2)=21 pairs.

Output schema matches ``build_episodic_features.py`` (unchanged from v3).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

CELLS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"]
ACTIVE_COUNT = 2
SCHEMA = "cyclic7_shift2_v4"


def shift_set(active, k):
    """Shift every active cell forward by k (mod N)."""
    n = len(active)
    return [active[(i - k) % n] for i in range(n)]


def names(active):
    return [CELLS[i] for i, bit in enumerate(active) if bit]


def active_text(active):
    picked = names(active)
    return "none" if not picked else ", ".join(picked)


def demo_text(k):
    n = len(CELLS)
    inp = [int(i == 0) for i in range(n)]
    out = shift_set(inp, k)
    return (
        f"Known shift-rule example: if only {CELLS[0]} is active, "
        f"the next active component is {active_text(out)}."
    )


def state_text(active):
    return f"The active components are: {active_text(active)}."


def option_block():
    return [
        {"key": chr(ord("a") + i), "desc": f"Activate {CELLS[i]}"}
        for i in range(len(CELLS))
    ]


def option_ids():
    return [f"activate-{cell.lower()}" for cell in CELLS]


def distinct_rule(rng, current):
    choices = [k for k in range(1, len(CELLS)) if k != current]
    return rng.choice(choices)


def gen_episode(rng, horizon, rule_switch_prob):
    k = rng.choice(range(1, len(CELLS)))  # k in 1..N-1
    demos = [demo_text(k)]
    active = [0] * len(CELLS)
    for i in rng.sample(range(len(CELLS)), ACTIVE_COUNT):
        active[i] = 1
    steps = []
    for t in range(horizon):
        rule_at_decision = k
        gold = shift_set(active, rule_at_decision)
        executed = gold
        feedback = (
            f"After this decision the active components are {active_text(executed)}."
        )
        if t < horizon - 1 and rng.random() < rule_switch_prob:
            k = distinct_rule(rng, k)
            feedback += " The rule changed. " + demo_text(k)
        steps.append(
            {
                "step_index": t,
                "oracle_rule_before_decision": rule_at_decision,
                "decision": {
                    "state": state_text(active),
                    "question": "Which components should be active now?",
                    "options": option_block(),
                    "selection_mode": "multi",
                    "allow_empty": False,
                    "allowed_cardinalities": [ACTIVE_COUNT],
                },
                "action_ids": option_ids(),
                "gold": gold,
                "executed_set": executed,
                "post_observation": feedback,
                "done": t == horizon - 1,
            }
        )
        active = gold
    return {
        "environment_schema": SCHEMA,
        "episode_id": f"ep-{rng.getrandbits(32):08x}",
        "demonstrations": demos,
        "steps": steps,
    }


def validate_episode(episode):
    if episode.get("environment_schema") != SCHEMA:
        raise ValueError("wrong generator version")
    if not isinstance(episode.get("episode_id"), str) or not episode["episode_id"]:
        raise ValueError("episode_id must be nonempty")
    demos = episode.get("demonstrations", [])
    if not isinstance(demos, list) or any(not isinstance(x, str) for x in demos):
        raise ValueError("demonstrations must be strings")
    sequence = episode.get("steps")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("steps must be a nonempty chronological list")
    k = len(CELLS)
    if k % 2 == 0:
        raise ValueError("shift benchmark requires odd cell count")
    for t, step in enumerate(sequence):
        if step.get("oracle_rule_before_decision") not in range(1, k):
            raise ValueError("missing private pre-decision shift audit field")
        if step.get("step_index") != t:
            raise ValueError("step_index must equal chronological position")
        if step.get("decision", {}).get("allowed_cardinalities") != [ACTIVE_COUNT]:
            raise ValueError("environment must publicly enforce fixed count")
        ids = step.get("action_ids")
        if not isinstance(ids, list) or len(ids) != k or len(set(ids)) != k:
            raise ValueError("action_ids must be K distinct strings")
        for field in ("gold", "executed_set"):
            value = step.get(field)
            if not (
                isinstance(value, list)
                and len(value) == k
                and all(type(x) is int and x in (0, 1) for x in value)
            ):
                raise ValueError(f"{field} must be K-length multi-hot")
            if sum(value) != ACTIVE_COUNT:
                raise ValueError(f"{field} must select exactly {ACTIVE_COUNT} actions")
        if step.get("done") is not (t == len(sequence) - 1):
            raise ValueError("done must be true only for the final step")
    if len(demos) != 1:
        raise ValueError("one initial demonstration required")
    known_rule = sequence[0]["oracle_rule_before_decision"]
    if demos[0] != demo_text(known_rule):
        raise ValueError("initial demonstrated rule does not match oracle")
    # Reconstruct the exact trajectory from the visible state + private rule.
    candidates = []
    for pair in itertools.combinations(range(k), ACTIVE_COUNT):
        bits = [int(i in pair) for i in range(k)]
        if state_text(bits) == sequence[0]["decision"]["state"]:
            candidates.append(bits)
    if len(candidates) != 1:
        raise ValueError("invalid initial active state")
    active = candidates[0]
    for t, step in enumerate(sequence):
        state = step["decision"]
        if (
            state.get("state") != state_text(active)
            or state.get("selection_mode") != "multi"
            or state.get("allow_empty") is not False
            or state.get("options") != option_block()
            or step.get("action_ids") != option_ids()
        ):
            raise ValueError("decision state/action protocol diverges from simulator")
        rule = step["oracle_rule_before_decision"]
        # Identifiability: every reachable 2-active state must distinguish every
        # pair of distinct shifts (no symmetric / absorbing shortcut).
        for other in range(1, k):
            if other != rule and shift_set(active, rule) == shift_set(active, other):
                raise ValueError("ambiguous state: two shifts imply the same answer")
        expected = shift_set(active, rule)
        if step["gold"] != expected or step["executed_set"] != expected:
            raise ValueError("oracle action/label disagrees with state and rule")
        feedback = (
            "After this decision the active components are "
            + active_text(step["executed_set"])
            + "."
        )
        if t < len(sequence) - 1:
            next_rule = sequence[t + 1]["oracle_rule_before_decision"]
            if next_rule != rule:
                feedback += " The rule changed. " + demo_text(next_rule)
        if step.get("post_observation") != feedback:
            raise ValueError("post-action feedback leaks or omits a rule change")
        active = step["executed_set"]
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--rule-switch-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("episodes and horizon must be positive")
    if not 0.0 <= args.rule_switch_prob <= 1.0:
        parser.error("rule-switch-prob must be in [0,1]")
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for episode_index in range(args.episodes):
            episode = gen_episode(rng, args.horizon, args.rule_switch_prob)
            episode["episode_id"] = f"ep-{args.seed}-{episode_index:08d}"
            validate_episode(episode)
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
