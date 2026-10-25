#!/usr/bin/env python3
"""Generate textual rule-retention / rule-change episodes (JSONL) for the Qwen
episodic pipeline (`build_episodic_features.py`).

A deterministic cyclic rule over N cells rotates the active set left or right.
A demonstration shows the rule once at episode start. With ``--rule-switch-prob``
the rule flips mid-episode and a new demonstration is appended to that step's
post-action observation. Each decision is multi-select over the cells ("which
components should be active now?"), so the episode is a verifiable dynamic
multi-set task, not a static CMB/MultiRC row.

Output schema matches ``build_episodic_features.py`` (one episode per JSONL line;
``action_ids`` are K distinct semantic option IDs; ``gold``/``executed_set`` are
K-length multi-hot lists).
"""

from __future__ import annotations

import argparse
import json
import random
import itertools
from pathlib import Path

# Five positions eliminate left/right ambiguity for every proper nonempty set.
CELLS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
ACTIVE_COUNT = 2


def rotate(active, direction):
    n = len(active)
    return [active[(i - direction) % n] for i in range(n)]


def names(active):
    return [CELLS[i] for i, bit in enumerate(active) if bit]


def active_text(active):
    picked = names(active)
    return "none" if not picked else ", ".join(picked)


def demo_text(direction):
    n = len(CELLS)
    inp = [int(i == 0) for i in range(n)]
    out = rotate(inp, direction)
    # Do not leak the symbolic answer "left" or "right". Only show a
    # visible input/output demonstration from which the transformation may
    # be inferred; the next decision must remember this observation.
    return (
        f"Known cyclic-rule example: if only {CELLS[0]} is active, "
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


def gen_episode(rng, horizon, rule_switch_prob):
    direction = rng.choice((-1, 1))
    demos = [demo_text(direction)]
    active = [0] * len(CELLS)
    for i in rng.sample(range(len(CELLS)), ACTIVE_COUNT):
        active[i] = 1
    steps = []
    for t in range(horizon):
        # Rule in force was known at the start of this decision. No future
        # transition may alter gold before the agent has observed feedback.
        rule_at_decision = direction
        gold = rotate(active, rule_at_decision)
        executed = gold  # oracle trajectory for supervised OFFLINE training
        feedback = (
            f"After this decision the active components are {active_text(executed)}."
        )
        # Change the hidden rule only AFTER executing the current action.
        # Its new demonstration is observed now and applies from t+1 onward.
        if t < horizon - 1 and rng.random() < rule_switch_prob:
            direction = -direction
            feedback += " The rule changed. " + demo_text(direction)
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
        "environment_schema": "cyclic5_fixed2_v3",
        "episode_id": f"ep-{rng.getrandbits(32):08x}",
        "demonstrations": demos,
        "steps": steps,
    }


def validate_episode(episode):
    if episode.get("environment_schema") != "cyclic5_fixed2_v3":
        raise ValueError("wrong generator version or legacy symmetric dataset")
    if not isinstance(episode.get("episode_id"), str) or not episode["episode_id"]:
        raise ValueError("episode_id must be nonempty")
    demos = episode.get("demonstrations", [])
    if not isinstance(demos, list) or any(not isinstance(x, str) for x in demos):
        raise ValueError("demonstrations must be strings")
    sequence = episode.get("steps")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("steps must be a nonempty chronological list")
    k = len(CELLS)
    for t, step in enumerate(sequence):
        if step.get("oracle_rule_before_decision") not in (-1, 1):
            raise ValueError("missing private pre-decision rule audit field")
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
    # Every generated state must distinguish left from right. A symmetric
    # state would make a no-memory policy look artificially successful.
    if len(CELLS) % 2 == 0:
        raise ValueError("cyclic-rule benchmark requires odd ring width")
    # Audit exact oracle trajectory and change notification chronology,
    # not merely that labels are valid 2-hot arrays.
    if len(demos) != 1:
        raise ValueError("this synthetic dataset requires one initial demonstration")
    known_rule = sequence[0]["oracle_rule_before_decision"]
    if demos[0] != demo_text(known_rule):
        raise ValueError("initial demonstrated rule does not match oracle")
    candidates = []
    for pair in itertools.combinations(range(k), ACTIVE_COUNT):
        bits = [int(i in pair) for i in range(k)]
        if state_text(bits) == sequence[0]["decision"]["state"]:
            candidates.append(bits)
    if len(candidates) != 1:
        raise ValueError("invalid initial two-active state")
    active = candidates[0]
    for t, step in enumerate(sequence):
        state = step["decision"]
        if (state.get("state") != state_text(active)
            or state.get("selection_mode") != "multi"
            or state.get("allow_empty") is not False
            or state.get("options") != option_block()
            or step.get("action_ids") != option_ids()):
            raise ValueError("decision state/action protocol diverges from the simulator")
        direction = step["oracle_rule_before_decision"]
        if rotate(active, 1) == rotate(active, -1):
            raise ValueError("ambiguous state: both hidden rules imply same answer")
        expected_gold = rotate(active, direction)
        if step["gold"] != expected_gold or step["executed_set"] != expected_gold:
            raise ValueError("oracle action/label disagrees with visible state and rule")
        feedback = ("After this decision the active components are "
                    + active_text(step["executed_set"]) + ".")
        if t < len(sequence) - 1:
            next_rule = sequence[t + 1]["oracle_rule_before_decision"]
            if next_rule != direction:
                feedback += " The rule changed. " + demo_text(next_rule)
        if step.get("post_observation") != feedback:
            raise ValueError("post-action feedback leaks or omits a rule change")
        active = step["executed_set"]
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--rule-switch-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("episodes and horizon must be positive")
    if not 0.0 <= args.rule_switch_prob <= 1.0:
        parser.error("rule-switch-prob must be in [0,1]")
    rng = random.Random(args.seed)
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    with args.output.open("w", encoding="utf-8") as stream:
        for episode_index in range(args.episodes):
            episode = gen_episode(rng, args.horizon, args.rule_switch_prob)
            # Deterministic distinct IDs across train/dev generation seeds.
            episode["episode_id"] = f"ep-{args.seed}-{episode_index:08d}"
            validate_episode(episode)
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
