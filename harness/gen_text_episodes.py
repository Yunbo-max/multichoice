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
from pathlib import Path

CELLS = ["Alpha", "Beta", "Gamma", "Delta"]


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
    direction_name = "right" if direction > 0 else "left"
    return (
        f"Rule example: when only {CELLS[0]} is active, the active set becomes "
        f"{active_text(out)} (a cyclic {direction_name} shift)."
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
    for i in rng.sample(range(len(CELLS)), 2):
        active[i] = 1
    steps = []
    for t in range(horizon):
        extra = ""
        if t > 0 and rng.random() < rule_switch_prob:
            direction = -direction
            extra = " The rule changed. " + demo_text(direction)
        gold = rotate(active, direction)
        executed = gold  # oracle trajectory for supervised training
        steps.append(
            {
                "step_index": t,
                "decision": {
                    "state": state_text(active),
                    "question": "Which components should be active now?",
                    "options": option_block(),
                    "selection_mode": "multi",
                    "allow_empty": False,
                },
                "action_ids": option_ids(),
                "gold": gold,
                "executed_set": executed,
                "post_observation": (
                    f"After this decision the active components are {active_text(gold)}."
                    + extra
                ),
                "done": t == horizon - 1,
            }
        )
        active = gold
    return {
        "episode_id": f"ep-{rng.getrandbits(32):08x}",
        "demonstrations": demos,
        "steps": steps,
    }


def validate_episode(episode):
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
        if step.get("step_index") != t:
            raise ValueError("step_index must equal chronological position")
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
            if sum(value) not in range(1, k + 1):
                raise ValueError(f"{field} must be a nonempty selection")
        if step.get("done") is not (t == len(sequence) - 1):
            raise ValueError("done must be true only for the final step")
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
    with args.output.open("w", encoding="utf-8") as stream:
        for _ in range(args.episodes):
            episode = gen_episode(rng, args.horizon, args.rule_switch_prob)
            validate_episode(episode)
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
