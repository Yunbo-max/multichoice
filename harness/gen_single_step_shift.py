#!/usr/bin/env python3
"""Generate single-step shift samples for Stage 1 (single-step decision capability).

Enumerates all 126 elementary state-rule combinations (C(7,2)=21 active 2-sets x
6 cyclic shifts). Each sample is ONE decision: given the rule demonstration and
the current active set, output the shifted 2-set. No memory, no sequence — this
isolates whether the base decision representation / input encoding / optimizer can
learn the elementary transition at all.

Rows are in the static multi-select JSONL format consumed by train_multiselect.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

CELLS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"]
N = len(CELLS)


def shift_set(active, k):
    return [active[(i - k) % N] for i in range(N)]


def names(active):
    return [CELLS[i] for i, bit in enumerate(active) if bit]


def active_text(active):
    return ", ".join(names(active))


def demo_text(k):
    inp = [int(i == 0) for i in range(N)]
    out = shift_set(inp, k)
    return (
        f"Known shift-rule example: if only {CELLS[0]} is active, "
        f"the next active component is {active_text(out)}."
    )


def options_block():
    return [
        {"key": chr(ord("a") + i), "desc": f"Activate {CELLS[i]}"} for i in range(N)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-output", type=Path, required=True)
    p.add_argument("--test-output", type=Path, required=True)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    states = []
    for pair in itertools.combinations(range(N), 2):
        states.append([int(i in pair) for i in range(N)])
    combos = [(s, k) for s in states for k in range(1, N)]
    assert len(combos) == 126, len(combos)

    rng = random.Random(args.seed)
    rng.shuffle(combos)
    n_test = int(len(combos) * args.test_fraction)
    test_combos = combos[:n_test]
    train_combos = combos[n_test:]

    for path, subset in (
        (args.train_output, train_combos),
        (args.test_output, test_combos),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        with path.open("w", encoding="utf-8") as f:
            for state, k in subset:
                target = shift_set(state, k)
                gold_letters = "".join(
                    chr(ord("a") + i) for i, bit in enumerate(target) if bit
                )
                row = {
                    "id": f"shift-step-{active_text(state).replace(', ', '-')}-k{k}",
                    "source": "shift_single_step",
                    "lang": "en",
                    "split": "train" if path == args.train_output else "dev",
                    "state": f"The active components are: {active_text(state)}.",
                    "question": (
                        demo_text(k)
                        + " The active components are: "
                        + active_text(state)
                        + ". Which components should be active now?"
                    ),
                    "options": options_block(),
                    "target": target,
                    "n_options": N,
                    "gold_letters": gold_letters,
                    "selection_mode": "multi",
                    "allow_empty": False,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(subset)} rows to {path}")


if __name__ == "__main__":
    main()
