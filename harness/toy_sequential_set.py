#!/usr/bin/env python3
"""Deterministic, CPU-friendly sequential-choice sandbox (not ARC).

A task defines a hidden left/right cyclic rule via a demonstration (x -> y).
At each time step the agent selects *all* cells activated by applying that rule;
its selected set becomes the following environment state. The agent emits
sets only, never explanatory tokens. Train on oracle trajectories, evaluate
policy-dependent online rollouts, and compare depths and feedback modes.

This is a unit-scale *mechanism probe*, not evidence of ARC generalization.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

import torch

try:
    from recurrent_set_decision import (
        RecurrentSetConfig, RecurrentSetDecision, loss_over_steps, set_map,
    )
except ModuleNotFoundError:
    from harness.recurrent_set_decision import (
        RecurrentSetConfig, RecurrentSetDecision, loss_over_steps, set_map,
    )


WIDTH = 4
CONTEXT_DIM = 12


def rotate(bits, direction):
    return tuple(bits[(i - direction) % WIDTH] for i in range(WIDTH))


def embed_query(bits, time_step):
    step = [float(time_step == j) for j in range(4)]
    return torch.tensor(list(map(float, bits)) + step + [0.] * 4)


def embed_demo(bits, target):
    return torch.tensor(list(map(float, bits)) + list(map(float, target)) + [0.] * 4)


def native_logits(bits):
    return torch.tensor([1.5 if bit else -1.5 for bit in bits])


@dataclass
class Episode:
    direction: int
    first_state: tuple
    demo: torch.Tensor

    @staticmethod
    def sample(rng):
        direction = rng.choice((-1, 1))
        # One-hot demonstration disambiguates direction unambiguously.
        one = rng.randrange(WIDTH)
        demo_in = tuple(int(i == one) for i in range(WIDTH))
        target = rotate(demo_in, direction)
        first = tuple(int(i in rng.sample(range(WIDTH), 2)) for i in range(WIDTH))
        return Episode(direction, first, embed_demo(demo_in, target))


def make_episodes(n, seed):
    rng = random.Random(seed)
    return [Episode.sample(rng) for _ in range(n)]


def oracle_records(episodes, horizon):
    examples = []
    for ep in episodes:
        bits = ep.first_state
        for t in range(horizon):
            gold = rotate(bits, ep.direction)
            examples.append((embed_query(bits, t), native_logits(bits),
                             torch.tensor(gold, dtype=torch.float32), ep.demo))
            bits = gold
    return examples


def train(examples, epochs, seed, feedback, steps, lr):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = RecurrentSetDecision(RecurrentSetConfig(
        context_dim=CONTEXT_DIM, latent_dim=48, max_options=WIDTH, feedback=feedback
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        total = 0.0
        for start in range(0, len(order), 64):
            batch = [examples[i] for i in order[start:start + 64]]
            context, native, gold, demo = [
                torch.stack([record[j] for record in batch]) for j in range(4)
            ]
            mask = torch.ones_like(native, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(context, native, mask, steps=steps,
                            demonstrations=demo[:, None])
            loss = loss_over_steps(outputs, gold, mask,
                                   [list(range(1, WIDTH + 1))] * len(batch),
                                   step_supervision="all")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        if epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(json.dumps({"epoch": epoch + 1, "train_set_nll_per_option":
                              total / len(examples)}))
    return model


@torch.no_grad()
def rollout(model, episodes, horizon, steps, use_demo):
    model.eval()
    exact = trajectories = total = 0
    for ep in episodes:
        bits = ep.first_state
        all_correct = True
        for t in range(horizon):
            context = embed_query(bits, t)[None]
            native = native_logits(bits)[None]
            mask = torch.ones_like(native, dtype=torch.bool)
            demo = ep.demo[None, None] if use_demo else None
            output = model(context, native, mask, steps=steps, demonstrations=demo)[-1]
            decoded = set_map(output["scores"][0], output["correction"][0, :WIDTH + 1],
                              list(range(1, WIDTH + 1)))
            selected = tuple(int(i in decoded["indices"]) for i in range(WIDTH))
            gold = rotate(bits, ep.direction)
            correct = selected == gold
            exact += int(correct)
            total += 1
            all_correct &= correct
            # Crucial: later inputs depend on the model's *actual* previous action.
            bits = selected
        trajectories += int(all_correct)
    return {"step_em": exact / total, "trajectory_exact": trajectories / len(episodes),
            "n_episodes": len(episodes), "steps": steps, "use_demo": use_demo}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-episodes", type=int, default=512)
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--feedback", choices=("set", "latent_only", "static"), default="set")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()
    if args.horizon < 1 or args.horizon > 4 or args.steps < 1:
        p.error("horizon must be 1..4 and trained steps >= 1")
    training = make_episodes(args.train_episodes, args.seed)
    evaluation = make_episodes(args.eval_episodes, args.seed + 100_000)
    examples = oracle_records(training, args.horizon)
    model = train(examples, args.epochs, args.seed, args.feedback, args.steps, args.lr)
    report = {
        "config": vars(args), "evaluations": {
            "native_no_reason": rollout(model, evaluation, args.horizon, 0, True),
            "recurrent_with_demo": rollout(model, evaluation, args.horizon, args.steps, True),
            "recurrent_no_demo": rollout(model, evaluation, args.horizon, args.steps, False),
        },
        "warning": "synthetic toy, oracle-trajectory supervised, NOT ARC-AGI or BDH-CQ replication",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
