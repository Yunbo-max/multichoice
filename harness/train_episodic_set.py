#!/usr/bin/env python3
"""Causal multi-select environment benchmark with persistent episodic memory.

Rule-shift episodes: before step zero the agent sees one demonstration of an
unknown cyclic shift. Subsequent prompts contain only the current state.
The agent must keep the task rule while selecting a set of next active cells.
Environment state t+1 = executed set t (not the gold oracle set). During
training a mixture of oracle and policy actions is executed to reduce
distribution shift; the oracle never enters decide() or observe().
No natural-language CoT and no API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from recurrent_set_decision import gold_set_nll, set_map

SCHEMA = "episodic-set-shift-v1"
DIRECTIONS = (-2, -1, 1, 2)


def context_dim(width: int) -> int:
    return 2 * width + 2


def observation_features(state: torch.Tensor, t: int, horizon: int) -> torch.Tensor:
    """Visible prompt / post-action observation. No hidden rule."""
    b, width = state.shape
    pad = state.new_zeros(b, width)
    time = state.new_full((b, 1), t / max(1, horizon))
    return torch.cat((state.float(), pad, time, torch.ones_like(time)), dim=-1)


def demo_features(initial: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
    """Known input/output demonstration supplied only at episode start."""
    b, width = initial.shape
    extra = initial.new_zeros(b, 2)
    return torch.cat((initial.float(), transformed.float(), extra), dim=-1)


def native_option_logits(state: torch.Tensor) -> torch.Tensor:
    """Myopic baseline: carry current active cells."""
    return 1.5 * (2 * state.float() - 1)


def shift_sets(states: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Oracle evaluates targets on the currently executed state."""
    if states.ndim != 2 or shifts.shape != (states.shape[0],):
        raise ValueError("states must be B x K and shifts B")
    return torch.stack([
        torch.roll(states[i], shifts=int(shifts[i]), dims=0)
        for i in range(states.shape[0])
    ])


def sample_batch(batch: int, width: int, rng: random.Random, device: torch.device):
    if width < 5 or batch < 1:
        raise ValueError("width must be >= 5 and batch positive")
    shifts = torch.tensor([rng.choice(DIRECTIONS) for _ in range(batch)],
                          dtype=torch.long, device=device)
    chosen = [rng.sample(range(width), 2) for _ in range(batch)]
    state = torch.zeros(batch, width, dtype=torch.float32, device=device)
    for i, inds in enumerate(chosen):
        state[i, inds] = 1
    demo_start = torch.zeros(batch, width, dtype=torch.float32, device=device)
    for i in range(batch):
        demo_start[i, rng.randrange(width)] = 1
    demo_end = shift_sets(demo_start, shifts)
    demos = demo_features(demo_start, demo_end).unsqueeze(1)
    return state, shifts, demos


@torch.no_grad()
def predict_sets(scores: torch.Tensor, correction: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    batch, _ = scores.shape
    chosen = torch.zeros_like(scores)
    for i in range(batch):
        k = int(mask[i].sum().item())
        selected = set_map(
            scores[i, :k], correction[i, : k + 1], list(range(1, k + 1))
        )["indices"]
        chosen[i, selected] = 1
    return chosen


def one_episode_batch(
    agent: EpisodicSetDecision,
    *,
    batch: int,
    width: int,
    horizon: int,
    rng: random.Random,
    device: torch.device,
    teacher_force_prob: float,
    train: bool,
    carry_memory: bool = True,
    rule_switch_prob: float = 0.0,
) -> dict:
    if not 0.0 <= teacher_force_prob <= 1.0 or not 0 <= rule_switch_prob <= 1:
        raise ValueError("forcing and rule-switch probabilities must be within [0, 1]")
    current, rule, demos = sample_batch(batch, width, rng, device)
    mask = torch.ones_like(current, dtype=torch.bool)
    # A new rule may be observed only AFTER a selected action is executed.
    switch_times = [
        rng.randint(1, horizon - 1) if rng.random() < rule_switch_prob else -1
        for _ in range(batch)
    ]
    new_rules = [rng.choice([d for d in DIRECTIONS if d != int(rule[i])])
                 for i in range(batch)]
    memory = agent.prime(agent.empty_memory(batch, device=device), demos)
    episode_all = torch.ones(batch, dtype=torch.bool, device=device)
    first_correct = later_correct = later_total = total_exact = 0.0
    per_step_losses = []
    for t in range(horizon):
        # The same weights and compute; only persistence of prior memory differs.
        if t > 0 and not carry_memory:
            memory = agent.empty_memory(batch, device=device)
        query = observation_features(current, t, horizon)
        native = native_option_logits(current)
        history = agent.decide(memory, query, native, mask)
        final = history[-1]
        gold = shift_sets(current, rule)
        losses = [
            gold_set_nll(
                final["scores"][i, :width], final["correction"][i, : width + 1],
                gold[i], list(range(1, width + 1)),
            )
            for i in range(batch)
        ]
        per_step_losses.append(torch.stack(losses).mean())
        selected = predict_sets(final["scores"], final["correction"], mask)
        correct = torch.all(selected == gold, dim=1)
        exact = int(correct.sum().item())
        total_exact += exact
        if t == 0:
            first_correct = exact
        else:
            later_correct += exact
            later_total += batch
        episode_all &= correct
        # Teacher forcing is a *training-only* state-distribution intervention.
        # No oracle labels are passed to the policy's memory update.
        if train and teacher_force_prob > 0:
            use_expert = torch.tensor(
                [rng.random() < teacher_force_prob for _ in range(batch)],
                device=device, dtype=torch.bool,
            )
            executed = torch.where(use_expert[:, None], gold, selected)
        else:
            executed = selected
        executed = executed.detach()
        # The only available post-action observation is the new visible state.
        after = observation_features(executed, t + 1, horizon)
        to_switch = [i for i in range(batch) if switch_times[i] == t + 1]
        if to_switch:
            demo_start = torch.zeros(
                len(to_switch), width, dtype=executed.dtype, device=device,
            )
            for j in range(len(to_switch)):
                demo_start[j, rng.randrange(width)] = 1
            next_rule = torch.tensor(
                [new_rules[i] for i in to_switch], device=device, dtype=torch.long,
            )
            demo_end = shift_sets(demo_start, next_rule)
            # External feedback reveals a fresh input/output demonstration
            # for the NEXT decision; no future oracle target is used.
            after[to_switch] = demo_features(demo_start, demo_end)
            rule[to_switch] = next_rule
        memory = agent.observe(
            memory, query, executed, mask, after,
            done=torch.full(
                (batch,), t == horizon - 1, device=device, dtype=torch.bool,
            ),
        )
        current = executed
    mean_set_nll = torch.stack(per_step_losses).mean()
    return {
        "loss": mean_set_nll,
        "set_nll_per_option": float(mean_set_nll.detach()),
        "first_em": first_correct / batch,
        "later_em": later_correct / later_total if later_total else float("nan"),
        "step_em": total_exact / (batch * horizon),
        "trajectory_em": episode_all.float().mean().item(),
    }


def training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    config = EpisodicSetConfig(
        context_dim=context_dim(args.width), latent_dim=args.latent_dim,
        max_options=args.width, inner_steps=args.inner_steps,
    )
    agent = EpisodicSetDecision(config).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=0)
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = []
    best_metric = -1.0
    for epoch in range(args.epochs):
        agent.train()
        tf_prob = args.teacher_force_end + (
            args.teacher_force_start - args.teacher_force_end
        ) * max(0.0, (args.epochs - 1 - epoch) / max(1, args.epochs - 1))
        epoch_loss = 0.0
        for _ in range(args.train_batches):
            optimizer.zero_grad(set_to_none=True)
            result = one_episode_batch(
                agent, batch=args.batch_size, width=args.width, horizon=args.horizon,
                rng=rng, device=device, teacher_force_prob=tf_prob,
                train=True, carry_memory=True,
                rule_switch_prob=args.rule_switch_prob,
            )
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("nonfinite episodic set loss")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_loss += float(result["loss"].detach())
        agent.eval()
        with torch.no_grad():
            eval_results = {}
            for carry in (True, False):
                eval_rng = random.Random(args.seed + 100_000)
                reports = [
                    one_episode_batch(
                        agent, batch=args.batch_size, width=args.width,
                        horizon=args.horizon, rng=eval_rng, device=device,
                        teacher_force_prob=0.0, train=False, carry_memory=carry,
                        rule_switch_prob=args.rule_switch_prob,
                    )
                    for _ in range(args.eval_batches)
                ]
                eval_results["carry" if carry else "reset_every_step"] = {
                    key: sum(float(x[key]) for x in reports) / len(reports)
                    for key in ("first_em", "later_em", "step_em", "trajectory_em", "set_nll_per_option")
                }
        record = {
            "epoch": epoch + 1, "train_set_nll": epoch_loss / args.train_batches,
            "teacher_force_prob": tf_prob, "eval": eval_results,
        }
        print(json.dumps(record), flush=True)
        metrics.append(record)
        # This is selected dev performance, not an untouched test.
        if eval_results["carry"]["trajectory_em"] >= best_metric:
            best_metric = eval_results["carry"]["trajectory_em"]
            torch.save({
                "schema": SCHEMA, "config": asdict(config),
                "model": {
                    key: val.detach().cpu()
                    for key, val in agent.state_dict().items()
                },
                "seed": args.seed, "epoch": epoch + 1,
                "width": args.width, "horizon": args.horizon,
                "eval": eval_results,
            }, args.output / "best.pt")
    (args.output / "training.json").write_text(json.dumps({
        "schema": SCHEMA, "arguments": vars(args) | {"output": str(args.output)},
        "history": metrics,
        "warning": "synthetic online-set task, NOT ARC-AGI or BDH-CQ replication",
    }, indent=2) + "\n")


def evaluating(args: argparse.Namespace) -> None:
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA:
        raise ValueError("unsupported episodic checkpoint")
    config = EpisodicSetConfig(**saved["config"])
    agent = EpisodicSetDecision(config).to(args.device)
    agent.load_state_dict(saved["model"], strict=True)
    agent.eval()
    results = {}
    with torch.no_grad():
        for carry in (True, False):
            rng = random.Random(args.seed)
            trials = [
                one_episode_batch(
                    agent, batch=args.batch_size, width=config.max_options,
                    horizon=args.horizon, rng=rng,
                    device=torch.device(args.device), teacher_force_prob=0,
                    train=False, carry_memory=carry,
                    rule_switch_prob=args.rule_switch_prob,
                )
                for _ in range(args.eval_batches)
            ]
            results["carry" if carry else "reset_every_step"] = {
                key: sum(float(x[key]) for x in trials) / len(trials)
                for key in ("first_em", "later_em", "step_em", "trajectory_em", "set_nll_per_option")
            }
    print(json.dumps({
        "schema": SCHEMA, "checkpoint": str(args.checkpoint),
        "seed": args.seed, "horizon": args.horizon, "results": results,
    }, indent=2))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--width", type=int, default=6)
    train.add_argument("--horizon", type=int, default=4)
    train.add_argument("--latent-dim", type=int, default=48)
    train.add_argument("--inner-steps", type=int, default=4)
    train.add_argument("--train-batches", type=int, default=20)
    train.add_argument("--eval-batches", type=int, default=4)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--teacher-force-start", type=float, default=1.0)
    train.add_argument("--teacher-force-end", type=float, default=0.25)
    train.add_argument("--rule-switch-prob", type=float, default=0.25)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--horizon", type=int, default=6)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--eval-batches", type=int, default=32)
    evaluate.add_argument("--rule-switch-prob", type=float, default=0.25)
    for item in (train, evaluate):
        item.add_argument("--device", default="cpu")
        item.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if getattr(args, "width", 6) < 5 or args.horizon < 2 or args.batch_size < 1:
        parser.error("width>=5, horizon>=2, batch-size>=1 required")
    if getattr(args, "epochs", 1) < 1 or getattr(args, "train_batches", 1) < 1:
        parser.error("positive epochs/batches required")
    if getattr(args, "eval_batches", 1) < 1:
        parser.error("eval-batches must be positive")
    if not 0 <= args.rule_switch_prob <= 1:
        parser.error("rule-switch-prob must be within [0,1]")
    if hasattr(args, "teacher_force_start") and not (
        0 <= args.teacher_force_end <= args.teacher_force_start <= 1
    ):
        parser.error("teacher forcing probabilities must descend within [0,1]")
    return args


if __name__ == "__main__":
    args = parse()
    if args.command == "train":
        training(args)
    else:
        evaluating(args)
