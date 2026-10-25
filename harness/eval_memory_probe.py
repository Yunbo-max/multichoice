#!/usr/bin/env python3
"""P1: linear probe on memory + same-state counterfactual (frozen v4 checkpoint).

Answers: does the persistent memory encode the shift k? (probe accuracy vs 1/6),
and does the decision respond to the history? (same state, six different demos).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

try:
    from eval_episodic_text_online import (
        _generator,
        _apply_rule,
        _valid_rule,
        initial_state,
    )
    from structured_readout import shift
    from memory_probe import fit_probe, counterfactual_summary
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from recurrent_set_decision import set_map
    from train_episodic_features import CKPT_SCHEMA
    from set_distillation import allowed_counts
except ModuleNotFoundError:
    from harness.eval_episodic_text_online import (
        _generator,
        _apply_rule,
        _valid_rule,
        initial_state,
    )
    from harness.structured_readout import shift
    from harness.memory_probe import fit_probe, counterfactual_summary
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map
    from harness.train_episodic_features import CKPT_SCHEMA
    from harness.set_distillation import allowed_counts


@torch.no_grad()
def collect_probe(gen, encoder, agent, episodes):
    memories, labels = [], []
    for episode in episodes:
        active = initial_state(gen, episode)
        memory = agent.empty_memory(1, device=encoder.device)
        demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
        if demos.shape[1]:
            memory = agent.prime(memory, demos)
        for oracle in episode["steps"]:
            rule = oracle["oracle_rule_before_decision"]
            if not _valid_rule(gen, rule):
                raise ValueError("invalid rule timeline")
            memories.append(memory.detach().cpu())
            labels.append(int(rule))
            row = dict(oracle["decision"])
            row["state"] = gen.state_text(active)
            context, logits, mask = encoder.encode_question(row)
            final = agent.decide(memory, context, logits, mask)[-1]
            k = len(gen.CELLS)
            support = allowed_counts(row)
            gold = _apply_rule(gen, active, rule)
            idx = set_map(
                final["scores"][0, :k], final["correction"][0, : k + 1], support
            )["indices"]
            executed = [int(i in idx) for i in range(k)]
            feedback = (
                "After this decision the active components are "
                + gen.active_text(executed)
                + "."
            )
            obs = encoder.encode_observation(feedback)
            option_features = encoder.encode_option_meanings(row)
            action = torch.tensor(executed, dtype=torch.float32, device=encoder.device)[
                None
            ]
            memory = agent.observe(
                memory,
                context,
                action,
                mask,
                obs,
                option_features=option_features,
                done=torch.tensor([False], dtype=torch.bool, device=encoder.device),
            )
            active = executed
    return memories, labels


@torch.no_grad()
def counterfactual(gen, encoder, agent, state):
    n = len(gen.CELLS)
    predicted = {}
    for k in range(1, n):
        memory = agent.empty_memory(1, device=encoder.device)
        demos = encoder.encode_demonstrations([gen.demo_text(k)])
        memory = agent.prime(memory, demos)
        row = {
            "state": gen.state_text(state),
            "question": "Which components should be active now?",
            "options": gen.option_block(),
            "selection_mode": "multi",
            "allow_empty": False,
            "allowed_cardinalities": [2],
        }
        context, logits, mask = encoder.encode_question(row)
        final = agent.decide(memory, context, logits, mask)[-1]
        idx = set_map(final["scores"][0, :n], final["correction"][0, : n + 1], [2])[
            "indices"
        ]
        predicted[k] = [int(i in idx) for i in range(n)]
    return predicted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--model-identity", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("report already exists")
    episodes = [json.loads(line) for line in args.episodes.open() if line.strip()]
    if not episodes:
        raise ValueError("empty eval set")
    episodes = episodes[: args.limit] if args.limit else episodes
    schema = episodes[0].get("environment_schema")
    gen = _generator(schema)
    for episode in episodes:
        gen.validate_episode(episode)
    saved = torch.load(args.head, map_location="cpu", weights_only=True)
    if (
        saved.get("schema") != CKPT_SCHEMA
        or saved.get("model_identity") != args.model_identity
    ):
        raise ValueError("checkpoint identity mismatch")
    encoder = FrozenQwenDecisionEncoder(
        args.qwen_model, device=args.device, dtype=args.dtype
    )
    config = EpisodicSetConfig(**saved["config"])
    agent = EpisodicSetDecision(config).to(encoder.device).eval()
    agent.load_state_dict(saved["model"], strict=True)

    memories, labels = collect_probe(gen, encoder, agent, episodes)
    probe, train_acc = fit_probe(memories, labels, n_classes=len(gen.CELLS) - 1)

    state = [0] * len(gen.CELLS)
    state[0] = state[1] = 1
    predicted = counterfactual(gen, encoder, agent, state)
    # each k should select shift_k(state); count how many are correct
    cf_correct = {k: (predicted[k] == shift(state, k)) for k in predicted}

    report = {
        "head": str(args.head),
        "dataset": str(args.episodes),
        "schema": schema,
        "probe": {
            "n_samples": len(memories),
            "train_accuracy": train_acc,
            "chance": 1.0 / (len(gen.CELLS) - 1),
            "stored_rule": train_acc > 1.0 / (len(gen.CELLS) - 1) + 0.05,
        },
        "counterfactual": {
            "state": gen.state_text(state),
            "summary": counterfactual_summary(predicted),
            "per_shift_correct": {str(k): bool(v) for k, v in cf_correct.items()},
            "n_correct": sum(cf_correct.values()),
            "n_conditions": len(cf_correct),
        },
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
