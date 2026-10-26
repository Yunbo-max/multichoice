#!/usr/bin/env python3
"""Collect on-policy (DAgger) training cache from the model's own rollouts.

Loads a frozen checkpoint and runs the carry rollout on the training episodes,
recording each step's features with the MODEL's actual executed action (not the
oracle logged action) and the feedback rebuilt from that action. The output cache
matches ``build_episodic_features.py`` so it can be merged with the expert cache
and fed straight into ``train_episodic_features.py``.
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
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from recurrent_set_decision import set_map
    from train_episodic_features import SCHEMA, CKPT_SCHEMA
    from set_distillation import allowed_counts
except ModuleNotFoundError:
    from harness.eval_episodic_text_online import (
        _generator,
        _apply_rule,
        _valid_rule,
        initial_state,
    )
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map
    from harness.train_episodic_features import SCHEMA, CKPT_SCHEMA
    from harness.set_distillation import allowed_counts


@torch.no_grad()
def collect(gen, encoder, agent, episodes):
    cache_episodes = []
    for episode in episodes:
        active = initial_state(gen, episode)
        memory = agent.empty_memory(1, device=encoder.device)
        demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
        demo_features = (
            demos[0].detach().cpu() if demos.shape[1] else demos[0].detach().cpu()
        )
        if demos.shape[1]:
            memory = agent.prime(memory, demos)
        steps = []
        k = len(gen.CELLS)
        for t, oracle in enumerate(episode["steps"]):
            rule = oracle["oracle_rule_before_decision"]
            if not _valid_rule(gen, rule):
                raise ValueError("invalid rule timeline")
            row = dict(oracle["decision"])
            row["state"] = gen.state_text(active)
            context, logits, mask = encoder.encode_question(row)
            final = agent.decide(memory, context, logits, mask)[-1]
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
            if t + 1 < len(episode["steps"]):
                following = episode["steps"][t + 1]["oracle_rule_before_decision"]
                if following != rule:
                    feedback += " The rule changed. " + gen.demo_text(following)
            obs = encoder.encode_observation(feedback)[0].detach().cpu()
            options = encoder.encode_option_meanings(row)[0].detach().cpu()
            steps.append(
                {
                    "step_index": t,
                    "context": context[0].detach().cpu(),
                    "native_logits": logits[0].detach().cpu(),
                    "option_features": options,
                    "option_ids": gen.option_ids(),
                    "gold": torch.tensor(gold, dtype=torch.float32),
                    "executed_set": torch.tensor(executed, dtype=torch.float32),
                    "post_observation": obs,
                    "allowed": support,
                    "done": t == len(episode["steps"]) - 1,
                    "rule": rule,
                }
            )
            action = torch.tensor(executed, dtype=torch.float32, device=encoder.device)[
                None
            ]
            memory = agent.observe(
                memory,
                context,
                action,
                mask,
                encoder.encode_observation(feedback),
                option_features=encoder.encode_option_meanings(row),
                done=torch.tensor(
                    [t == len(episode["steps"]) - 1],
                    dtype=torch.bool,
                    device=encoder.device,
                ),
            )
            active = executed
        cache_episodes.append(
            {
                "episode_id": episode["episode_id"] + "-policy",
                "demonstrations": demo_features,
                "steps": steps,
            }
        )
    return cache_episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--model-identity", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("policy cache already exists")
    episodes = [json.loads(line) for line in args.episodes.open() if line.strip()]
    if not episodes:
        raise ValueError("empty episode set")
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
    cache_episodes = collect(gen, encoder, agent, episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "model_identity": args.model_identity,
            "episodes": cache_episodes,
        },
        args.output,
    )
    print(f"wrote {len(cache_episodes)} policy episodes to {args.output}")


if __name__ == "__main__":
    main()
