#!/usr/bin/env python3
"""P0: structured set readout on the frozen v4 checkpoints (no retraining).

Runs the carry policy, collects per-step option scores, and compares two readouts
of the SAME scores:
  raw        set-MAP over all C(7,2)=21 two-subsets (the current decoder)
  structured softmax over the 6 reachable successors E_k = sum_{i in shift_k(S)} s_i

Also reports the theory baselines: random-legal-shift EM = 1/6, uniform-6 NLL =
log 6, uniform-21 NLL = log 21. This isolates whether the low on-policy accuracy
is a readout problem (mass wasted on the 15 impossible sets) or a memory problem.
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
    from structured_readout import baselines, structured_nll, structured_predict
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from recurrent_set_decision import set_map, gold_set_nll
    from train_episodic_features import CKPT_SCHEMA
    from set_distillation import allowed_counts
except ModuleNotFoundError:
    from harness.eval_episodic_text_online import (
        _generator,
        _apply_rule,
        _valid_rule,
        initial_state,
    )
    from harness.structured_readout import baselines, structured_nll, structured_predict
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map, gold_set_nll
    from harness.train_episodic_features import CKPT_SCHEMA
    from harness.set_distillation import allowed_counts


@torch.no_grad()
def run_episode(gen, encoder, agent, episode: dict, tau: float) -> dict:
    states = episode["steps"]
    active = initial_state(gen, episode)
    memory = agent.empty_memory(1, device=encoder.device)
    demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
    if demos.shape[1]:
        memory = agent.prime(memory, demos)

    raw_first = raw_later = 0
    struct_first = struct_later = 0
    raw_traj = struct_traj = True
    raw_nll = struct_nll_sum = 0.0
    k = len(gen.CELLS)

    for t, oracle in enumerate(states):
        rule = oracle["oracle_rule_before_decision"]
        if not _valid_rule(gen, rule):
            raise ValueError("invalid rule timeline")
        row = dict(oracle["decision"])
        row["state"] = gen.state_text(active)
        context, logits, mask = encoder.encode_question(row)
        final = agent.decide(memory, context, logits, mask)[-1]
        scores = final["scores"][0, :k].float().cpu().tolist()
        support = allowed_counts(row)

        gold = _apply_rule(gen, active, rule)

        # raw readout: set-MAP over 21 subsets
        raw_idx = set_map(
            final["scores"][0, :k], final["correction"][0, : k + 1], support
        )["indices"]
        raw_pred = [int(i in raw_idx) for i in range(k)]
        raw_correct = raw_pred == gold

        # structured readout: softmax over 6 successors
        struct = structured_predict(active, scores, tau=tau)
        struct_correct = struct["set"] == gold

        if t == 0:
            raw_first += int(raw_correct)
            struct_first += int(struct_correct)
        else:
            raw_later += int(raw_correct)
            struct_later += int(struct_correct)
        raw_traj = raw_traj and raw_correct
        struct_traj = struct_traj and struct_correct

        raw_nll += float(
            gold_set_nll(
                final["scores"][0, :k],
                final["correction"][0, : k + 1],
                torch.tensor(gold, dtype=torch.float32, device=encoder.device),
                support,
                normalize=False,
            )
        )
        struct_nll_sum += structured_nll(active, scores, rule, tau=tau)

        # advance the environment with the model's ACTUAL (raw) selected action
        executed = raw_pred
        feedback = (
            "After this decision the active components are "
            + gen.active_text(executed)
            + "."
        )
        if t + 1 < len(states):
            following = states[t + 1]["oracle_rule_before_decision"]
            if following != rule:
                feedback += " The rule changed. " + gen.demo_text(following)
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
            done=torch.tensor(
                [t == len(states) - 1], dtype=torch.bool, device=encoder.device
            ),
        )
        active = executed

    return {
        "n_steps": len(states),
        "raw_first": raw_first,
        "raw_later": raw_later,
        "raw_traj": int(raw_traj),
        "raw_nll": raw_nll,
        "struct_first": struct_first,
        "struct_later": struct_later,
        "struct_traj": int(struct_traj),
        "struct_nll": struct_nll_sum,
    }


def evaluate(gen, encoder, agent, episodes, tau):
    r = {
        k: 0
        for k in (
            "raw_first",
            "raw_later",
            "raw_traj",
            "struct_first",
            "struct_later",
            "struct_traj",
        )
    }
    raw_nll = struct_nll = total = 0.0
    for episode in episodes:
        out = run_episode(gen, encoder, agent, episode, tau)
        for key in r:
            r[key] += out[key]
        raw_nll += out["raw_nll"]
        struct_nll += out["struct_nll"]
        total += out["n_steps"]
    n = len(episodes)
    later_count = total - n
    return {
        "n_episodes": n,
        "n_decisions": int(total),
        "raw": {
            "step_em": (r["raw_first"] + r["raw_later"]) / total,
            "later_em": r["raw_later"] / later_count if later_count else None,
            "trajectory_em": r["raw_traj"] / n,
            "set_nll": raw_nll / total,
        },
        "structured": {
            "step_em": (r["struct_first"] + r["struct_later"]) / total,
            "later_em": r["struct_later"] / later_count if later_count else None,
            "trajectory_em": r["struct_traj"] / n,
            "shift_nll": struct_nll / total,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--model-identity", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.tau <= 0:
        p.error("--tau must be positive")
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
    report = {
        "head": str(args.head),
        "dataset": str(args.episodes),
        "schema": schema,
        "tau": args.tau,
        "baselines": baselines(len(gen.CELLS)),
        "comparison": evaluate(gen, encoder, agent, episodes, args.tau),
        "note": "raw set-MAP (21 subsets) vs structured softmax over 6 reachable successors",
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
