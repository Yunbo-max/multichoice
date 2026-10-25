#!/usr/bin/env python3
"""On-policy generated cyclic-rule evaluation using frozen Qwen + learned head.

Unlike offline replay, model actions determine the next observed state.
Private oracle_rule_before_decision is used solely by the environment/scorer,
never included in the decision prompt or the memory update.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

try:
    from gen_text_episodes import CELLS, rotate, state_text, active_text, demo_text, validate_episode
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from recurrent_set_decision import set_map, gold_set_nll
    from train_episodic_features import CKPT_SCHEMA
    from set_distillation import allowed_counts
except ModuleNotFoundError:
    from harness.gen_text_episodes import CELLS, rotate, state_text, active_text, demo_text, validate_episode
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map, gold_set_nll
    from harness.train_episodic_features import CKPT_SCHEMA
    from harness.set_distillation import allowed_counts


def initial_state(episode: dict) -> list[int]:
    line = episode["steps"][0]["decision"]["state"]
    marker = "The active components are: "
    if not isinstance(line, str) or not line.startswith(marker) or not line.endswith("."):
        raise ValueError("not a generated cyclic-rule episode")
    payload = line[len(marker):-1]
    labels = [] if payload == "none" else payload.split(", ")
    if len(set(labels)) != len(labels) or any(x not in CELLS for x in labels):
        raise ValueError("invalid initial state")
    return [int(cell in labels) for cell in CELLS]


@torch.no_grad()
def run_episode(encoder, agent, episode: dict, *, carry: bool) -> dict:
    states = episode.get("steps")
    if not states:
        raise ValueError("empty episode")
    active = initial_state(episode)
    memory = agent.empty_memory(1, device=encoder.device)
    demonstrations = encoder.encode_demonstrations(episode.get("demonstrations", []))
    if demonstrations.shape[1]:
        memory = agent.prime(memory, demonstrations)
    first = later = 0
    trajectory_correct = True
    nll = 0.
    observed_feedback, executed_actions = [], []
    for t, oracle in enumerate(states):
        if oracle.get("step_index") != t:
            raise ValueError("nonchronological oracle episode")
        if t > 0 and not carry:
            memory = agent.empty_memory(1, device=encoder.device)
        rule = oracle.get("oracle_rule_before_decision")
        if rule not in (-1, 1):
            raise ValueError("requires causally audited synthetic rule timeline")
        row = dict(oracle["decision"])
        row["state"] = state_text(active)
        context, logits, mask = encoder.encode_question(row)
        final = agent.decide(memory, context, logits, mask)[-1]
        k = len(CELLS)
        if logits.shape != (1, k):
            raise ValueError("generated action width mismatch")
        support = allowed_counts(row)
        if support != [2]:
            raise ValueError("this benchmark requires exactly two executed actions")
        if sum(active) != support[0]:
            raise ValueError("simulator entered invalid state cardinality")
        indices = set_map(
            final["scores"][0, :k],
            final["correction"][0, :k + 1], support,
        )["indices"]
        executed = [int(i in indices) for i in range(k)]
        if sum(executed) not in support:
            raise ValueError("predicted action violates publicly allowed set size")
        gold = rotate(active, rule)
        if rotate(active, 1) == rotate(active, -1):
            raise ValueError("unidentifiable state: left/right yield identical targets")
        nll += float(gold_set_nll(
            final["scores"][0, :k],
            final["correction"][0, :k + 1],
            torch.tensor(gold, dtype=torch.float32, device=encoder.device),
            support, normalize=False,
        ))
        correct = executed == gold
        trajectory_correct = trajectory_correct and correct
        if t == 0:
            first += int(correct)
        else:
            later += int(correct)
        executed_actions.append(executed)

        # Rebuild observation from the policy's actual action, not oracle log.
        feedback = "After this decision the active components are " + active_text(executed) + "."
        if t + 1 < len(states):
            following = states[t + 1].get("oracle_rule_before_decision")
            if following not in (-1, 1):
                raise ValueError("next oracle rule invalid")
            if following != rule:
                feedback += " The rule changed. " + demo_text(following)
        observed_feedback.append(feedback)
        obs = encoder.encode_observation(feedback)
        option_features = encoder.encode_option_meanings(row)
        action = torch.tensor(executed, dtype=torch.float32, device=encoder.device)[None]
        memory = agent.observe(
            memory, context, action, mask, obs,
            option_features=option_features,
            done=torch.tensor([t == len(states) - 1], dtype=torch.bool, device=encoder.device),
        )
        active = executed
    return {
        "first_correct": first, "later_correct": later,
        "n_steps": len(states),
        "trajectory_correct": int(trajectory_correct),
        "total_set_nll": nll,
        "observed_feedback": observed_feedback,
        "executed_actions": executed_actions,
    }


@torch.no_grad()
def evaluate(encoder, agent, episodes: list[dict], *, carry: bool) -> dict:
    first = later = later_count = total = traj = 0
    nll = 0.
    for episode in episodes:
        result = run_episode(encoder, agent, episode, carry=carry)
        first += result["first_correct"]
        later += result["later_correct"]
        later_count += result["n_steps"] - 1
        total += result["n_steps"]
        traj += result["trajectory_correct"]
        nll += result["total_set_nll"]
    return {
        "n_episodes": len(episodes), "n_decisions": total,
        "first_em": first / len(episodes),
        "later_em": later / later_count if later_count else None,
        "step_em": (first + later) / total,
        "trajectory_em": traj / len(episodes),
        "set_nll": nll / total,
        "protocol": "on-policy execution in deterministic cyclic-rule environment",
    }


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
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    if args.output is not None and args.output.exists():
        raise FileExistsError("online report already exists; refuse expensive rerun")
    episodes = []
    with args.episodes.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    if not episodes:
        raise ValueError("empty eval set")
    episodes = episodes[:args.limit] if args.limit else episodes
    for episode in episodes:
        validate_episode(episode)  # fail before allocating 4B Qwen weights
    saved = torch.load(args.head, map_location="cpu", weights_only=True)
    if saved.get("schema") != CKPT_SCHEMA:
        raise ValueError("unsupported episodic checkpoint")
    if saved.get("model_identity") != args.model_identity:
        raise ValueError("head/Qwen checkpoint identity mismatch")
    encoder = FrozenQwenDecisionEncoder(args.qwen_model, device=args.device, dtype=args.dtype)
    config = EpisodicSetConfig(**saved["config"])
    if config.context_dim != encoder.hidden_size or config.max_options < len(CELLS):
        raise ValueError("head incompatible with Qwen hidden width or action count")
    agent = EpisodicSetDecision(config).to(encoder.device).eval()
    agent.load_state_dict(saved["model"], strict=True)
    report = {
        "head": str(args.head), "qwen_identity": args.model_identity,
        "dataset": str(args.episodes),
        "carry": evaluate(encoder, agent, episodes, carry=True),
        "reset_every_step": evaluate(encoder, agent, episodes, carry=False),
        "note": "private rule used by environment only; Qwen actions are executed online",
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError("online report already exists")
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
