#!/usr/bin/env python3
"""P0: structured set readout on frozen v4 checkpoints (independent on-policy).

Runs THREE separate on-policy rollouts, one per readout, so each readout drives
its own state distribution:
  raw         set-MAP over all C(7,2)=21 subsets (current decoder)
  structured  softmax over the 6 reachable successors E_k = sum_{i in shift_k(S)} s_i
  shift_head  softmax over the 6 shifts from the memory-conditioned shift head

Also reports theory baselines (random shift EM 1/6, uniform-6 NLL log 6,
uniform-21 NLL log 21).
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch

try:
    from eval_episodic_text_online import (
        _generator,
        _apply_rule,
        _valid_rule,
        initial_state,
    )
    from structured_readout import (
        baselines,
        shift_head_predict,
        structured_nll,
        structured_predict,
    )
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
    from harness.structured_readout import (
        baselines,
        shift_head_predict,
        structured_nll,
        structured_predict,
    )
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map, gold_set_nll
    from harness.train_episodic_features import CKPT_SCHEMA
    from harness.set_distillation import allowed_counts

READOUTS = ("raw", "structured", "shift_head")


def _predict(readout, gen, active, scores, correction, support, shift_logits, tau):
    k = len(gen.CELLS)
    if readout == "raw":
        idx = set_map(scores, correction, support)["indices"]
        return [int(i in idx) for i in range(k)]
    if readout == "structured":
        return structured_predict(active, scores.float().cpu().tolist(), tau=tau)["set"]
    if readout == "shift_head":
        return shift_head_predict(active, shift_logits, tau=tau)["set"]
    raise ValueError(f"unknown readout {readout}")


def _step_nll(
    readout, active, scores, correction, support, rule, shift_logits, tau, gold_tensor
):
    if readout == "raw":
        return float(
            gold_set_nll(scores, correction, gold_tensor, support, normalize=False)
        )
    if readout == "structured":
        return structured_nll(active, scores.float().cpu().tolist(), rule, tau=tau)
    probs = shift_head_predict(active, shift_logits, tau=tau)["probs"]
    return -math.log(max(probs[rule], 1e-300))


@torch.no_grad()
def run_readout(gen, encoder, agent, episode, tau, readout):
    states = episode["steps"]
    active = initial_state(gen, episode)
    memory = agent.empty_memory(1, device=encoder.device)
    demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
    if demos.shape[1]:
        memory = agent.prime(memory, demos)
    first = later = 0
    traj = True
    nll = 0.0
    k = len(gen.CELLS)
    for t, oracle in enumerate(states):
        rule = oracle["oracle_rule_before_decision"]
        if not _valid_rule(gen, rule):
            raise ValueError("invalid rule timeline")
        row = dict(oracle["decision"])
        row["state"] = gen.state_text(active)
        context, logits, mask = encoder.encode_question(row)
        final = agent.decide(memory, context, logits, mask)[-1]
        scores = final["scores"][0, :k]
        correction = final["correction"][0, : k + 1]
        shift_logits = final["shift_logits"][0].float().cpu().tolist()
        support = allowed_counts(row)
        gold = _apply_rule(gen, active, rule)
        gold_tensor = torch.tensor(gold, dtype=torch.float32, device=encoder.device)

        pred = _predict(
            readout, gen, active, scores, correction, support, shift_logits, tau
        )
        nll += _step_nll(
            readout,
            active,
            scores,
            correction,
            support,
            rule,
            shift_logits,
            tau,
            gold_tensor,
        )

        correct = pred == gold
        if t == 0:
            first += int(correct)
        else:
            later += int(correct)
        traj = traj and correct

        # advance the environment with THIS readout's own prediction
        executed = pred
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
        "first": first,
        "later": later,
        "traj": int(traj),
        "nll": nll,
    }


def evaluate(gen, encoder, agent, episodes, tau):
    totals = {
        m: {"first": 0, "later": 0, "traj": 0, "nll": 0.0, "steps": 0} for m in READOUTS
    }
    for episode in episodes:
        for mode in READOUTS:
            out = run_readout(gen, encoder, agent, episode, tau, mode)
            totals[mode]["first"] += out["first"]
            totals[mode]["later"] += out["later"]
            totals[mode]["traj"] += out["traj"]
            totals[mode]["nll"] += out["nll"]
            totals[mode]["steps"] += out["n_steps"]
    n = len(episodes)
    report = {}
    for mode in READOUTS:
        t = totals[mode]
        later_count = t["steps"] - n
        report[mode] = {
            "step_em": (t["first"] + t["later"]) / t["steps"],
            "later_em": t["later"] / later_count if later_count else None,
            "trajectory_em": t["traj"] / n,
            "set_nll": t["nll"] / t["steps"],
        }
    return {"n_episodes": n, "n_decisions": totals["raw"]["steps"], **report}


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
        "note": "each readout runs its own on-policy rollout (independent state distribution)",
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
