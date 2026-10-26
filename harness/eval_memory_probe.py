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
    from memory_probe import counterfactual_summary, fit_probe, fit_probe_split, probe_predictions
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
    from harness.memory_probe import fit_probe, fit_probe_split, counterfactual_summary, probe_predictions
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.recurrent_set_decision import set_map
    from harness.train_episodic_features import CKPT_SCHEMA
    from harness.set_distillation import allowed_counts


@torch.no_grad()
def collect_probe(gen, encoder, agent, episodes, *, action_source="policy"):
    """Capture causal memory states under an explicit action source.

    expert: execute the true current-rule action (imitates offline logged
            oracle trajectories; action itself reveals the hidden rule).
    policy: execute the agent's predicted set as in on-policy evaluation.

    A decision never sees gold or the next rule before acting. This routine is
    for evaluation only; expert action is supplied to observe AFTER deciding.
    """
    if action_source not in ("expert", "policy"):
        raise ValueError("action_source must be expert or policy")
    memories, labels, switched, episode_ids = [], [], [], []
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("all probe episodes must have stable nonempty IDs")
        active = initial_state(gen, episode)
        memory = agent.empty_memory(1, device=encoder.device)
        demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
        if demos.shape[1]:
            memory = agent.prime(memory, demos)
        prev_rule = None
        for t, oracle in enumerate(episode["steps"]):
            rule = oracle["oracle_rule_before_decision"]
            if not _valid_rule(gen, rule):
                raise ValueError("invalid rule timeline")
            memories.append(memory.detach().cpu())
            labels.append(int(rule))
            episode_ids.append(episode_id)
            switched.append(prev_rule is not None and rule != prev_rule)
            prev_rule = rule
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
            predicted = [int(i in idx) for i in range(k)]
            executed = list(gold) if action_source == "expert" else predicted
            # Rebuild the causally available feedback, INCLUDING the rule-change
            # demonstration disclosed after this action (the environment may have
            # switched the rule here; the memory must be able to ingest it).
            feedback = (
                "After this decision the active components are "
                + gen.active_text(executed)
                + "."
            )
            if t + 1 < len(episode["steps"]):
                following = episode["steps"][t + 1].get("oracle_rule_before_decision")
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
                done=torch.tensor([False], dtype=torch.bool, device=encoder.device),
            )
            active = executed
    return memories, labels, switched, episode_ids


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
    p.add_argument("--probe-fit-episodes", type=Path,
                   help="independent DEV episode logs for probe fitting; --episodes stays untouched TEST")
    p.add_argument("--compare-action-sources", action="store_true",
                   help="fit on DEV expert states, test the SAME probe on matched TEST expert vs policy states")
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
        if episode.get("environment_schema") != schema:
            raise ValueError("mixed evaluation schemas")
        gen.validate_episode(episode)
    if args.compare_action_sources and args.probe_fit_episodes is None:
        p.error("--compare-action-sources requires --probe-fit-episodes DEV.jsonl")
    fitting_episodes = None
    if args.probe_fit_episodes is not None:
        fitting_episodes = [
            json.loads(line) for line in args.probe_fit_episodes.open()
            if line.strip()
        ]
        if not fitting_episodes:
            raise ValueError("empty DEV probe-fit episodes")
        ids_test = {ep["episode_id"] for ep in episodes}
        ids_fit = {ep["episode_id"] for ep in fitting_episodes}
        if len(ids_fit) != len(fitting_episodes) or len(ids_test) != len(episodes):
            raise ValueError("duplicate episode IDs in fit/test data")
        if ids_test & ids_fit:
            raise ValueError("DEV probe fitting and TEST evaluation episodes overlap")
        for ep in fitting_episodes:
            if ep.get("environment_schema") != schema:
                raise ValueError("probe train/test schema mismatch")
            gen.validate_episode(ep)
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

    memories, labels, switched, episode_ids = collect_probe(gen, encoder, agent, episodes)
    n_classes = len(gen.CELLS) - 1
    if fitting_episodes is None:
        probe, train_acc, test_acc, split = fit_probe_split(
            memories, labels, n_classes=n_classes, groups=episode_ids,
            return_indices=True,
        )
    else:
        fit_memories, fit_labels, _, fit_ids = collect_probe(
            gen, encoder, agent, fitting_episodes
        )
        probe, train_acc = fit_probe(
            fit_memories, fit_labels, n_classes=n_classes,
        )
        predictions = probe_predictions(probe, memories)
        test_acc = sum(int(p == y) for p, y in zip(predictions, labels)) / len(labels)
        split = {
            "train_indices": list(range(len(fit_memories))),
            "test_indices": list(range(len(memories))),
            "train_episodes": len(set(fit_ids)),
            "test_episodes": len(set(episode_ids)),
            "split_unit": "separate_dev_test_episodes",
        }

    def _split_acc(flags):
        # Condition-specific diagnostics on ONE fixed probe and the SAME
        # episode-held-out rows. Never retrain on a switch subgroup.
        idx = [i for i in split["test_indices"] if flags[i]]
        if not idx:
            return None
        predicted = probe_predictions(probe, [memories[i] for i in idx])
        return sum(int(p == labels[i]) for p, i in zip(predicted, idx)) / len(idx)

    chance = 1.0 / n_classes

    state = [0] * len(gen.CELLS)
    state[0] = state[1] = 1
    predicted = counterfactual(gen, encoder, agent, state)
    cf_correct = {k: (predicted[k] == shift(state, k)) for k in predicted}

    action_source_diagnostic = None
    if args.compare_action_sources:
        # The two TEST cohorts are paired by episode and current hidden rule.
        # The ONLY difference is whether past executed actions were oracle
        # actions or the policy's own predictions. The same probe is evaluated
        # on both cohorts without re-fitting on any TEST data.
        fit_expert_mem, fit_expert_labels, _, fit_expert_ids = collect_probe(
            gen, encoder, agent, fitting_episodes, action_source="expert"
        )
        test_expert_mem, expert_labels, _, expert_ids = collect_probe(
            gen, encoder, agent, episodes, action_source="expert"
        )
        if labels != expert_labels or episode_ids != expert_ids:
            raise RuntimeError("expert/policy trajectories must align on TEST rule timeline")
        expert_probe, expert_train_acc = fit_probe(
            fit_expert_mem, fit_expert_labels, n_classes=n_classes,
        )
        expert_preds = probe_predictions(expert_probe, test_expert_mem)
        policy_preds = probe_predictions(expert_probe, memories)
        # Exclude first decision: both modes have exactly the same primed state.
        seen = set()
        later = []
        for i, episode_id in enumerate(episode_ids):
            if episode_id in seen:
                later.append(i)
            seen.add(episode_id)

        def accuracy(indices, predictions):
            if not indices:
                return None
            return sum(int(predictions[i] == labels[i]) for i in indices) / len(indices)

        test_indices = list(range(len(labels)))
        switched_later = [i for i in later if switched[i]]
        stable_later = [i for i in later if not switched[i]]
        action_source_diagnostic = {
            "probe_fit": "EXPERT trajectories on DEV episodes (frozen)",
            "n_dev_episodes": len(set(fit_expert_ids)),
            "n_test_episodes": len(set(episode_ids)),
            "expert_fit_accuracy": expert_train_acc,
            "test_expert_accuracy": accuracy(test_indices, expert_preds),
            "test_policy_accuracy": accuracy(test_indices, policy_preds),
            "test_expert_later_accuracy": accuracy(later, expert_preds),
            "test_policy_later_accuracy": accuracy(later, policy_preds),
            "test_expert_after_switch": accuracy(switched_later, expert_preds),
            "test_policy_after_switch": accuracy(switched_later, policy_preds),
            "test_expert_stable_later": accuracy(stable_later, expert_preds),
            "test_policy_stable_later": accuracy(stable_later, policy_preds),
            "n_later_steps": len(later),
            "n_after_switch_steps": len(switched_later),
            "n_stable_later_steps": len(stable_later),
            "note": "The expert rollout inputs contain past correct executed actions, which are extra information absent from a mistaken on-policy rollout; differences measure history distribution shift, not exclusively forgetting.",
        }

    report = {
        "head": str(args.head),
        "dataset": str(args.episodes),
        "schema": schema,
        "probe": {
            "n_samples": len(memories),
            "split_unit": split["split_unit"],
            "fit_dataset": str(args.probe_fit_episodes) if args.probe_fit_episodes is not None else "group_split_within_eval_dataset",
            "train_episodes": split["train_episodes"],
            "test_episodes": split["test_episodes"],
            "train_steps": len(split["train_indices"]),
            "test_steps": len(split["test_indices"]),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "chance": chance,
            "stored_rule": test_acc > chance + 0.05,
            "after_switch_test_accuracy": _split_acc(switched),
            "no_switch_test_accuracy": _split_acc([not f for f in switched]),
            "after_switch_test_steps": sum(int(switched[i]) for i in split["test_indices"]),
            "no_switch_test_steps": sum(int(not switched[i]) for i in split["test_indices"]),
            "note": "post-switch and no-switch accuracy use the SAME probe trained on disjoint episodes",
        },
        "action_source_diagnostic": action_source_diagnostic,
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
