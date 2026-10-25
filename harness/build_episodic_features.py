#!/usr/bin/env python3
"""Build chronological episodic Qwen features from real environment transcripts.

One JSONL object per EPISODE:
  episode_id: stable string
  demonstrations: list of KNOWN demonstration strings (may be empty)
  steps: chronological list of
    step_index, decision {state, question, options, selection_mode,
                          allow_empty}, action_ids (stable semantic IDs),
    gold (multi-hot for supervision), executed_set (observed chosen multi-hot),
    post_observation (string or null), done (true only last step)

The next observation must have actually arrived after executed_set.
Gold is never rendered into the prompt or passed to the memory update.
Train/dev/test episode logs must be split before encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from train_episodic_features import SCHEMA
    from set_distillation import allowed_counts
except ModuleNotFoundError:
    from harness.episodic_qwen_bridge import FrozenQwenDecisionEncoder
    from harness.train_episodic_features import SCHEMA
    from harness.set_distillation import allowed_counts


def encode_episode(source: dict, encoder) -> dict:
    """Pure adapter: can be tested with a fake encoder without loading Qwen."""
    schema = source.get("environment_schema")
    if schema is not None:
        if schema == "cyclic5_fixed2_v3":
            try:
                from gen_text_episodes import validate_episode
            except ModuleNotFoundError:
                from harness.gen_text_episodes import validate_episode
        elif schema == "cyclic7_shift2_v4":
            try:
                from gen_text_episodes_shift import validate_episode
            except ModuleNotFoundError:
                from harness.gen_text_episodes_shift import validate_episode
        else:
            raise ValueError("unknown/legacy synthetic environment protocol")
        validate_episode(source)
    ident = source.get("episode_id")
    if not isinstance(ident, str) or not ident:
        raise ValueError("episode_id must be nonempty")
    demos = source.get("demonstrations", [])
    if not isinstance(demos, list) or any(not isinstance(x, str) for x in demos):
        raise ValueError("demonstrations must be a list of visible strings")
    sequence = source.get("steps")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("episode must have one or more chronological steps")
    demo_features = encoder.encode_demonstrations(demos)[0].detach().cpu()
    entries = []
    for t, event in enumerate(sequence):
        if event.get("step_index") != t:
            raise ValueError("steps must appear in true chronological order")
        # Whitelist exactly the fields the old SetJev prompt builder can see.
        raw = event.get("decision")
        if not isinstance(raw, dict):
            raise ValueError("missing decision")
        prompt_row = {
            key: raw[key]
            for key in (
                "state",
                "question",
                "options",
                "selection_mode",
                "allow_empty",
                "allowed_cardinalities",
            )
            if key in raw
        }
        if prompt_row.get("selection_mode") != "multi":
            raise ValueError("expected a multi-select problem")
        opts = prompt_row.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            raise ValueError("at least 2 options required")
        k = len(opts)
        action_ids = event.get("action_ids")
        if (
            not isinstance(action_ids, list)
            or len(action_ids) != k
            or any(not isinstance(a, str) or not a for a in action_ids)
        ):
            raise ValueError("semantic action_ids must be K nonempty strings")
        if len(set(action_ids)) != k:
            raise ValueError("duplicate semantic action identity")
        gold = event.get("gold")
        executed = event.get("executed_set")
        if not all(
            isinstance(y, list)
            and len(y) == k
            and all(type(x) is int and x in (0, 1) for x in y)
            for y in (gold, executed)
        ):
            raise ValueError("gold and executed_set must be K-length multi-hot lists")
        support = allowed_counts(prompt_row)
        if sum(gold) not in support or sum(executed) not in support:
            raise ValueError("gold/executed action violates public selection protocol")
        if event.get("done") is not (t == len(sequence) - 1):
            raise ValueError("done must be true only for final step")
        feedback = event.get("post_observation")
        if feedback is not None and not isinstance(feedback, str):
            raise ValueError("post_observation must be text or null")
        context, logits, mask = encoder.encode_question(prompt_row)
        options = encoder.encode_option_meanings(prompt_row)
        observed = (
            encoder.encode_observation(feedback)[0].detach().cpu()
            if feedback is not None
            else None
        )
        if (
            context.shape[0] != 1
            or logits.shape != (1, k)
            or options.shape != (1, k, context.shape[1])
        ):
            raise ValueError("encoder returned wrong decision/option dimensions")
        entries.append(
            {
                "step_index": t,
                "context": context[0].detach().cpu(),
                "native_logits": logits[0].detach().cpu(),
                "option_features": options[0].detach().cpu(),
                "option_ids": action_ids,
                "gold": torch.tensor(gold, dtype=torch.float32),
                "executed_set": torch.tensor(executed, dtype=torch.float32),
                "post_observation": observed,
                "allowed": support,
                "done": event["done"],
                "rule": event.get("oracle_rule_before_decision"),
            }
        )
    return {
        "episode_id": ident,
        "demonstrations": demo_features,
        "steps": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new episodic cache filename")
    encoder = FrozenQwenDecisionEncoder(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
    )
    episodes = []
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                episodes.append(encode_episode(event, encoder))
            except Exception as exc:
                raise ValueError(f"{args.input}:{line_number}: {exc}") from exc
            print(f"encoded episode {len(episodes)}", flush=True)
    if not episodes:
        raise ValueError("empty episode log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "model_identity": args.model_identity,
            "episodes": episodes,
            "source_path": str(args.input),
            "note": "Post-action feedback and executed actions come from logged environment",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
