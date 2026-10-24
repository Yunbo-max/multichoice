#!/usr/bin/env python3
"""Train persistent SetJev memory on chronological, cached Qwen episodes.

OFFLINE REPLAY, NOT ONLINE SUCCESS: logged executed actions and post-action
observations update memory, while target gold labels are used for losses only.
Never shuffle steps inside one episode or feed future feedback into decide().
"""
from __future__ import annotations

import argparse
import hashlib
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

SCHEMA = "episodic-set-feature-v1"
CKPT_SCHEMA = "episodic-set-head-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_episodes(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema") != SCHEMA or not isinstance(data.get("model_identity"), str):
        raise ValueError("bad episode-cache schema / model identity")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("cache needs a nonempty episodes list")
    ids, dim, max_options = set(), None, 0
    for ep in episodes:
        ident = ep.get("episode_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("empty or duplicate episode ID")
        ids.add(ident)
        demos, steps = ep.get("demonstrations"), ep.get("steps")
        if not isinstance(demos, torch.Tensor) or demos.ndim != 2:
            raise ValueError("demonstrations must be D x H tensor (D may be zero)")
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a nonempty chronological list")
        width = demos.shape[1]
        if dim is None:
            dim = width
        if dim != width or width < 1:
            raise ValueError("inconsistent backbone feature dimension")
        for t, step in enumerate(steps):
            if step.get("step_index") != t:
                raise ValueError("step_index must be chronological without gaps")
            context = step.get("context")
            scores = step.get("native_logits")
            gold = step.get("gold")
            action = step.get("executed_set")
            option_features = step.get("option_features")
            if not isinstance(context, torch.Tensor) or context.shape != (width,):
                raise ValueError("context must be H")
            if not all(
                isinstance(z, torch.Tensor) and z.ndim == 1
                for z in (scores, gold, action)
            ):
                raise ValueError("native logits, gold, executed_set must be 1D tensors")
            k = scores.numel()
            if k < 2 or gold.shape != (k,) or action.shape != (k,):
                raise ValueError("invalid candidate dimensions")
            if not bool(((gold == 0) | (gold == 1)).all()
                        and ((action == 0) | (action == 1)).all()):
                raise ValueError("gold and recorded action must be binary")
            allowed = step.get("allowed")
            if not isinstance(allowed, list) or not allowed:
                raise ValueError("missing public cardinality support")
            if sorted(set(allowed)) != allowed or min(allowed) < 0 or max(allowed) > k:
                raise ValueError("invalid cardinality support")
            if int(gold.sum()) not in allowed:
                raise ValueError("gold violates public selection protocol")
            if step.get("done") is not (t == len(steps) - 1):
                raise ValueError("terminal done flag must be true on last step only")
            ids_for_step = step.get("option_ids")
            if not isinstance(ids_for_step, list) or len(ids_for_step) != k:
                raise ValueError("option_ids must align to logits")
            if any(not isinstance(v, str) or not v for v in ids_for_step):
                raise ValueError("option IDs must be nonempty strings")
            if len(set(ids_for_step)) != k:
                raise ValueError("duplicate option IDs")
            if option_features is not None and (
                not isinstance(option_features, torch.Tensor)
                or option_features.shape != (k, width)
            ):
                raise ValueError("option_features must be K x H or None")
            if t > 0 and ids_for_step != steps[t - 1]["option_ids"] and option_features is None:
                raise ValueError("changed option menu requires semantic option_features")
            observed = step.get("post_observation")
            if observed is not None and (
                not isinstance(observed, torch.Tensor) or observed.shape != (width,)
            ):
                raise ValueError("post_observation must be H or None")
            tensors = [context, scores, gold, action, demos]
            tensors.extend(z for z in (observed, option_features) if z is not None)
            if any(not bool(torch.isfinite(z).all()) for z in tensors):
                raise ValueError("nonfinite cached feature")
            max_options = max(max_options, k)
        # If a menu changes anywhere, all actions must have semantic
        # embeddings; otherwise early transitions would still encode A/B
        # position rather than the changing action's meaning.
        variable_menu = any(
            step["option_ids"] != steps[0]["option_ids"]
            for step in steps[1:]
        )
        if variable_menu and any(step.get("option_features") is None for step in steps):
            raise ValueError("changing menus require semantic features for ALL steps")
    return data, ids, dim, max_options


def replay_episode(agent, episode, device, *, carry=True):
    """Differentiable offline replay. Gold is NEVER a memory input."""
    demos = episode["demonstrations"].unsqueeze(0).to(device).float()
    memory = agent.empty_memory(1, device=device)
    if demos.shape[1]:
        memory = agent.prime(memory, demos)
    losses, correct, count_error = [], 0, 0
    for t, step in enumerate(episode["steps"]):
        if t > 0 and not carry:
            memory = agent.empty_memory(1, device=device)
        context = step["context"][None].to(device).float()
        native = step["native_logits"][None].to(device).float()
        gold = step["gold"].to(device).float()
        k = native.shape[1]
        mask = torch.ones(1, k, dtype=torch.bool, device=device)
        final = agent.decide(memory, context, native, mask)[-1]
        loss = gold_set_nll(
            final["scores"][0], final["correction"][0, : k + 1],
            gold, step["allowed"],
        )
        losses.append(loss)
        with torch.no_grad():
            selected = set_map(
                final["scores"][0], final["correction"][0, : k + 1],
                step["allowed"],
            )["indices"]
            guessed = torch.zeros(k, device=device)
            guessed[selected] = 1
            correct += int(torch.equal(guessed, gold))
            count_error += abs(len(selected) - int(gold.sum()))
        # Recorded executed action and environment feedback are observable;
        # gold is not passed to observe(), even for teacher-forced logs.
        action = step["executed_set"][None].to(device).float()
        observed = step["post_observation"]
        observed = observed[None].to(device).float() if observed is not None else None
        options = step["option_features"]
        options = options[None].to(device).float() if options is not None else None
        memory = agent.observe(
            memory, context, action, mask, observed,
            option_features=options,
            done=torch.tensor([step["done"]], device=device, dtype=torch.bool),
        )
    return {
        "loss": torch.stack(losses).mean(), "em": correct,
        "n_steps": len(losses), "count_error": count_error,
        "episode_exact": int(correct == len(losses)),
    }


@torch.no_grad()
def evaluate(agent, cache, device, *, carry=True):
    agent.eval()
    n = correct = complete = 0
    nll = count_mae = 0.0
    for episode in cache["episodes"]:
        result = replay_episode(agent, episode, device, carry=carry)
        n += result["n_steps"]
        correct += result["em"]
        complete += result["episode_exact"]
        nll += float(result["loss"]) * result["n_steps"]
        count_mae += result["count_error"]
    return {
        "offline_step_em": correct / n,
        "offline_episode_exact": complete / len(cache["episodes"]),
        "offline_set_nll_per_step": nll / n,
        "count_mae": count_mae / n,
        "n_steps": n, "n_episodes": len(cache["episodes"]),
        "protocol": "offline logged-action replay (NOT on-policy rollout)",
    }


def train(args):
    train_data, train_ids, hidden, train_k = read_episodes(args.train_cache)
    dev_data, dev_ids, dev_hidden, dev_k = read_episodes(args.dev_cache)
    if train_ids & dev_ids:
        raise ValueError("train and dev episode IDs overlap")
    if hidden != dev_hidden or train_data["model_identity"] != dev_data["model_identity"]:
        raise ValueError("backbone checkpoint / feature size mismatch")
    max_k = max(train_k, dev_k)
    if max_k > args.max_options:
        raise ValueError("requested max_options too low")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    config = EpisodicSetConfig(
        context_dim=hidden, latent_dim=args.latent_dim,
        max_options=max_k, inner_steps=args.inner_steps,
    )
    agent = EpisodicSetDecision(config).to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr)
    if args.output.exists():
        raise FileExistsError("use a new output folder")
    args.output.mkdir(parents=True)
    best, history = -1.0, []
    for epoch in range(args.epochs):
        order = list(range(len(train_data["episodes"])))
        rng.shuffle(order)
        agent.train()
        optimizer.zero_grad(set_to_none=True)
        train_nll = 0.0
        for index, key in enumerate(order):
            result = replay_episode(agent, train_data["episodes"][key], device)
            # An incomplete final accumulation group must not receive
            # less weight than a full group of episodes.
            group_start = (index // args.accumulate_episodes) * args.accumulate_episodes
            group_size = min(args.accumulate_episodes, len(order) - group_start)
            (result["loss"] / group_size).backward()
            train_nll += float(result["loss"].detach())
            if (index + 1) % args.accumulate_episodes == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    agent.parameters(), args.grad_clip, error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        carry = evaluate(agent, dev_data, device, carry=True)
        reset = evaluate(agent, dev_data, device, carry=False)
        row = {
            "epoch": epoch + 1, "train_nll": train_nll / len(order),
            "dev_carry": carry, "dev_reset": reset,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if carry["offline_step_em"] >= best:
            best = carry["offline_step_em"]
            torch.save({
                "schema": CKPT_SCHEMA,
                "config": asdict(config),
                "model": {key: value.detach().cpu()
                          for key, value in agent.state_dict().items()},
                "model_identity": train_data["model_identity"],
                "train_hash": sha256(args.train_cache),
                "dev_hash": sha256(args.dev_cache),
                "selected_epoch": epoch + 1,
                "dev_carry": carry, "dev_reset": reset,
            }, args.output / "best.pt")
    (args.output / "summary.json").write_text(
        json.dumps({"history": history}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--accumulate-episodes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulate_episodes < 1 or args.lr <= 0 or args.grad_clip <= 0:
        parser.error("invalid training settings")
    train(args)


if __name__ == "__main__":
    main()
