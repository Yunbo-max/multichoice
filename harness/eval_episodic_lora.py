#!/usr/bin/env python3
"""Evaluate a C-experiment checkpoint (LoRA Qwen + memory head) on-policy.

Loads the LoRA adapters + memory head saved by ``train_episodic_lora.py`` and runs
the carry / reset / prime_only comparison on held-out test episodes.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

try:
    from train_episodic_lora import TrainableQwenEncoder
    from eval_episodic_text_online import _generator, evaluate
except ModuleNotFoundError:
    from harness.train_episodic_lora import TrainableQwenEncoder
    from harness.eval_episodic_text_online import _generator, evaluate
try:
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
except ModuleNotFoundError:
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--modules", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("report already exists")
    episodes = [json.loads(l) for l in args.episodes.open() if l.strip()]
    if not episodes:
        raise ValueError("empty eval set")
    schema = episodes[0].get("environment_schema")
    gen = _generator(schema)
    for ep in episodes:
        gen.validate_episode(ep)

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    modules = json.loads(args.modules.read_text())
    # lora_r/lora_alpha are not stored; reconstruct the common config (16/32)
    encoder = TrainableQwenEncoder(
        args.qwen_model,
        modules=modules,
        lora_r=16,
        lora_alpha=32,
        device=args.device,
        dtype=args.dtype,
    )
    lora_state = {k: v for k, v in saved["lora"].items()}
    encoder.model.load_state_dict(lora_state, strict=False)
    encoder.model.eval()
    for param in encoder.model.parameters():
        param.requires_grad_(False)

    config = EpisodicSetConfig(**saved["config"])
    agent = EpisodicSetDecision(config).to(encoder.device).eval()
    agent.load_state_dict(saved["model"], strict=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.episodes),
        "schema": schema,
        "carry": evaluate(gen, encoder, agent, episodes, mode="carry"),
        "reset_every_step": evaluate(gen, encoder, agent, episodes, mode="reset"),
        "prime_only": evaluate(gen, encoder, agent, episodes, mode="prime_only"),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
