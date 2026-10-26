#!/usr/bin/env python3
"""Evaluate a C-experiment checkpoint (LoRA Qwen + memory head) on-policy.

Loads the LoRA adapters + memory head saved by ``train_episodic_lora.py`` and runs
the carry / reset / prime_only comparison on held-out test episodes.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import torch

try:
    from train_episodic_lora import TrainableQwenEncoder
    from eval_episodic_text_online import _generator, evaluate
except ModuleNotFoundError:
    from harness.train_episodic_lora import TrainableQwenEncoder
    from harness.eval_episodic_text_online import _generator, evaluate
try:
    from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from episodic_lora_checkpoint import load_lora_tensors_verified
except ModuleNotFoundError:
    from harness.episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
    from harness.episodic_lora_checkpoint import load_lora_tensors_verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--modules", type=Path, required=True)
    p.add_argument("--model-identity", default="BCE_S44_FINAL")
    p.add_argument("--legacy-lora-r", type=int, default=16,
                   help="only used for checkpoints created before LoRA metadata was saved")
    p.add_argument("--legacy-lora-alpha", type=int, default=32,
                   help="only used for legacy checkpoints")
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
    if saved.get("schema") != "episodic-set-head-v1":
        raise ValueError("unsupported episodic checkpoint schema")
    if saved.get("model_identity") != args.model_identity:
        raise ValueError("checkpoint and specified Qwen model identity mismatch")
    if saved.get("checkpoint_type") not in (
        None, "qwen3.5_lora_plus_episodic_head"
    ):
        raise ValueError("checkpoint type is not LoRA + episodic head")
    modules = json.loads(args.modules.read_text())
    meta = saved.get("lora_config")
    if meta is None:
        warnings.warn(
            "Legacy checkpoint lacks LoRA rank/alpha metadata; using "
            "explicit legacy settings and verifying all restored tensors.",
            RuntimeWarning,
        )
        lora_r, lora_alpha = args.legacy_lora_r, args.legacy_lora_alpha
    else:
        if sorted(meta.get("target_modules", [])) != sorted(modules):
            raise ValueError("checkpoint LoRA target_modules differ from eval config")
        lora_r, lora_alpha = int(meta["r"]), int(meta["alpha"])
        if meta.get("dtype") != args.dtype:
            raise ValueError("checkpoint and evaluation precision settings differ")
    encoder = TrainableQwenEncoder(
        args.qwen_model,
        modules=modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        device=args.device,
        dtype=args.dtype,
    )
    restored_tensors = load_lora_tensors_verified(encoder.model, saved["lora"])
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
        "lora_tensors_verified": restored_tensors,
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha,
        "selected_epoch": saved.get("selected_epoch"),
        "dev_selection": saved.get("dev_selection"),
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
