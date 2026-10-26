#!/usr/bin/env python3
"""Merge two episode feature caches into one (for DAgger data aggregation)."""

from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("merged cache already exists")
    expert = torch.load(args.expert, map_location="cpu", weights_only=True)
    policy = torch.load(args.policy, map_location="cpu", weights_only=True)
    if expert.get("schema") != policy.get("schema") or expert.get(
        "model_identity"
    ) != policy.get("model_identity"):
        raise ValueError("cache schema / model identity mismatch")
    merged = dict(expert)
    merged["episodes"] = expert["episodes"] + policy["episodes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"merged {len(expert['episodes'])} expert + {len(policy['episodes'])} policy episodes -> {args.output}"
    )


if __name__ == "__main__":
    main()
