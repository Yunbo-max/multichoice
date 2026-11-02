#!/usr/bin/env python3
"""Create pre-registered, equal-update Static SetJev runs; never execute GPU jobs.

Example:
python paper_static/plan_experiments.py --output paper_static/runs_4b.jsonl
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ARMS = {
    "bce_continue": ("bce", None, False, False, False),
    "set_ce_independent": ("set", "set_ce", False, False, False),
    "set_ce_anchor": ("set", "set_ce", True, False, False),
    "set_ce_anchor_detach": ("set", "set_ce", True, True, False),
    "induced_independent": ("set", "induced_marginal", False, False, False),
    "induced_anchor": ("set", "induced_marginal", True, False, False),
    "induced_anchor_detach": ("set", "induced_marginal", True, True, False),
    "bce_plus_gold_count": ("set", "set_ce", False, False, True),
}

def sha256_file(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def make_plan(model, modules, train, dev, out_root, seeds=(43, 44, 45),
              model_tag="qwen3.5-4b-bce-s44", epochs=1, batch=64, lr=2e-6,
              world_size=4, max_steps=79, lora=False):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_tag):
        raise ValueError("model_tag must be a simple path component")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be nonempty, unique and nonnegative")
    if world_size < 1 or batch < 1 or batch % world_size or epochs < 1:
        raise ValueError("batch must divide by world_size; epochs positive")
    if lr <= 0 or max_steps < 1:
        raise ValueError("learning rate and max_steps must be positive")
    hashes = {"train": sha256_file(train), "dev": sha256_file(dev),
              "modules": sha256_file(modules)}
    rows = []
    for seed in seeds:
        for arm, (objective, family, anchor, detach, goldcount) in ARMS.items():
            run_id = f"{model_tag}/seed_{seed}/{arm}"
            dest = str(Path(out_root) / run_id)
            argv = [
                "torchrun", "--standalone", "--nnodes=1",
                f"--nproc_per_node={world_size}", "harness/train_multiselect.py",
                "--train", str(train), "--dev", str(dev),
                "--model", str(model), "--modules", str(modules),
                "--target", "gold", "--objective", objective,
                "--output", dest, "--lr", str(lr), "--seed", str(seed),
                "--global-batch", str(batch), "--epochs", str(epochs),
                "--max-steps", str(max_steps), "--eval-every-frac", "1.0",
                "--expect-trainable", "0",
                "--init-checkpoint-id", f"{model_tag}|same_pretrained_checkpoint",
            ]
            if objective == "set":
                argv += ["--set-loss", family]
                if anchor:
                    argv.append("--count-anchor")
                if detach:
                    argv.append("--count-anchor-detach")
                if goldcount:
                    argv += ["--count-target", "gold"]
            if lora:
                argv += ["--lora", "--lora-r", "16", "--lora-alpha", "32"]
            rows.append({
                "id": run_id, "model": str(model), "model_tag": model_tag,
                "checkpoint_is_pretuned_bce": True, "arm": arm, "seed": seed,
                "objective": objective, "family": family,
                "count_anchor": anchor, "count_anchor_detach": detach,
                "gold_count": goldcount, "train_sha256": hashes["train"],
                "dev_sha256": hashes["dev"], "module_sha256": hashes["modules"],
                "nproc_per_node": world_size, "update_budget": max_steps,
                "checkpoint_selection": "terminal_step_only", "argv": argv,
                "output": dest, "status": "NOT_RUN",
            })
    if len({r["output"] for r in rows}) != len(rows):
        raise RuntimeError("run outputs collide")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/qz/grouped_bce_4gpu_s44/v1/final")
    parser.add_argument("--modules", default="innerjev/configs/4B/modules.json")
    parser.add_argument("--model-tag", default="qwen3.5-4b-bce-s44")
    parser.add_argument("--train", default="data/multiselect_grouped_multi_5k/train.jsonl")
    parser.add_argument("--dev", default="data/multiselect_grouped_multi/dev.jsonl")
    parser.add_argument("--run-root", default="runs/paper_static")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=79)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    if args.require_files:
        needed = [args.train, args.dev, args.modules, Path(args.model) / "config.json"]
        absent = [str(p) for p in needed if not Path(p).is_file()]
        if absent:
            parser.error("required local files missing: " + ", ".join(absent))
    if args.output.exists():
        parser.error(f"refusing to overwrite existing experiment manifest: {args.output}")
    runs = make_plan(args.model, args.modules, args.train, args.dev, args.run_root,
                     seeds=tuple(args.seeds), model_tag=args.model_tag,
                     epochs=args.epochs, batch=args.batch, lr=args.lr,
                     max_steps=args.max_steps, world_size=args.world_size,
                     lora=args.lora)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for run in runs:
            out.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"runs": len(runs), "model": args.model, "output": str(args.output),
                      "all_status": "NOT_RUN", "arms": list(ARMS)}, indent=2))

if __name__ == "__main__":
    main()
