#!/usr/bin/env python3
"""Prepare (do not submit) reproducible EUR-LEX 4B one-GPU QZ jobs.

Uses qz_c64/c64-warm-bce.json ONLY as the authentic workspace/GPU template.
Require already prepared, TEST-UNOPENED EUR-LEX manifest. Sixteen jobs for
three seeds, or six jobs for one-seed pilot. There is NO destructive rm -rf,
automatic submission, test access, or hidden extra training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shlex


ARMS = {
    "bce_continue": ["--objective", "bce"],
    "bce_goldcount_w100": [
        "--objective", "set", "--set-loss", "set_ce",
        "--count-target", "gold", "--gold-count-weight", "100"
    ],
    "set_ce_independent": ["--objective", "set", "--set-loss", "set_ce"],
    "set_ce_anchor": [
        "--objective", "set", "--set-loss", "set_ce", "--count-anchor"
    ],
    "set_ce_anchor_detach": [
        "--objective", "set", "--set-loss", "set_ce",
        "--count-anchor", "--count-anchor-detach"
    ],
}


def job_specs(template, manifest, *, repo_root, data_root, output_root,
              model_root, seeds=(43,), pilot=True,
              warm_steps=120, continuation_steps=80,
              batch=8, lr=2e-5, max_input=8192, max_batch=8192):
    if manifest.get("dataset") != "coastalcph/lex_glue":
        raise ValueError("not a verified EUR-LEX manifest")
    if manifest.get("test_loaded"):
        raise ValueError("train/DEV job must not use a materialized TEST package")
    if len(manifest.get("label_ids", [])) != 100:
        raise ValueError("EUR-LEX requires the complete, fixed 100-class catalogue")
    if any(part not in manifest.get("splits", {}) for part in ("train", "validation")):
        raise ValueError("training and validation splits are mandatory")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed list is empty or repeated")
    if not (0 < lr < 1 and batch >= 1 and continuation_steps > 0 and warm_steps > 0):
        raise ValueError("invalid training hyperparameters")
    if not (max_input >= 512 and max_batch >= max_input):
        raise ValueError("invalid context/microbatch budget")
    if manifest.get("max_input_tokens") != max_input:
        raise ValueError("input token cap differs from the audited data package")
    count = (manifest.get("pilot", {}).get("train") if pilot
             else manifest["splits"]["train"]["n"])
    if count is None or (count + batch - 1) // batch < max(warm_steps, continuation_steps):
        raise ValueError("requested optimizer steps exceed available train rows")
    root = Path(repo_root)
    data = Path(data_root)
    runs = Path(output_root)
    if any(not p.is_absolute() for p in (root, data, runs, Path(model_root))):
        raise ValueError("cluster root/data/run/model paths must be absolute")
    train = data / ("pilot/train.jsonl" if pilot else "train.jsonl")
    dev = data / ("pilot/dev.jsonl" if pilot else "dev.jsonl")
    executable = root / ".venv-sii/bin/torchrun"
    modules = root / "innerjev/configs/4B/modules.json"
    trainer = root / "harness/train_multiselect.py"
    warm_final = runs / "warm_bce/final"
    tags = "pilot" if pilot else "main"
    generic = [
        "--train", str(train), "--dev", str(dev), "--modules", str(modules),
        "--target", "gold", "--global-batch", str(batch), "--epochs", "1",
        "--eval-every-frac", "1.0", "--lr", str(lr), "--lora",
        "--lora-r", "16", "--lora-alpha", "32",
        "--max-input-tokens", str(max_input), "--max-batch-tokens", str(max_batch),
        "--expect-trainable", "0"
    ]
    jobs = [("warm-bce-s44", ["--objective", "bce"], 44, Path(model_root),
             runs / "warm_bce", warm_steps)]
    for seed in seeds:
        for arm, extra in ARMS.items():
            jobs.append((f"{arm}-s{seed}", extra, seed, warm_final,
                         runs / arm / f"seed_{seed}", continuation_steps))
    outputs = []
    template = dict(template)
    for name, objective, seed, initial, final_output, steps in jobs:
        args = [str(executable), "--standalone", "--nnodes", "1",
                "--nproc_per_node", "1", str(trainer),
                "--model", str(initial), *generic,
                "--seed", str(seed), "--max-steps", str(steps),
                "--output", str(final_output), *objective]
        log_dir = root / "logs/qz/eurlex"
        log_path = log_dir / f"{tags}-{name}.log"
        # Shell-safe, non-destructive; all prerequisites must exist.
        checks = [
            "set -euo pipefail",
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTHONPATH=" + shlex.quote(str(root)),
            "test -s " + shlex.quote(str(train)),
            "test -s " + shlex.quote(str(dev)),
            "test -s " + shlex.quote(str(data / "manifest.json")),
            "test -s " + shlex.quote(str(initial / "config.json")),
            "test ! -e " + shlex.quote(str(final_output)),
            "mkdir -p " + shlex.quote(str(log_dir)),
            "cd " + shlex.quote(str(root)),
        ]
        # Validate that remote data remains the same SHA and no test exists.
        verify = (
            "import json,pathlib,hashlib;"
            f"p=pathlib.Path({str(data / 'manifest.json')!r});"
            "j=json.loads(p.read_text());"
            "assert j['dataset']=='coastalcph/lex_glue' and not j['test_loaded'];"
            "assert len(j['label_ids'])==100;"
            f"assert j['max_input_tokens']=={max_input};"
        )
        checks.append(shlex.quote(str(root / ".venv-sii/bin/python"))
                      + " -c " + shlex.quote(verify))
        checks.append(shlex.join(args) + " > " + shlex.quote(str(log_path)) + " 2>&1")
        payload = dict(template)
        payload["name"] = f"paper1-eurlex-{tags}-{name}"
        payload["description"] = (
            "EUR-LEX train/DEV only; no test; Qwen3.5-4B LoRA; "
            f"arm={name}; common warm-start; status=STAGED_NOT_SUBMITTED"
        )
        payload["command"] = "\n".join(checks) + "\n"
        if any(x in payload["command"] for x in ("rm -rf", "/test.jsonl", "--expected-split test")):
            raise ValueError("unsafe QZ payload")
        outputs.append(payload)
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, default=Path("qz_c64/c64-warm-bce.json"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--full", action="store_true", help="use official full TRAIN/DEV (not TEST)")
    p.add_argument("--warm-steps", type=int, default=120)
    p.add_argument("--continuation-steps", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--max-batch-tokens", type=int, default=8192)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        p.error("refusing to overwrite existing QZ payloads")
    template = json.loads(args.template.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plans = job_specs(template, manifest, repo_root=args.repo_root,
                     data_root=args.data_root, output_root=args.run_root,
                     model_root=args.model_root, seeds=tuple(args.seeds),
                     pilot=not args.full, warm_steps=args.warm_steps,
                     continuation_steps=args.continuation_steps,
                     batch=args.batch, lr=args.lr,
                     max_input=args.max_input_tokens, max_batch=args.max_batch_tokens)
    args.output_dir.mkdir(parents=True)
    for pld in plans:
        path = args.output_dir / (pld["name"] + ".json")
        path.write_text(json.dumps(pld, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":"STAGED_NOT_SUBMITTED",
                      "jobs":len(plans),"output_dir":str(args.output_dir),
                      "first_warm_job":plans[0]["name"]}, indent=2))


if __name__ == "__main__":
    main()
