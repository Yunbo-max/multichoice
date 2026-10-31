#!/usr/bin/env python3
"""Compute per-arm count KL for the K=16 C-regime anchor-vs-BCE comparison (CPU).

Reads saved predictions (anchor: count_probs; BCE: independent probabilities ->
Poisson-binomial count law) and the oracle count distribution, and reports mean
count KL over contexts and seeds. DEV only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def poisson_binomial(probs):
    n = len(probs)
    dp = [1.0] + [0.0] * n
    for p in probs:
        for m in range(n, 0, -1):
            dp[m] = dp[m] * (1 - p) + dp[m - 1] * p
        dp[0] *= (1 - p)
    return dp


def count_kl(model_rho, true_rho):
    return sum(p * math.log(p / max(q, 1e-12)) for p, q in zip(true_rho, model_rho) if p > 0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--predictions-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    args = p.parse_args(argv)

    oracle = [json.loads(l) for l in
              (args.data_root / "oracle_DO_NOT_USE_AS_MODEL_INPUT" / "dev.jsonl").read_text().splitlines()]
    true_rho = {o["context_id"]: o["true_count_probs"] for o in oracle}

    result = {"seeds": {}}
    for seed in args.seeds:
        arm_kl = {}
        for arm in ("anchor", "bce"):
            preds = {}
            for line in (args.predictions_root / arm / f"seed_{seed}" / "predictions.jsonl").read_text().splitlines():
                r = json.loads(line)
                preds[r["group_id"]] = preds.get(r["group_id"], r)
            kls = []
            for cid, rho in true_rho.items():
                r = preds.get(cid)
                if r is None:
                    continue
                k = len(rho) - 1
                if arm == "anchor" and "count_probs" in r:
                    model_rho = r["count_probs"][: k + 1]
                else:
                    model_rho = poisson_binomial(r["p"][: k])
                kls.append(count_kl(model_rho, rho))
            arm_kl[arm] = sum(kls) / len(kls) if kls else None
        result["seeds"][str(seed)] = arm_kl

    mean = {}
    for arm in ("anchor", "bce"):
        vals = [result["seeds"][str(s)][arm] for s in args.seeds
                if result["seeds"][str(s)][arm] is not None]
        mean[arm] = sum(vals) / len(vals) if vals else None
    result["seed_mean"] = mean
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
