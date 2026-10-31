#!/usr/bin/env python3
"""Generate C-regime (cardinality-dependent) controlled data for large K.

Uses the O(K^2) log-partition DP (set_distillation.log_partitions) instead of
2^K enumeration, so K can go well beyond the pairwise exact limit. The
cardinality correction is a_m = -strength * (m - center)^2; candidate scores come
from observable utilities. Oracle marginals/count law and sampled gold sets are
written exactly as controlled_regimes does, but C-only.

This is the E4 data source for "does anchor still beat BCE at larger K".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_partitions(scores):
    # log Z_m(s) for m = 0..k, O(k^2), stable via max-centered DP.
    from set_distillation import log_partitions
    return log_partitions(scores)


def _logsumexp(xs):
    hi = max(xs)
    return hi + math.log(sum(math.exp(x - hi) for x in xs))


def _sample_count(rng, log_rho):
    mx = max(log_rho)
    weights = [math.exp(v - mx) for v in log_rho]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for m, w in enumerate(weights):
        acc += w
        if r <= acc:
            return m
    return len(weights) - 1


def _sample_set_given_count(rng, scores, m):
    # Gumbel-max: P(Y|m) proportional to exp(sum_{i in Y} s_i).
    gumbel = [s + random.gumbel if False else s - math.log(-math.log(rng.random()))
              for s in scores]
    order = sorted(range(len(scores)), key=lambda i: -gumbel[i])
    return set(order[:m])


def _marginals(scores, rho, logz):
    # q_i = sum_m rho[m] * P(i in Y | m); P(i|m) = exp(s_i) * Z_{m-1}(-i) / Z_m.
    k = len(scores)
    q = [0.0] * k
    for i in range(k):
        sub = [s for j, s in enumerate(scores) if j != i]
        sub_logz = _log_partitions(sub)
        acc = 0.0
        for m in range(1, k + 1):
            # Z_{m-1}(-i)
            zm1 = sub_logz[m - 1] if m - 1 <= len(sub) else -math.inf
            zm = logz[m]
            acc += rho[m] * math.exp(scores[i] + zm1 - zm) if zm > -math.inf else 0.0
        q[i] = acc
    return q


def generate(out, *, seed=20261030, ntrain=200, ndev=50, ntest=50, repeats=4, k=16):
    out = Path(out)
    if out.exists():
        raise ValueError(f"output exists: {out}")
    rng = random.Random(seed)
    allocations = {"train": ntrain, "dev": ndev, "test": ntest}
    student = {s: [] for s in allocations}
    oracle = {s: [] for s in allocations}
    for split, n in allocations.items():
        for t in range(n):
            cid = f"{split}-C-{t:05d}"
            utilities = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(k)]
            scores = [0.8 * u for u in utilities]
            load = rng.choice(["low", "medium", "high"])
            center = {"low": 1.8, "medium": k * 0.48, "high": k * 0.75}[load]
            strength = {"zero": 0.0, "mild": 0.5, "medium": 1.0, "strong": 1.8}[
                rng.choice(["zero", "mild", "medium", "strong"])]
            residual = [-strength * (m - center) ** 2 for m in range(k + 1)]
            logz = _log_partitions(scores)
            log_rho = [logz[m] + residual[m] for m in range(k + 1)]
            norm = _logsumexp(log_rho)
            rho = [math.exp(v - norm) for v in log_rho]
            q = _marginals(scores, rho, logz)
            rule = (f"Shared workload regime {load}; collective pressure "
                    f"{['zero','mild','medium','strong'][['zero','mild','medium','strong'].index('zero')]}.")
            options = [{"key": ALPHABET[i % 26],
                        "desc": f"Candidate {i:02d}, utility feature {utilities[i]}, compatibility code none"}
                       for i in range(k)]
            public = {"state": f"Selection task. {rule} Workload={load}.",
                      "question": "Select all appropriate candidates; zero selections are allowed.",
                      "options": options, "n_options": k,
                      "selection_mode": "multi", "allow_empty": True}
            oracle[split].append({
                "context_id": cid, "regime": "C", "strength": strength,
                "teacher_scores": scores, "teacher_residual": residual,
                "teacher_pairs": [], "teacher_pair_weights": [],
                "true_marginals": q, "true_count_probs": rho,
                "true_pairwise_marginals": [[0.0] * k for _ in range(k)],
                "true_set_probs": {}})
            for repeat in range(repeats):
                m = _sample_count(rng, log_rho)
                sel = _sample_set_given_count(rng, scores, m)
                student[split].append({
                    **public, "id": f"{cid}/draw-{repeat:02d}",
                    "group_id": cid, "source": "controlled_C", "lang": "en", "split": split,
                    "target": [int(i in sel) for i in range(k)],
                    "gold_letters": "".join(ALPHABET[i % 26] for i in range(k) if i in sel)})
    out.mkdir(parents=True)
    data_dir = out / "student"
    oracle_dir = out / "oracle_DO_NOT_USE_AS_MODEL_INPUT"
    data_dir.mkdir()
    oracle_dir.mkdir()
    written = {}
    for split in allocations:
        df = data_dir / f"{split}.jsonl"
        of = oracle_dir / f"{split}.jsonl"
        df.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in student[split]))
        of.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in oracle[split]))
        written[split] = {"contexts": n, "rows": len(student[split]),
                          "student_sha256": _sha(df), "oracle_sha256": _sha(of)}
    (out / "manifest.json").write_text(json.dumps({
        "status": "SYNTHETIC_ORACLE_DATA_ONLY_NOT_TRAINED_MODEL",
        "regime": "C", "seed": seed, "k": k, "repeats": repeats,
        "files": written}, indent=2) + "\n")
    print(json.dumps({"status": "ok", "k": k, "files": written}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--ntrain", type=int, default=200)
    p.add_argument("--ndev", type=int, default=50)
    p.add_argument("--ntest", type=int, default=50)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261030)
    args = p.parse_args()
    generate(args.output, seed=args.seed, ntrain=args.ntrain, ndev=args.ndev,
             ntest=args.ntest, repeats=args.repeats, k=args.k)
