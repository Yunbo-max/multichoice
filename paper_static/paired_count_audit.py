#!/usr/bin/env python3
"""DEV-only, label-stratified, paired BCE vs anchored SetJev diagnosis.

Consumes the unmodified ``predictions.jsonl`` emitted by
``harness/eval_multiselect.py``.  No model weights, GPU, or extra training.

Buckets use BCE's gold-count NLL, not the anchor prediction. This is a
*retrospective diagnostic*, never a deployable gating rule or a performance
prediction from unlabeled inputs. The benchmark has already been opened on
TEST; do not use TEST to develop or select a method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return rows


def keyed(rows, label):
    ans = {}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in ans:
            raise ValueError(f"{label}: missing or duplicate id {ident!r}")
        ans[ident] = row
    if not ans:
        raise ValueError(f"{label}: empty")
    return ans


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def logsum(values):
    acc = -math.inf
    for v in values:
        acc = logadd(acc, v)
    return acc


def log_count_distribution(scores, allowed):
    """Exact log Poisson-binomial counts from RAW logits, conditioned on protocol.

    O(K^2) log-domain elementary-symmetric-polynomial DP. Crucially, it
    preserves the common logit shift: no sigmoid(large-logit) rounding.
    """
    if not scores or any(type(s) not in (float, int) or not math.isfinite(s) for s in scores):
        raise ValueError("finite raw BCE logits are required")
    k = len(scores)
    support = set(allowed)
    if not support or any(type(m) is not int or not 0 <= m <= k for m in support):
        raise ValueError("invalid public count support")
    z = [0.0]
    for score in scores:
        z = [z[0]] + [logadd(z[m], score + z[m-1]) for m in range(1, len(z))] + [score + z[-1]]
    normalizer = logsum(z[m] for m in sorted(support))
    return [z[m] - normalizer if m in support else -math.inf for m in range(k+1)]


def allowed_counts(row):
    mode = row.get("selection_mode", "multi")
    empty = row.get("allow_empty", False)
    if type(empty) is not bool or mode not in ("single", "multi"):
        raise ValueError(f"{row.get('id')}: invalid selection protocol")
    k = len(row["options"])
    if mode == "single":
        if empty:
            raise ValueError("single cannot allow empty")
        return [1]
    return list(range(0 if empty else 1, k + 1))


def input_digest(row):
    """Matches harness/set_distillation.py input_digest; never hash gold label."""
    payload = {key: row[key] for key in ("state", "question", "options")}
    payload.update(selection_mode=row.get("selection_mode", "multi"),
                   allow_empty=row.get("allow_empty", False))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _predicted_indices(pred, k, *, structured):
    if structured:
        if pred.get("decoder") != "set_map":
            raise ValueError("anchor must be an exported structured set_map prediction")
        selected = pred.get("predicted_indices")
    else:
        if pred.get("decoder") == "set_map":
            raise ValueError("BCE baseline cannot carry a structured set_map decoder")
        selected = pred.get("predicted_indices")
    if (not isinstance(selected, list) or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < k for i in selected)):
        raise ValueError("missing or invalid predicted_indices; use eval_multiselect exports")
    return set(selected)


def _scores(pred, k):
    s = pred.get("candidate_logits", pred.get("logits"))
    if (not isinstance(s, list) or len(s) != k or any(type(v) not in (int,float)
        or not math.isfinite(v) for v in s)):
        raise ValueError("BCE prediction missing finite raw logits; probabilities are insufficient")
    return s


def _validate_prediction(pred, row, arm, seed, structured):
    ident = row["id"]
    k = len(row["options"])
    keys = [option["key"] for option in row["options"]]
    if pred.get("option_keys") != keys or pred.get("input_digest") != input_digest(row):
        raise ValueError(f"{ident}: {arm} option identity/prompt digest mismatch")
    if pred.get("checkpoint_id") != f"paper1|{arm}|seed_{seed}":
        raise ValueError(f"{ident}: {arm} checkpoint identity or seed mismatch")
    probs = pred.get("p")
    if (not isinstance(probs,list) or len(probs)!=k or any(type(p) not in (int,float)
        or not math.isfinite(p) or not 0<=p<=1 for p in probs)):
        raise ValueError(f"{ident}: invalid membership marginal probabilities")
    indices = _predicted_indices(pred, k, structured=structured)
    return indices


def zero_residual_map(scores, allowed):
    """MAP of identical candidate scores with a_m=0 under the public protocol."""
    order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    running=[0.0]
    for i in order:
        running.append(running[-1]+scores[i])
    m=max(sorted(allowed),key=lambda m:(running[m],-m))
    return set(order[:m])


def paired_rows(gold, bce, anchor, seed):
    g = keyed(gold, "gold")
    b = keyed(bce, "BCE")
    a = keyed(anchor, "Anchor")
    if set(b) != set(g) or set(a) != set(g):
        raise ValueError("BCE/Anchor/gold ID sets differ; no partial cohort comparisons")
    out = []
    for ident in sorted(g):
        row = g[ident]
        if row.get("split") != "dev":
            raise ValueError("DEV-only audit: wrong split; TEST is locked")
        target = row["target"]
        k = len(row["options"])
        if (k < 2 or len(target)!=k or any(type(y) is not int or y not in (0,1) for y in target)
                or sum(target) not in allowed_counts(row)):
            raise ValueError(f"{ident}: invalid gold multi-hot/protocol")
        bs = _validate_prediction(b[ident], row, "bce_continue", seed, False)
        an = _validate_prediction(a[ident], row, "set_ce_anchor", seed, True)
        raw = _scores(b[ident], k)
        log_b = log_count_distribution(raw, allowed_counts(row))
        counts = a[ident].get("count_probs")
        if (not isinstance(counts,list) or len(counts)!=k+1
                or any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in counts)
                or abs(sum(counts)-1)>1e-6
                or any(counts[m]>1e-6 for m in range(k+1) if m not in allowed_counts(row))):
            raise ValueError(f"{ident}: missing or invalid anchor count distribution")
        log_a = a[ident].get("count_log_probs")
        if log_a is None:
            raise ValueError(f"{ident}: stable count_log_probs required, not clipped probabilities")
        if len(log_a)!=k+1 or any(
            ((v is not None) if m not in allowed_counts(row) else
             (type(v) not in (int,float) or not math.isfinite(v) or abs(math.exp(v)-counts[m])>1e-6))
            for m,v in enumerate(log_a)):
            raise ValueError(f"{ident}: inconsistent anchor log-count distribution")
        if row.get("selection_mode","multi")=="single":
            # Native BCE evaluator uses argmax, never 0.5 threshold in single mode.
            if bs != {max(range(k), key=lambda i:(raw[i],-i))}:
                raise ValueError(f"{ident}: BCE single-choice decoder mismatch")
        else:
            expected={i for i,p in enumerate(b[ident]["p"]) if p>=0.5}
            if bs!=expected:
                raise ValueError(f"{ident}: BCE threshold must be fixed 0.5")
        actual = {i for i,y in enumerate(target) if y}
        positives = [raw[i] for i in actual]
        negatives = [raw[i] for i in range(k) if i not in actual]
        strict_rank = (not positives or not negatives or min(positives)>max(negatives))
        m = len(actual)
        anc_scores=_scores(a[ident],k)
        zero_log=log_count_distribution(anc_scores,allowed_counts(row))
        zero_pred=zero_residual_map(anc_scores,allowed_counts(row))
        raw_head=a[ident].get("count_residual_logits")
        if (not isinstance(raw_head,list) or len(raw_head)!=k+1 or any(
                type(v) not in (int,float) or not math.isfinite(v) for v in raw_head)):
            raise ValueError(f"{ident}: anchor residual logits missing; re-run patched evaluator")
        residual_span=max(raw_head[z] for z in allowed_counts(row))-min(
            raw_head[z] for z in allowed_counts(row))
        # Source-scoped passage groups; no bootstrap of a MultiRC passage as independent questions.
        group_id=row.get("group_id") or ident
        group = f"{row.get('source','')}/{group_id}"
        out.append({
            "id":ident,"seed":seed,"group":group,"source":row.get("source"),
            "k":k,"gold_count":m,"gold":sorted(actual),"bce":sorted(bs),"anchor":sorted(an),
            "bce_count_nll":-log_b[m],"anchor_count_nll":-log_a[m],
            "bce_em":int(bs==actual),"anchor_em":int(an==actual),
            "bce_count_mae":abs(len(bs)-m),"anchor_count_mae":abs(len(an)-m),
            "strict_rank":int(strict_rank),"anchor_residual_span":residual_span,
            "anchor_zero_em":int(zero_pred==actual),
            "anchor_zero_count_nll":-zero_log[m],
        })
    return out


def _mean(items, key):
    return sum(item[key] for item in items)/len(items)


def summarize(items):
    if not items:
        raise ValueError("empty analysis group")
    return {"n":len(items),"groups":len({x["group"] for x in items}),
            "bce_em":_mean(items,"bce_em"),"anchor_em":_mean(items,"anchor_em"),
            "delta_em":_mean(items,"anchor_em")-_mean(items,"bce_em"),
            "bce_count_nll":_mean(items,"bce_count_nll"),
            "anchor_count_nll":_mean(items,"anchor_count_nll"),
            "delta_count_nll":_mean(items,"anchor_count_nll")-_mean(items,"bce_count_nll"),
            "bce_count_mae":_mean(items,"bce_count_mae"),
            "anchor_count_mae":_mean(items,"anchor_count_mae"),
            "strict_rank_fraction":_mean(items,"strict_rank"),
            "anchor_residual_span":_mean(items,"anchor_residual_span"),
            "anchor_zero_em":_mean(items,"anchor_zero_em"),
            "anchor_minus_zero_em":_mean(items,"anchor_em")-_mean(items,"anchor_zero_em"),
            "anchor_zero_count_nll":_mean(items,"anchor_zero_count_nll"),
            "anchor_minus_zero_count_nll":_mean(items,"anchor_count_nll")-_mean(items,"anchor_zero_count_nll")}


def quartile_buckets(items):
    """Fixed rank-quantiles of BCE gold-count NLL only; deterministic ties."""
    if len(items)<4:
        raise ValueError("quartiles need >=4 DEV examples")
    sorted_rows=sorted(items,key=lambda x:(x["bce_count_nll"],x["id"]))
    groups={f"Q{i+1}":[] for i in range(4)}
    for rank,row in enumerate(sorted_rows):
        groups[f"Q{min(4,4*rank//len(sorted_rows)+1)}"].append(row)
    return groups


def bootstrap_ci(items, key, iterations=1000, seed=0):
    """Paired cluster bootstrap percentile interval on the same questions."""
    if iterations<1:
        raise ValueError("bootstrap iterations must be positive")
    clusters=defaultdict(list)
    for row in items:
        clusters[row["group"]].append(row)
    clusters=list(clusters.values())
    rng=random.Random(seed)
    vals=[]
    for _ in range(iterations):
        sampled=[row for _ in clusters for row in clusters[rng.randrange(len(clusters))]]
        vals.append(summarize(sampled)[key])
    vals.sort()
    low=vals[max(0,int(0.025*iterations)-1)]
    high=vals[min(iterations-1,math.ceil(0.975*iterations)-1)]
    return [low,high]


def analyze_seed(rows, iters, bootstrap_seed):
    result={"overall":summarize(rows),"quartiles":{},"rank_separated":{}}
    result["overall"]["delta_em_cluster_bootstrap_ci95"]=bootstrap_ci(rows,"delta_em",iters,bootstrap_seed)
    for q, subset in quartile_buckets(rows).items():
        stat=summarize(subset)
        stat["nll_range"]=[min(x["bce_count_nll"] for x in subset),
                           max(x["bce_count_nll"] for x in subset)]
        stat["delta_em_cluster_bootstrap_ci95"]=bootstrap_ci(subset,"delta_em",iters,bootstrap_seed+int(q[1:]))
        result["quartiles"][q]=stat
    for flag in (0,1):
        subset=[r for r in rows if r["strict_rank"]==flag]
        if subset:
            result["rank_separated"][str(flag)]=summarize(subset)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold",type=Path,required=True,help="DEV JSONL, never TEST")
    ap.add_argument("--predictions-root",type=Path,required=True,
                    help="root/seed_43/{bce_continue,set_ce_anchor}/predictions.jsonl")
    ap.add_argument("--output",type=Path,required=True,help="new JSON summary")
    ap.add_argument("--per-question",type=Path,help="optional new JSONL; diagnostic not deployable")
    ap.add_argument("--seeds",type=int,nargs="+",default=[43,44,45])
    ap.add_argument("--bootstrap",type=int,default=500)
    args=ap.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:
        ap.error("duplicate or missing seeds")
    if args.bootstrap<1:
        ap.error("bootstrap must be positive")
    if args.output.exists() or (args.per_question and args.per_question.exists()):
        ap.error("refusing to overwrite results")
    gold=read_jsonl(args.gold)
    if any(row.get("split")!="dev" for row in gold):
        ap.error("this audit is DEV-only; no TEST-based method selection")
    output={"status":"COMPLETE","data_role":"DEV_DIAGNOSTIC_ONLY",
            "oracle_gold_count_used_for_stratification":True,
            "strata_not_deployable":True,"zero_residual_ablation":"frozen anchor candidate scores, a_m=0, no retraining",
            "count_nll_baseline":"BCE Poisson-binomial conditioned on public count support",
            "gold_sha256":sha256_file(args.gold),"seeds":{},"cohort_frozen_by":"BCE per-item gold-count NLL"}
    all_rows=[]
    for seed in args.seeds:
        base=args.predictions_root/f"seed_{seed}"
        bp=base/"bce_continue"/"predictions.jsonl"
        apath=base/"set_ce_anchor"/"predictions.jsonl"
        pairs=paired_rows(gold,read_jsonl(bp),read_jsonl(apath),seed)
        output["seeds"][str(seed)]={"bce_sha256":sha256_file(bp),
                                     "anchor_sha256":sha256_file(apath),
                                     **analyze_seed(pairs,args.bootstrap,seed+20261030)}
        all_rows.extend(pairs)
    # Report means over SEEDS (not pseudoreplicated pooled questions).
    output["seed_mean_delta_em"]=sum(z["overall"]["delta_em"] for z in output["seeds"].values())/len(args.seeds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if args.per_question:
        args.per_question.parent.mkdir(parents=True,exist_ok=True)
        with args.per_question.open("w",encoding="utf-8") as stream:
            for row in all_rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":"COMPLETE","seeds":list(output["seeds"]),
                      "seed_mean_delta_em":output["seed_mean_delta_em"],
                      "output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
