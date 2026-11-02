#!/usr/bin/env python3
"""Safely re-decode saved DEV predictions with the exact structured set law.

No Qwen weights or GPUs are required. This fixes historical Set-MAP index
errors without changing s_i, count probabilities, marginals, or Gold-Set NLL.

Full law:
  P(S|x) = rho_|S| exp(sum_{i in S} s_i) / Z_|S|(s)
Whether rho came from independent c_m or anchored a_m+logZ_m is immaterial.
Set-MAP uses log(rho_m) + top-m sum - logZ_m, not Count Mode + Top-k.
Use the saved *log* count probabilities to avoid underflow and never use
the correct answer count to select m. Only DEV is accepted in this script.

Usage:
  python -m paper_static.redecode_set_map \
    --gold data/eurlex_decision_v1/pilot/dev.jsonl \
    --predictions runs/paper1_eurlex_pilot_dev_eval/set_ce_independent/seed_43/predictions.jsonl \
    --output runs/paper1_eurlex_pilot_dev_eval_fixed/set_ce_independent/seed_43

Run separately for each previously saved structured DEV checkpoint and seed.
Always use a fresh --output directory, never replace the original predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from harness.multiselect_metrics import prediction_records, compute_metrics, read_jsonl, read_dataset
from harness.set_distillation import allowed_counts, input_digest, log_partitions


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1<<20),b""):
            digest.update(part)
    return digest.hexdigest()


def decode(scores, log_count_probs, allowed):
    """Return exact MAP indices and log probability given public count support."""
    if not isinstance(scores,list) or not scores or any(
        type(v) not in (float,int) or not math.isfinite(v) for v in scores
    ):
        raise ValueError("finite raw candidate_logits are mandatory")
    k=len(scores)
    support=sorted(set(allowed))
    if not support or any(type(m) is not int or not 0<=m<=k for m in support):
        raise ValueError("invalid public count support")
    if not isinstance(log_count_probs,list) or len(log_count_probs)!=k+1:
        raise ValueError("requires stable K+1 count_log_probs, not rounded count_probs")
    if any(type(log_count_probs[m]) not in (float,int) or
           not math.isfinite(log_count_probs[m]) for m in support):
        raise ValueError("nonfinite log probability on public count support")
    if any(log_count_probs[m] is not None for m in range(k+1) if m not in support):
        raise ValueError("forbidden count log probabilities must be None")
    center=max(scores)
    centered=[s-center for s in scores]
    logz=log_partitions(centered)
    order=sorted(range(k),key=lambda i:(-centered[i],i))
    prefix=[0.0]
    for index in order:
        prefix.append(prefix[-1]+centered[index])
    m=max(support,key=lambda m:(log_count_probs[m]+prefix[m]-logz[m],-m))
    selected=sorted(order[:m])
    return selected, min(0.0,log_count_probs[m]+prefix[m]-logz[m])


def redecode(rows,predictions):
    """Retain every original probability and model identity, correcting sets only."""
    if not rows or any(row.get("split")!="dev" for row in rows):
        raise ValueError("DEV-only: refusing to inspect or re-decode TEST")
    indexed={}
    for item in predictions:
        ident=item.get("id")
        if not isinstance(ident,str) or ident in indexed:
            raise ValueError("invalid or duplicated saved prediction id")
        indexed[ident]=item
    if set(indexed)!={row["id"] for row in rows}:
        raise ValueError("prediction IDs do not match DEV gold dataset")
    corrected=[]
    changed=0
    for row in rows:
        item=indexed[row["id"]]
        ident=row["id"]
        k=len(row["options"])
        keys=[o["key"] for o in row["options"]]
        if item.get("decoder")!="set_map":
            raise ValueError(f"{ident}: model is not structured set_map")
        if item.get("option_keys")!=keys or item.get("input_digest")!=input_digest(row):
            raise ValueError(f"{ident}: option order or prompt digest mismatch")
        logits=item.get("candidate_logits")
        if not isinstance(logits,list) or len(logits)!=k:
            raise ValueError(f"{ident}: missing raw candidate logits")
        if not isinstance(item.get("count_probs"),list) or len(item["count_probs"])!=k+1:
            raise ValueError(f"{ident}: missing count probabilities")
        new_set,new_log_probability=decode(logits,item.get("count_log_probs"),
                                           allowed_counts(row))
        old=item.get("predicted_indices")
        if not isinstance(old,list):
            raise ValueError(f"{ident}: missing old MAP indices")
        if sorted(old)!=new_set:
            changed+=1
        updated=dict(item)
        updated["predicted_indices"]=new_set
        updated["predicted_set_log_probability"]=new_log_probability
        updated["decoder"]="set_map"
        corrected.append(updated)
    # The standard evaluator checks ALL predicted marginals, Count NLL and
    # MAP consistency; never silently accept missing or incoherent checkpoint data.
    prediction_records(rows,corrected,threshold=.5,decoder="auto")
    return corrected,{"changed_questions":changed,"n_questions":len(rows),
                       "changed_fraction":changed/len(rows)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold",type=Path,required=True,help="DEV JSONL only")
    p.add_argument("--predictions",type=Path,required=True,
                   help="saved original structured predictions.jsonl")
    p.add_argument("--output",type=Path,required=True,
                   help="NEW directory, never overwrite original results")
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("output exists: refusing to overwrite prior inference")
    rows=read_dataset(args.gold,expected_split="dev")
    original=read_jsonl(args.predictions)
    corrected,change=redecode(rows,original)
    report=compute_metrics(rows,corrected,threshold=.5,decoder="auto")
    summary={
        "status":"DEV_REDECODED_FROM_SAVED_PREDICTIONS_NO_GPU",
        "gold_sha256":_sha256(args.gold),
        "source_predictions_sha256":_sha256(args.predictions),
        "source_path":str(args.predictions),
        "decoder":"exact_set_map_by_log_rho_plus_top_sum_minus_log_z",
        "no_model_forward_or_parameter_updates":True,
        **change,
    }
    # Full 100-candidate EUR-LEX diagnostic is optional and uses the same
    # already validated model marginals and corrected predicted sets.
    eurlex_report=None
    if all(row.get("source")=="lexglue_eurlex" and len(row["options"])==100
           for row in rows):
        from .eurlex_metrics import score
        eurlex_report=score(rows,corrected,threshold=.5)
    args.output.mkdir(parents=True)
    with (args.output/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for item in corrected:
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
    for name,object_ in (("metrics.json",report),
                          ("redecode_manifest.json",summary),
                          ("eurlex_paper_metrics.json",eurlex_report)):
        if object_ is None:
            continue
        (args.output/name).write_text(
            json.dumps(object_,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
    print(json.dumps({**summary,"question_em":report["question_em"],
                      "f1":report["f1"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
