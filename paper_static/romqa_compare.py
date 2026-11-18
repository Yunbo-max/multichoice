#!/usr/bin/env python3
"""Paired RoMQA DEV-only comparison: BCE, Anchor, TRAIN-prior Independent Count.

Read report.json + per_question.jsonl from romqa_metrics.py. Require exactly
the same derived source dataset, Gold sidecar, IDs and question clusters.
Bootstrap resamples question CLUSTERS (not options or individual answers).
No blind TEST, no optimistic selected-test analysis, no oracle Gold counts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

METRICS={"count_nll":-1,"joint_set_nll":-1,"exact_match":1,"set_f1":1}
ROLES=("bce","anchor","prior")


def _quantile(values,p):
    values=sorted(values)
    point=(len(values)-1)*p
    a,b=math.floor(point),math.ceil(point)
    return values[a] if a==b else values[a]*(b-point)+values[b]*(point-a)


def _read_run(directory):
    directory=Path(directory)
    report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
    if report.get("status")!="ROMQA_DEV_OFFLINE_SCORED" or report.get("test_read") is not False:
        raise ValueError("only completed RoMQA DEV scoring reports allowed")
    rows=[]
    with (directory/"per_question.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    ids=[r["id"] for r in rows]
    if not rows or len(ids)!=len(set(ids)) or len(rows)!=report["n"]:
        raise ValueError("duplicate, missing or unpaired per-question predictions")
    return report,{row["id"]:row for row in rows}


def paired(results, *, bootstrap=1000, seed=43):
    if set(results)!=set(ROLES):
        raise ValueError("require BCE, Anchor and TRAIN-prior independent roles")
    bce_report,bce=results["bce"]
    base_id=bce_report.get("frozen_bce_checkpoint")
    if base_id is not None:
        # For the actual frozen controls, comparing an Anchor head trained
        # on a different BCE base is a hard scientific protocol violation.
        expected={
            "bce":"|frozen|bce|",
            "anchor":"|frozen|anchor|zero|",
            "prior":"|frozen|independent|prior|",
        }
        for name,(report,_) in results.items():
            if (report.get("frozen_bce_checkpoint")!=base_id or
                expected[name] not in (report.get("checkpoint_id") or "")):
                raise ValueError(f"{name}: not the same frozen BCE or missing TRAIN-prior control")
    others={}
    for name in ("anchor","prior"):
        meta,rows=results[name]
        for key in ("source_data_sha256","sidecar_sha256","n","n_clusters"):
            if meta.get(key)!=bce_report.get(key):
                raise ValueError(f"{name}: incompatible matched DEV data/sidecar {key}")
        if set(rows)!=set(bce):
            raise ValueError(f"{name}: prediction ID mismatch")
        for ident,r in bce.items():
            if rows[ident].get("cluster_id")!=r.get("cluster_id"):
                raise ValueError(f"{name}: question cluster mismatch {ident}")
            for f in ("gold_count","complete_answer_available","candidate_complete"):
                if rows[ident].get(f)!=r.get(f):
                    raise ValueError(f"{name}: Gold/coverage mismatch {ident}")
        others[name]=rows
    groups=defaultdict(list)
    for ident,r in bce.items():groups[r["cluster_id"]].append(ident)
    names=sorted(groups)
    if len(names)<2:
        raise ValueError("need >=2 independent question clusters")
    rng=random.Random(seed)
    draws=[[names[rng.randrange(len(names))] for _ in names]
           for _ in range(bootstrap)]
    def cluster_worst(rows,samples):
        return statistics.mean(min(rows[i]["set_f1"] for i in groups[g])
                               for g in samples)
    def mean(rows,samples,metric):
        values=[rows[i][metric] for group in samples for i in groups[group]]
        return statistics.mean(values)
    def delta(rows_a,rows_b,samples,metric):
        direction=METRICS[metric]
        return direction*(mean(rows_a,samples,metric)-mean(rows_b,samples,metric))
    summary={
        "status":"ROMQA_DEV_PAIRED_UNTUNED_REPORT",
        "source_data_sha256":bce_report.get("source_data_sha256"),
        "sidecar_sha256":bce_report.get("sidecar_sha256"),
        "n":len(bce),"clusters":len(groups),
        "bootstrap_cluster_resamples":bootstrap,
        "test_read":False,
        "comparisons":{},
    }
    # Positive delta always means first method improved (lower NLL/higher F1).
    pairs=[
        ("anchor_vs_bce",others["anchor"],bce),
        ("anchor_vs_prior",others["anchor"],others["prior"]),
        ("prior_vs_bce",others["prior"],bce),
    ]
    for label,a,b in pairs:
        metrics={}
        for metric in METRICS:
            point=delta(a,b,names,metric)
            sampled=[delta(a,b,d,metric) for d in draws]
            metrics[metric]={
                "improvement":point,
                "ci95":([_quantile(sampled,.025),_quantile(sampled,.975)]
                        if sampled else None),
            }
        point=cluster_worst(a,names)-cluster_worst(b,names)
        sampled=[cluster_worst(a,d)-cluster_worst(b,d) for d in draws]
        metrics["cluster_min_f1"]={
            "improvement":point,
            "ci95":([_quantile(sampled,.025),_quantile(sampled,.975)]
                    if sampled else None),
        }
        summary["comparisons"][label]=metrics
    count=summary["comparisons"]
    summary["interpretation"]={
        "prior_beats_bce_count_nll":
            count["prior_vs_bce"]["count_nll"]["improvement"]>0,
        "anchor_beats_bce_count_nll":
            count["anchor_vs_bce"]["count_nll"]["improvement"]>0,
        "anchor_beats_prior_count_nll":
            count["anchor_vs_prior"]["count_nll"]["improvement"]>0,
        "anchor_beats_both_proper_nll":
            all(count[k][m]["improvement"]>0
                for k in ("anchor_vs_bce","anchor_vs_prior")
                for m in ("count_nll","joint_set_nll")),
        "warning":"DEV-only descriptive comparisons: three seeds and true held-out evaluation still required. No outcome is presumed."
    }
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ROLES:
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--seed",type=int,default=43)
    args=p.parse_args(argv)
    if args.output.exists() or args.bootstrap<0:
        p.error("output already exists or invalid bootstrap")
    runs={name:_read_run(getattr(args,name)) for name in ROLES}
    result=paired(runs,bootstrap=args.bootstrap,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"comparisons":result["comparisons"],
                      "test_read":False},indent=2))


if __name__=="__main__":
    main()
