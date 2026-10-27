#!/usr/bin/env python3
"""Audit and aggregate equal-step Static SetJev result manifests (no GPU).

Rejects wrong arm flags, wrong model path, different train/dev bytes, unequal
update counts, and missing selected DEV metrics; never treats empty runs as data.
Example: python paper_static/collect_results.py --manifest paper_static/runs_4b.jsonl --output paper_static/report_4b.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
try:
    from .plan_experiments import sha256_file
except ImportError:
    from plan_experiments import sha256_file

METRICS = ("question_em","f1","bce","membership_brier","count_nll",
           "count_brier","count_accuracy","count_mae","joint_set_nll",
           "protocol_violation_rate")

def audit_run(spec):
    root=Path(spec["output"])
    if not (root/"training_complete.json").exists():
        return {"id":spec["id"],"status":"NOT_RUN",
                "arm":spec["arm"],"seed":spec["seed"],"model_tag":spec["model_tag"]}
    completed=json.loads((root/"training_complete.json").read_text())
    config=json.loads((root/"training_config.json").read_text())
    if int(completed["steps"])!=int(spec["update_budget"]):
        raise ValueError(f"{spec['id']} used different update count")
    selected=completed["selected"]
    if int(selected["step"])!=int(spec["update_budget"]):
        raise ValueError(f"{spec['id']} selected non-terminal checkpoint")
    for key in ("count_anchor","count_anchor_detach"):
        if bool(config.get(key,False))!=bool(spec[key]):
            raise ValueError(f"{spec['id']} configuration {key} mismatch")
    if config["objective"]!=spec["objective"] or config["seed"]!=spec["seed"]:
        raise ValueError(f"{spec['id']} objective/seed mismatch")
    if Path(config["model"])!=Path(spec["model"]):
        raise ValueError(f"{spec['id']} model/checkpoint mismatch")
    for kind,key in (("train","train_sha256"),("dev","dev_sha256"),
                     ("modules","module_sha256")):
        expected=spec.get(key)
        recorded=config.get("modules_sha256" if kind=="modules" else key)
        if expected is not None and recorded!=expected:
            raise ValueError(f"{spec['id']} {kind} SHA mismatch")
    valid=root/f"validation_{selected['step']:06d}.json"
    if not valid.is_file():
        raise FileNotFoundError(f"{spec['id']} missing selected validation: {valid}")
    measurements=json.loads(valid.read_text())
    if abs(float(measurements["question_em"])-float(selected["question_em"]))>1e-9:
        raise ValueError(f"{spec['id']} selected metric disagreement")
    return {
        "id":spec["id"],"arm":spec["arm"],"seed":spec["seed"],
        "model_tag":spec["model_tag"],"step":selected["step"],
        "status":"COMPLETE","metrics":{k:measurements.get(k) for k in METRICS},
        "train_sha256":config["train_sha256"],"dev_sha256":config["dev_sha256"],
        "checkpoint":str(root/selected["checkpoint"]),
    }

def summary(manifest_rows, *, require_all=False):
    rows=[audit_run(spec) for spec in manifest_rows]
    pending=[r["id"] for r in rows if r["status"]!="COMPLETE"]
    if require_all and pending:
        raise ValueError(f"missing {len(pending)} GPU run(s): {pending[:5]}")
    seen={(r["model_tag"],r["seed"],r["arm"]) for r in rows}
    if len(seen)!=len(rows):
        raise ValueError("duplicate model/seed/arm in manifest")
    successful=[r for r in rows if r["status"]=="COMPLETE"]
    bykey={(r["model_tag"],r["seed"],r["arm"]):r for r in successful}
    curves=[];arms=defaultdict(lambda:defaultdict(list))
    for r in successful:
        for metric,value in r["metrics"].items():
            if value is not None:
                arms[(r["model_tag"],r["arm"])][metric].append(float(value))
    for (model,arm),measurements in sorted(arms.items()):
        curves.append({"model":model,"arm":arm,"seed_count":len(
            {r["seed"] for r in successful if r["model_tag"]==model and r["arm"]==arm}),
            "metrics":{m:{"mean":mean(v),"std":stdev(v) if len(v)>1 else None,
                             "n":len(v)} for m,v in measurements.items()}})
    pairs=[("set_ce_anchor","set_ce_independent"),
           ("set_ce_anchor","set_ce_anchor_detach"),
           ("induced_anchor","induced_independent"),
           ("induced_anchor","induced_anchor_detach"),
           ("set_ce_anchor","bce_continue"),
           ("induced_anchor","bce_continue")]
    diffs=[]
    models=sorted({r["model_tag"] for r in successful})
    for model in models:
        seeds=sorted({r["seed"] for r in successful if r["model_tag"]==model})
        for a,b in pairs:
            for metric in ("question_em","count_nll","membership_brier","joint_set_nll"):
                vals={}
                for seed in seeds:
                    left=bykey.get((model,seed,a));right=bykey.get((model,seed,b))
                    if left and right:
                        x=left["metrics"].get(metric);y=right["metrics"].get(metric)
                        if x is not None and y is not None:
                            vals[str(seed)]=float(x)-float(y)
                if vals:
                    ds=list(vals.values())
                    diffs.append({"model":model,"left":a,"right":b,"metric":metric,
                                  "paired_by_seed":vals,"n_pairs":len(ds),
                                  "mean_delta":mean(ds),
                                  "sample_std":stdev(ds) if len(ds)>1 else None})
    return {
        "status":"INCOMPLETE" if pending else "COMPLETE",
        "completed":len(successful),"planned":len(rows),"missing":pending,
        "rows":successful,"arm_summaries":curves,"paired":diffs,
        "note":"Aggregate DEV results only; no Test/paired-question bootstrap implied."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--require-all",action="store_true")
    args=p.parse_args()
    if args.output.exists():p.error(f"refusing to overwrite {args.output}")
    manifest=[json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    result=summary(manifest,require_all=args.require_all)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.output),"status":result["status"],
                      "completed":result["completed"],"planned":result["planned"]}))

if __name__=="__main__":main()
