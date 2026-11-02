#!/usr/bin/env python3
"""P1/P2 research gates and matched EUR-LEX QZ payload staging — NO submission.

P1: full 5k pilot pass (each arm sees 5k examples in continuation) with the
same batch and optimizer steps, instead of the 120+200-step underfit pilot.
P2: full official 55k x three seeds only if three independently planned
seed-level P0/P1 reports show count+set likelihood gains against BOTH BCE
and frozen independent learned-count, with non-inferior micro-F1.

QZ single-job time limit is respected: actual prior training.jsonl elapsed
seconds are required; if the needed steps exceed the observed safe budget,
the stage is marked BLOCKED and NO jobs are written. LoRA resume across
different QZ jobs is currently unsupported by the base trainer, so this
planner must not create fake runnable 6875-step jobs under a 3-hour limit.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics

from qz.make_eurlex_payloads import job_specs


def read_timing(paths, *, recent=50):
    values=[]
    if not paths:
        raise ValueError("real, previously completed pilot training logs are required")
    for path in paths:
        stream=Path(path)
        if not stream.is_file():
            raise ValueError(f"missing training.jsonl timing: {path}")
        series=[]
        for lineno,line in enumerate(stream.read_text().splitlines(),1):
            if not line.strip():
                continue
            row=json.loads(line)
            v=row.get("seconds")
            if type(v) in (float,int) and math.isfinite(v) and v>0:
                series.append(float(v))
        if len(series)<5:
            raise ValueError(f"{path}: require >=5 actual step timing measurements")
        values.extend(series[-recent:])
    values.sort()
    return {"seconds_p90":values[min(len(values)-1, math.ceil(.9*len(values))-1)],
            "n_steps_sampled":len(values),
            "seconds_p50":statistics.median(values),
            "timing_sources":[str(p) for p in paths]}


def pass_p2_gate(reports, *, anchor="frozen_anchor",
                 bce="bce_continue",learned="frozen_independent",
                 margin=.005, f1_tolerance=.02):
    if len(reports)<3:
        return {"passed":False,"reason":"Require 3 paired pilot seed reports; no full Test opening"}
    seed_results=[]
    common_gold=None
    for idx,r in enumerate(reports):
        if r.get("status")!="DEV_OFFLINE_COMPLETE_NO_GPU" or r.get("n_questions",0)<1000:
            return {"passed":False,"reason":f"report {idx} not a complete official DEV pilot"}
        if common_gold is None:
            common_gold=r.get("gold_sha256")
        if not common_gold or r.get("gold_sha256")!=common_gold:
            return {"passed":False,"reason":"P0 reports not evaluated on exact same DEV bytes"}
        arms=r.get("arms",{})
        if any(x not in arms for x in (anchor,bce,learned)):
            return {"passed":False,"reason":f"report {idx} missing BCE, Anchor or learned-count arm"}
        prob={a:arms[a]["proper_probability"] for a in (anchor,bce,learned)}
        f1={a:arms[a]["decoders"]["dev_tuned_marginal_threshold"]["micro_f1"]
            for a in (anchor,bce,learned)}
        gain_count=min(prob[q]["count_nll"]-prob[anchor]["count_nll"] for q in (bce,learned))
        gain_set=min(prob[q]["proper_joint_set_nll"]-
                     prob[anchor]["proper_joint_set_nll"] for q in (bce,learned))
        noninf=f1[anchor]-f1[bce]>=-f1_tolerance
        pass_this=(gain_count>margin and gain_set>margin and noninf)
        seed_results.append({"index":idx,"improvement_count_nll_min":gain_count,
                             "improvement_set_nll_min":gain_set,
                             "anchor_minus_bce_dev_micro_f1":f1[anchor]-f1[bce],
                             "passes":pass_this})
    return {"passed":all(x["passes"] for x in seed_results),
            "paired_seeds":len(reports),"seed_results":seed_results,
            "definition":"Anchor proper Count NLL and Set NLL each improve >=margin vs BOTH same-score learned count and matched BCE; microF1 >=BCE-tolerance",
            "margin":margin,"microf1_tolerance":f1_tolerance,
            "dev_only":True,"not_statistical_significance":True}


def stage(template,manifest,*,root,data,run_root,model,seed_list,
          pilot,timings,output,decision_reports=(),batch=8,
          safety=1.3,margin=.005):
    out=Path(output)
    if out.exists():
        raise ValueError("new stage output directory required")
    if not 1<=batch<=256 or safety<1 or not math.isfinite(safety):
        raise ValueError("invalid batch or safety")
    if pilot:
        if len(seed_list)<1 or len(set(seed_list))!=len(seed_list):
            raise ValueError("invalid seed list")
    elif len(set(seed_list))<3:
        raise ValueError("P2 official full plan requires at least three paired seeds")
    train_n=(manifest.get("pilot",{}).get("train") if pilot
             else manifest["splits"]["train"]["n"])
    if not isinstance(train_n,int) or train_n<batch:
        raise ValueError("invalid official train count")
    steps=math.ceil(train_n/batch)
    timing=read_timing(timings)
    limit=int(template["max_running_time_ms"])/1000
    estimated=steps*timing["seconds_p90"]*safety
    gate={"passed":True,"reason":"P1 full pilot budget only"}
    if not pilot:
        gate=pass_p2_gate(decision_reports,margin=margin)
    result={"status":"STAGED_NOT_SUBMITTED","pilot":pilot,"train_rows":train_n,
            "batch":batch,"steps_per_arm":steps,"seed_list":list(seed_list),
            "single_job_limit_seconds":limit,
            "estimated_upper_seconds_per_job":estimated,
            "timing":timing,"p2_gate":gate,"jobs":[]}
    if not gate["passed"]:
        result["status"]="BLOCKED_ON_P0_P1_EVIDENCE"
    elif estimated>limit:
        result["status"]="BLOCKED_ON_QZ_TIME_LIMIT"
        result["reason"]=(
          "Full LoRA continuation needs more than the QZ single-job time limit; "
          "do not submit a job that will be killed. Use a confirmed resumable "
          "LoRA/optimizer checkpoint implementation or a faster measured cluster spec.")
    else:
        payloads=job_specs(template,manifest,repo_root=Path(root),
            data_root=Path(data),output_root=Path(run_root),
            model_root=Path(model),seeds=tuple(seed_list),
            pilot=pilot,warm_steps=steps,continuation_steps=steps,
            batch=batch,lr=2e-5,max_input=manifest["max_input_tokens"],
            max_batch=manifest["max_input_tokens"])
        result["jobs"]=[p["name"] for p in payloads]
    out.mkdir(parents=True)
    if result["jobs"]:
        for payload in payloads:
            (out/(payload["name"]+".json")).write_text(
                json.dumps(payload,indent=2)+"\n")
    (out/"plan_status.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template",type=Path,default=Path("qz_c64/c64-warm-bce.json"))
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["p1-full-5k","p2-55k"],required=True)
    p.add_argument("--seed",type=int,nargs="+",default=[43,44,45])
    p.add_argument("--observed-training-log",type=Path,action="append",required=True)
    p.add_argument("--p0-report",type=Path,action="append",default=[])
    p.add_argument("--batch",type=int,default=8)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("new output directory required")
    tpl=json.loads(args.template.read_text())
    manifest=json.loads(args.manifest.read_text())
    reports=[json.loads(path.read_text()) for path in args.p0_report]
    result=stage(tpl,manifest,root=args.root,data=args.data,
        run_root=args.run_root,model=args.model,seed_list=args.seed,
        pilot=args.stage=="p1-full-5k",
        timings=args.observed_training_log,output=args.output,
        decision_reports=reports,batch=args.batch)
    print(json.dumps({"status":result["status"],"steps":result["steps_per_arm"],
                      "estimated_upper_seconds_per_job":
                          result["estimated_upper_seconds_per_job"],
                      "number_of_submittable_jobs":len(result["jobs"]),
                      "output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
