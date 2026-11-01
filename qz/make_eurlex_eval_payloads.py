#!/usr/bin/env python3
"""Stage, do not submit, single-GPU EUR-LEX DEV inference plus fair-score jobs.

One job per trained warm/continued checkpoint. NEVER accesses official TEST.
Requires the same QZ GPU template as the training pilot. Exports full raw
candidate logits and structured count probabilities for proper set NLL.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
from .make_eurlex_payloads import ARMS


def jobs(template,repo_root,data_root,run_root,eval_root, *,
         seeds=(43,),pilot=True,max_input=8192,max_batch=8192):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("seeds missing/repeated")
    root, data, runs, out = map(Path,(repo_root,data_root,run_root,eval_root))
    if any(not p.is_absolute() for p in (root,data,runs,out)):
        raise ValueError("absolute QZ paths required")
    mode="pilot" if pilot else "main"
    dev=data / ("pilot/dev.jsonl" if pilot else "dev.jsonl")
    names=[("warm_bce",44,runs/"warm_bce/final")]
    for seed in seeds:
        for arm in ARMS:
            names.append((arm,seed,runs/arm/f"seed_{seed}"/"final"))
    result=[]
    for arm,seed,model in names:
        name=f"eurlex-{mode}-eval-{arm}-s{seed}"
        output=out/arm/f"seed_{seed}"
        checkpoints=f"paper1|eurlex|{mode}|{arm}|seed_{seed}"
        py=root/".venv-sii/bin/python"
        cmd=[str(py),"-m","harness.eval_multiselect",
             "--input",str(dev),"--model",str(model),
             "--expected-split","dev","--threshold","0.5","--decoder","auto",
             "--output",str(output),"--checkpoint-id",checkpoints,
             "--batch-size","1","--max-input-tokens",str(max_input),
             "--max-batch-tokens",str(max_batch)]
        metrics=[str(py),"-m","paper_static.eurlex_metrics",
                 "--gold",str(dev),"--predictions",str(output/"predictions.jsonl"),
                 "--output",str(output/"eurlex_paper_metrics.json"),
                 "--split","dev","--threshold","0.5"]
        log=root/"logs/qz/eurlex"/(name+".log")
        script=[
           "set -euo pipefail",
           "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1",
           "export TOKENIZERS_PARALLELISM=false",
           "export PYTHONPATH="+shlex.quote(str(root)),
           "test -s "+shlex.quote(str(dev)),
           "test -s "+shlex.quote(str(model/"config.json")),
           "test ! -e "+shlex.quote(str(output)),
           "mkdir -p "+shlex.quote(str(log.parent)),
           "cd "+shlex.quote(str(root)),
           shlex.join(cmd)+" > "+shlex.quote(str(log))+" 2>&1",
           shlex.join(metrics)+" >> "+shlex.quote(str(log))+" 2>&1"
        ]
        if any(s in "\n".join(script) for s in ("/test.jsonl","--unlock-test","rm -rf")):
            raise ValueError("test leakage or destructive command")
        job=dict(template)
        job["name"]=name
        job["description"]="EUR-LEX DEV only, frozen checkpoint inference + proper-set scorer; UNRUN"
        job["command"]="\n".join(script)+"\n"
        result.append(job)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template",type=Path,default=Path("qz_c64/c64-warm-bce.json"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--eval-root",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[43])
    p.add_argument("--full",action="store_true")
    p.add_argument("--max-input-tokens",type=int,default=8192)
    p.add_argument("--max-batch-tokens",type=int,default=8192)
    args=p.parse_args(argv)
    if args.output_dir.exists():
        p.error("refusing overwrite")
    template=json.loads(args.template.read_text(encoding="utf-8"))
    planned=jobs(template,args.repo_root,args.data_root,args.run_root,args.eval_root,
                 seeds=tuple(args.seeds),pilot=not args.full,
                 max_input=args.max_input_tokens,max_batch=args.max_batch_tokens)
    args.output_dir.mkdir(parents=True)
    for job in planned:
        (args.output_dir/(job["name"]+".json")).write_text(
            json.dumps(job,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"STAGED_NOT_SUBMITTED","jobs":len(planned),
                      "output_dir":str(args.output_dir)},indent=2))


if __name__=="__main__":
    main()
