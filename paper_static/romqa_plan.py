#!/usr/bin/env python3
"""Stage-only RoMQA ICML paper experiments. NEVER launches a GPU/CPU training job.

All neural arms share official RoMQA Closed train/dev candidates+evidence,
a Qwen3.5 base, identical answer-slot readout, an independent RoMQA BCE warm
checkpoint, and seed/data/update budgets. No TEST. No QZ submit. Plan is
blocked until the RoMQA archive has been independently obtained and audited.

Outputs group-ordered shell scripts (warm, end-to-end controls, frozen feature
cache, 18 frozen heads, DEV probabilities/metrics). Scripts are NOT executed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex


SCHEMA="romqa_closed_decision_v1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for x in iter(lambda:file.read(1<<20),b""):h.update(x)
    return h.hexdigest()


def _cmd(args):
    return shlex.join([str(x) for x in args])


def _rel(args):
    if any(str(x).endswith("test.jsonl") or "test.noanswer" in str(x) for x in args):
        raise ValueError("RoMQA blind Test is prohibited in stage-only plan")
    return _cmd(args)


def stage(*,manifest,data,repo,base_model,modules,run_root,output,
          pilot=False,seeds=(43,44,45),lrs=(.001,.0003),
          head_steps=600,head_batch=64,batch=8,epochs=1):
    manifest=Path(manifest)
    data,repo,base_model,modules,run_root,output=map(
        Path,(data,repo,base_model,modules,run_root,output))
    if output.exists():
        raise ValueError("plan output exists; do not overwrite")
    if not all(p.is_absolute() for p in
               (data,repo,base_model,modules,run_root)):
        raise ValueError("all runtime paths must be absolute")
    if (not seeds or len(set(seeds))!=len(seeds) or
        not lrs or len(set(lrs))!=len(lrs) or
        any(not 0<v<1 for v in lrs) or
        batch<1 or head_batch<1 or head_steps<1 or epochs<1):
        raise ValueError("bad seed/LR/step configuration")
    meta=json.loads(manifest.read_text(encoding="utf-8"))
    if (meta.get("schema")!=SCHEMA or meta.get("dataset")!="facebookresearch/romqa"
        or meta.get("test_opened") is not False
        or meta.get("candidate_injection") is not False):
        raise ValueError("dataset source/provenance not approved")
    if not meta.get("source",{}).get("kind"):
        raise ValueError("official data origin SHA not recorded")
    group=meta["pilot"] if pilot else meta["files"]
    if any(key not in group for key in ("train","dev")):
        raise ValueError("requested pilot/full Train/DEV data missing")
    selected={
        name:data/group[name].get("path",group[name].get("name"))
        for name in ("train","dev")
    }
    eval_sidecar=data/group["dev"]["eval_sidecar"]
    for name in ("train","dev"):
        entry=group[name]
        if entry.get("n",0)<2 or (not isinstance(entry.get("sha256"),str)
                                or len(entry["sha256"])!=64):
            raise ValueError("no verified Train/DEV split")
        if not selected[name].is_file() or digest(selected[name])!=entry["sha256"]:
            raise ValueError(f"{name} source hash mismatch")
    if (not eval_sidecar.is_file() or
        digest(eval_sidecar)!=group["dev"]["eval_sha256"]):
        raise ValueError("DEV official full-answer sidecar missing or altered")
    max_input=meta["max_input_tokens"]
    if (not isinstance(max_input,int) or max_input<1024 or
        meta.get("expected_k",0)<2):
        raise ValueError("missing audited prompt/K budget")
    if not modules.is_file() or not base_model.is_dir():
        raise ValueError("Qwen base checkpoint and module manifest required")
    if not (repo/"harness"/"train_multiselect.py").is_file():
        raise ValueError("existing common Qwen trainer is missing")

    py="python"
    trainer=repo/"harness"/"train_multiselect.py"
    common=[
        "--train",selected["train"],"--dev",selected["dev"],
        "--modules",modules,"--target","gold","--lora",
        "--lora-r",16,"--lora-alpha",32,"--global-batch",batch,
        "--epochs",epochs,"--eval-every-frac",.2,
        "--lr",2e-5,"--max-input-tokens",max_input,
        "--max-batch-tokens",max_input,
    ]
    warm=run_root/"warm_bce"
    scripts={}
    scripts["01_warm_bce.sh"]=[
        _rel([py,trainer,"--model",base_model,*common,
              "--objective","bce","--seed",44,"--output",warm]),
    ]
    arms={
        "bce_continue":["--objective","bce"],
        "anchor":["--objective","set","--set-loss","set_ce",
                  "--count-anchor","--count-lr-mult",32],
        "anchor_detach":["--objective","set","--set-loss","set_ce",
                         "--count-anchor","--count-anchor-detach","--count-lr-mult",32],
        "independent_zero":["--objective","set","--set-loss","set_ce",
                             "--count-lr-mult",32],
    }
    continuations=[]
    for seed in seeds:
        for arm,extra in arms.items():
            dest=run_root/"continuations"/f"{arm}-s{seed}"
            continuations.append(_rel([
                py,trainer,"--model",warm/"final",*common,
                "--seed",seed,"--output",dest,*extra]))
    scripts["02_matched_lora_continuations.sh"]=continuations

    cache=run_root/"frozen_cache"
    extracts=[]
    for split in ("train","dev"):
        extracts.append(_rel([
            py,"-m","paper_static.romqa_frozen","extract",
            "--gold",selected[split],"--split",split,
            "--model",warm/"final",
            "--output",cache/f"{split}.pt",
            "--device","cuda","--dtype","bfloat16",
            "--batch-size",1,
            "--max-input-tokens",max_input,
            "--max-batch-tokens",max_input]))
    scripts["03_extract_same_bce_train_dev.sh"]=extracts

    heads=[]
    evals=[]
    combos=[("anchor","zero"),("independent","zero"),("independent","prior")]
    for seed in seeds:
        for mode,init in combos:
            for lr in lrs:
                slug=f"{mode}-{init}-s{seed}-lr{lr:g}"
                dest=run_root/"frozen_heads"/slug
                heads.append(_rel([
                    py,"-m","paper_static.romqa_frozen","train",
                    "--train-gold",selected["train"],
                    "--dev-gold",selected["dev"],
                    "--train-cache",cache/"train.pt",
                    "--dev-cache",cache/"dev.pt",
                    "--mode",mode,"--init",init,
                    "--output",dest,"--device","cpu",
                    "--steps",head_steps,"--batch",head_batch,
                    "--eval-every",50,"--lr",lr,"--seed",seed]))
                pred=run_root/"predictions"/slug
                report=run_root/"reports"/slug
                evals.append(_rel([
                    py,"-m","paper_static.romqa_frozen","predict",
                    "--gold",selected["dev"],"--cache",cache/"dev.pt",
                    "--mode",mode,"--head",dest,"--output",pred]))
                evals.append(_rel([
                    py,"-m","paper_static.romqa_metrics",
                    "--data",selected["dev"],"--sidecar",eval_sidecar,
                    "--predictions",pred/"predictions.jsonl",
                    "--output",report]))
    scripts["04_train_matched_frozen_heads.sh"]=heads
    bce_pred=run_root/"predictions"/"frozen-bce"
    bce_report=run_root/"reports"/"frozen-bce"
    evals=[_rel([py,"-m","paper_static.romqa_frozen","predict",
                 "--gold",selected["dev"],"--cache",cache/"dev.pt",
                 "--mode","bce","--output",bce_pred]),
           _rel([py,"-m","paper_static.romqa_metrics",
                 "--data",selected["dev"],"--sidecar",eval_sidecar,
                 "--predictions",bce_pred/"predictions.jsonl",
                 "--output",bce_report])]+evals
    scripts["05_evaluate_frozen_dev_only.sh"]=evals
    output.mkdir(parents=True)
    for file,commands in scripts.items():
        # No automatic execution. This is a human/agent-readable handoff.
        path=output/file
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "# STAGED, NOT EXECUTED. All model outputs must be NEW paths.\n"
            "# Caution: ensure valid H100 memory/context, scheduler wall and checkpoint presence.\n"
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
            "export PYTHONPATH="+shlex.quote(str(repo))+"\n"
            "cd "+shlex.quote(str(repo))+"\n"
            "\n".join(commands)+"\n",encoding="utf-8")
    summary={
        "status":"ROMQA_PLAN_ONLY_NOT_SUBMITTED_NOT_TRAINED",
        "data_manifest_sha256":digest(manifest),
        "source_data_sha256":{n:digest(v) for n,v in selected.items()},
        "dev_eval_sha256":digest(eval_sidecar),
        "full_or_pilot":"pilot" if pilot else "full",
        "train_n":group["train"]["n"],"dev_n":group["dev"]["n"],
        "k":meta["expected_k"],"max_input_tokens":max_input,
        "seeds":list(seeds),"head_lrs":list(lrs),
        "arms":[*arms],"frozen_heads":["bce","anchor_zero",
                         "independent_zero","independent_train_prior"],
        "steps_not_executed":True,
        "real_gpu_readiness_unverified":True,
        "test_opened":False,
        "evidence_mode":meta["evidence_mode"],
        "science_go_no_go":"Compare DEV proper Count/Joint NLL against BOTH "+
           "BCE and prior-initialized independent head; then hold-out evaluation. "+
           "Do NOT claim Count Anchor superiority from head-zero-init alone.",
        "scripts":{k:len(v) for k,v in scripts.items()},
    }
    (output/"plan.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base-model",type=Path,required=True)
    p.add_argument("--modules",type=Path,required=True)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--seed",type=int,nargs="+",default=[43,44,45])
    p.add_argument("--lr",type=float,nargs="+",default=[.001,.0003])
    p.add_argument("--head-steps",type=int,default=600)
    p.add_argument("--head-batch",type=int,default=64)
    p.add_argument("--batch",type=int,default=8)
    args=p.parse_args(argv)
    result=stage(manifest=args.manifest,data=args.data,repo=args.repo,
                 base_model=args.base_model,modules=args.modules,
                 run_root=args.run_root,output=args.output,
                 pilot=args.pilot,seeds=args.seed,lrs=args.lr,
                 head_steps=args.head_steps,head_batch=args.head_batch,
                 batch=args.batch)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
