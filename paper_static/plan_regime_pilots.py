"""Plan (do not launch) matched 4B LoRA experiments on three controlled regimes.

Stage 1 is a common BCE warm start from Qwen3.5-4B base. Stage 2 compares
identical data/updates and initialization: BCE continuation, independent
Set CE, anchored Set CE, and anchored-detached Set CE. The paired relational
Qwen model is not wired into the trainer yet, so it is NOT included here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

FAMILIES={
    "bce_continue":[],
    "set_ce_independent":["--objective","set","--set-loss","set_ce"],
    "set_ce_anchor":["--objective","set","--set-loss","set_ce","--count-anchor"],
    "set_ce_anchor_detach":["--objective","set","--set-loss","set_ce",
                             "--count-anchor","--count-anchor-detach"],
}


def _sha(path):
    p=Path(path)
    if not p.is_file():
        return None
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def plan(data_dir,model,modules,root,bce_checkpoint,seeds=(43,44,45),
         stage1_steps=450,stage2_steps=100,batch=64,lr=2e-5):
    train=data_dir/"student"/"train.jsonl"
    dev=data_dir/"student"/"dev.jsonl"
    if not train.exists() or not dev.exists():
        raise ValueError("generate controlled regime data first, with student/train.jsonl and dev.jsonl")
    if stage1_steps<1 or stage2_steps<1 or batch<1 or lr<=0:
        raise ValueError("invalid common training budget")
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("invalid paired seed list")
    common=["--train",str(train),"--dev",str(dev),"--modules",str(modules),
            "--target","gold","--global-batch",str(batch),
            "--epochs","1","--eval-every-frac","1.0",
            "--lr",str(lr),"--lora","--lora-r","16","--lora-alpha","32"]
    warm_root=root/"warm_bce_shared"
    warm_argv=["torchrun","--standalone","--nnodes=1","--nproc_per_node=1",
               "harness/train_multiselect.py",
               "--model",str(model),*common,
               "--objective","bce","--seed","44",
               "--max-steps",str(stage1_steps),"--output",str(warm_root)]
    rows=[{"id":"warm-bce-s44","phase":"prerequisite",
           "arm":"bce","seed":44,"argv":warm_argv,
           "status":"NOT_RUN","requires_checkpoint":str(model),
           "created_checkpoint":str(warm_root/"final")}]
    for seed in seeds:
        for arm,extra in FAMILIES.items():
            argv=["torchrun","--standalone","--nnodes=1","--nproc_per_node=1",
                  "harness/train_multiselect.py","--model",str(bce_checkpoint),
                  *common,"--seed",str(seed),"--max-steps",str(stage2_steps),
                  "--output",str(root/"matched"/f"seed_{seed}"/arm),*extra]
            if arm=="bce_continue":
                argv+=["--objective","bce"]
            rows.append({"id":f"seed-{seed}-{arm}","phase":"paired-continuation",
                        "arm":arm,"seed":seed,"argv":argv,
                        "requires_checkpoint":str(bce_checkpoint),"status":"NOT_RUN"})
    return {"status":"PLAN_ONLY_NO_GPU_JOB",
            "experiment":"I-C-R controlled 4B LoRA matched initialization",
            "data":{"train":str(train),"dev":str(dev),
                    "train_sha256":_sha(train),"dev_sha256":_sha(dev)},
            "requires":{"model":str(model),"modules":str(modules),
                        "common_bce_checkpoint":str(bce_checkpoint)},
            "budget":{"world_size":1,"batch":batch,"lr":lr,
                      "warm_steps":stage1_steps,"continuation_steps":stage2_steps},
            "rows":rows,"unimplemented":"learned pair predictor and Qwen pairwise training loss"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--modules",type=Path,default=Path("innerjev/configs/4B/modules.json"))
    p.add_argument("--run-root",type=Path,default=Path("runs/paper1_regime_4b"))
    p.add_argument("--bce-checkpoint",type=Path,
                   default=Path("runs/paper1_regime_4b/warm_bce_shared/final"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[43,44,45])
    p.add_argument("--warm-steps",type=int,default=450)
    p.add_argument("--continuation-steps",type=int,default=100)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--lr",type=float,default=2e-5)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite plan")
    result=plan(args.data_dir,args.model,args.modules,args.run_root,args.bce_checkpoint,
                seeds=tuple(args.seeds),stage1_steps=args.warm_steps,
                stage2_steps=args.continuation_steps,batch=args.batch,lr=args.lr)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"jobs":len(result["rows"]),
                      "output":str(args.output)},indent=2))


if __name__=="__main__":
    main()
