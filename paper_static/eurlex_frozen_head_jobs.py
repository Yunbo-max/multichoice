#!/usr/bin/env python3
"""Stage CPU-only frozen-BCE CountHead pilot experiments, never launch or overwrite.

Three paired seeds, two comparable count-head families, optionally
independent head initialized to the TRAIN cardinality prior. Same BCE
feature caches/scores, same gold counts, same batch and step budget.
Learning-rate sweep is symmetric and selected by DEV Count NLL.
Every head predicts its own held-out DEV; the P0 script compares them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex


def create_plan(*,train_cache,dev_cache,train_gold,dev_gold,
                output_root, seeds=(43,44,45),
                rates=(0.001,0.0003),steps=600,batch=64,
                python="python",init_prior=True):
    for path in (train_cache,dev_cache,train_gold,dev_gold,output_root):
        if not Path(path).is_absolute():
            raise ValueError("all cache/data/output paths must be absolute for QZ reproducibility")
    if not seeds or len(set(seeds))!=len(seeds) or not rates:
        raise ValueError("seeds and LR grid are mandatory")
    if steps<1 or batch<1 or any(not 0<lr<1 for lr in rates):
        raise ValueError("invalid steps/batch/LR")
    choices=[("anchor","zero"),("independent","zero")]
    if init_prior:
        choices.append(("independent","prior"))
    work=[]
    root=Path(output_root)
    for seed in seeds:
        for mode,init in choices:
            for lr in rates:
                slug=f"{mode}-{init}-s{seed}-lr{lr:g}"
                head=root/"heads"/slug
                predict=root/"predictions"/slug
                cmd=[python,"-m","paper_static.eurlex_frozen_count","train",
                     "--train-cache",str(train_cache),
                     "--dev-cache",str(dev_cache),
                     "--train-gold",str(train_gold),
                     "--dev-gold",str(dev_gold),
                     "--mode",mode,"--init",init,
                     "--seed",str(seed),"--steps",str(steps),
                     "--batch",str(batch),"--eval-every","50",
                     "--lr",str(lr),"--device","cpu","--output",str(head)]
                inference=[python,"-m","paper_static.eurlex_frozen_count","predict",
                           "--gold",str(dev_gold),"--cache",str(dev_cache),
                           "--mode",mode,"--head",str(head),"--output",str(predict)]
                work.append({"seed":seed,"mode":mode,"init":init,"lr":lr,
                             "head":str(head),"predictions":str(predict),
                             "train_argv":cmd,"predict_argv":inference})
    return {"status":"CPU_HEAD_PLAN_NOT_RUN","steps":steps,"batch":batch,
            "seed_list":list(seeds),"lr_grid":list(rates),
            "feature_identity":"shared frozen BCE train/dev cache files",
            "best_model_selection":"DEV Count NLL, shared LR grid, no Test",
            "jobs":work}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("train-cache","dev-cache","train-gold","dev-gold","output-root"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--plan-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,nargs="+",default=[43,44,45])
    p.add_argument("--lr",type=float,nargs="+",default=[0.001,0.0003])
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--python",default="python")
    p.add_argument("--without-prior-control",action="store_true")
    args=p.parse_args(argv)
    if args.plan_dir.exists():
        p.error("plan directory exists")
    result=create_plan(train_cache=args.train_cache,dev_cache=args.dev_cache,
        train_gold=args.train_gold,dev_gold=args.dev_gold,
        output_root=args.output_root,seeds=tuple(args.seed),rates=tuple(args.lr),
        steps=args.steps,batch=args.batch,python=args.python,
        init_prior=not args.without_prior_control)
    args.plan_dir.mkdir(parents=True)
    (args.plan_dir/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.plan_dir/"run_cpu_heads.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "".join(
          "\ntest ! -e "+shlex.quote(j["head"])+"\n"
          + "test ! -e "+shlex.quote(j["predictions"])+"\n"
          + shlex.join(j["train_argv"])+"\n"
          + shlex.join(j["predict_argv"])+"\n"
          for j in result["jobs"]),
        encoding="utf-8")
    print(json.dumps({"status":result["status"],
                      "number_of_cpu_train_and_dev_predict_pairs":len(result["jobs"]),
                      "command_file":str(args.plan_dir/"run_cpu_heads.sh")},
                     indent=2))


if __name__=="__main__":
    main()
