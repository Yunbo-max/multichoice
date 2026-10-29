#!/usr/bin/env python3
"""Prepare group-disjoint Sokoban visual SFT or exact ASCII multi-optimal SFT.

Does not open Valen official eval_sokoban or use any hidden metadata as input.
Examples:
python -m treejev.prepare valen --train-jsonl data/Valen-Eval-Game/train_sokoban/train.jsonl --output-dir runs/tree/valen
python -m treejev.prepare oracle --maps data/my_ascii_levels.jsonl --output-dir runs/tree/oracle
python -m treejev.prepare oracle --demo --output-dir runs/tree/demo
"""
import argparse
import json
from pathlib import Path
from treejev.sokoban import Board, DEMO_MAP
from treejev.data import (read_jsonl, save_jsonl, split_groups,
                          valen_records, generate_oracle_level)
from treejev.oracle import SearchBudgetExceeded, NoSolutionWithinBudget

def emit_summary(directory, data):
    path=Path(directory)/"summary.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    visual=sub.add_parser("valen",help="Valen 30k train only; never use official eval")
    visual.add_argument("--train-jsonl",type=Path,required=True)
    visual.add_argument("--asset-root",type=Path)
    visual.add_argument("--output-dir",type=Path,required=True)
    visual.add_argument("--no-image-check",action="store_true",
                        help="dataset-schema debugging before extracting official archives")
    oracle=sub.add_parser("oracle",help="certified all-shortest Sokoban DAG and paths")
    oracle.add_argument("--maps",type=Path,help="JSONL with level_id and ascii rows")
    oracle.add_argument("--demo",action="store_true")
    oracle.add_argument("--output-dir",type=Path,required=True)
    oracle.add_argument("--max-nodes",type=int,default=20000)
    oracle.add_argument("--max-depth",type=int,default=80)
    oracle.add_argument("--max-paths",type=int,default=32)
    oracle.add_argument("--skip-budget-failures",action="store_true")
    for obj in (visual,oracle):
        obj.add_argument("--seed",type=int,default=43)
        obj.add_argument("--dev-fraction",type=float,default=0.1)
    args=parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory exists: {args.output_dir}")
    if args.mode=="valen":
        if "eval_sokoban" in args.train_jsonl.parts:
            parser.error("official eval cannot be used for training")
        root=args.asset_root or args.train_jsonl.parent
        rows=valen_records(args.train_jsonl,root,require_images=not args.no_image_check)
        train,dev=split_groups(rows,args.dev_fraction,args.seed)
        args.output_dir.mkdir(parents=True)
        save_jsonl(args.output_dir/"train_visual.jsonl",train)
        save_jsonl(args.output_dir/"dev_visual.jsonl",dev)
        summary={"kind":"valen_image_sft","train_rows":len(train),"dev_rows":len(dev),
                 "train_groups":len({r["group_id"] for r in train}),
                 "dev_groups":len({r["group_id"] for r in dev}),
                 "multi_optimal_train":sum(len(r["optimal_actions"])>1 for r in train),
                 "source":str(args.train_jsonl),"official_test_touched":False,
                 "metadata_excluded_from_model":True}
        emit_summary(args.output_dir,summary)
    else:
        if args.demo == bool(args.maps):
            parser.error("provide exactly one of --maps or --demo")
        levels=([{"level_id":"demo","ascii":DEMO_MAP}] if args.demo else list(read_jsonl(args.maps)))
        if not levels: parser.error("no levels")
        seen=set(); rows=[]; trajectories=[]; solved=[]; skipped=[]
        for i,level in enumerate(levels):
            lid=str(level.get("level_id",i))
            if lid in seen: raise ValueError(f"duplicate level_id {lid}")
            seen.add(lid)
            board,initial=Board.from_ascii(level["ascii"])
            try:
                r,t,info=generate_oracle_level(
                    lid,board,initial,max_nodes=args.max_nodes,
                    max_depth=args.max_depth,trajectories=args.max_paths)
            except (SearchBudgetExceeded,NoSolutionWithinBudget) as exc:
                if not args.skip_budget_failures: raise
                skipped.append({"level_id":lid,"reason":str(exc)})
                continue
            rows.extend(r); trajectories.extend(t); solved.append(info)
        train,dev=split_groups(rows,args.dev_fraction,args.seed)
        dev_groups={r["group_id"] for r in dev}
        for r in dev:r["split"]="dev"
        args.output_dir.mkdir(parents=True)
        save_jsonl(args.output_dir/"train_multi.jsonl",train)
        save_jsonl(args.output_dir/"dev_multi.jsonl",dev)
        save_jsonl(args.output_dir/"trajectories_train.jsonl",
                   [r for r in trajectories if r["group_id"] not in dev_groups])
        save_jsonl(args.output_dir/"trajectories_dev.jsonl",
                   [r for r in trajectories if r["group_id"] in dev_groups])
        summary={"kind":"certified_shortest_dag","train_states":len(train),
                 "dev_states":len(dev),"total_paths":len(trajectories),
                 "solved_levels":solved,"skipped":skipped,
                 "certified_labels_only":True,"official_test_touched":False}
        emit_summary(args.output_dir,summary)
    print(json.dumps({"output":str(args.output_dir),**summary},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
