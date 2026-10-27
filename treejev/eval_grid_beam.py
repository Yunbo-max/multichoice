#!/usr/bin/env python3
"""Fixed-budget beam-control evaluation on user-supplied *held-out* ASCII maps.

The simulator is queried for every imaginary branch: this is a symbolic
planning baseline, not a valid ARC-AGI-3 world-model evaluation.
"""
import argparse
import json
from pathlib import Path
from treejev.sokoban import Board, ACTION_ORDER
from treejev.grid_model import grid_tensor, load_checkpoint
from treejev.beam import beam_search
from treejev.data import read_jsonl

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--maps",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--widths",type=int,nargs="+",default=[1,2,4,8])
    p.add_argument("--node-budget",type=int,default=128)
    p.add_argument("--horizon",type=int,default=64)
    p.add_argument("--device",default="cpu")
    p.add_argument("--value-weight",type=float,default=0.25)
    args=p.parse_args()
    if args.output.exists():p.error(f"refusing overwrite {args.output}")
    import torch
    import torch.nn.functional as F
    model,saved=load_checkpoint(args.checkpoint,args.device)
    results=[];levels=list(read_jsonl(args.maps))
    if not levels:raise ValueError("no held-out maps")
    for level in levels:
        board,root=Board.from_ascii(level["ascii"])
        def infer(state):
            with torch.no_grad():
                logits,value=model(grid_tensor([board.render_ascii(state)],saved["max_side"]).to(args.device))
                return F.log_softmax(logits[0],dim=-1),float(value[0])*10
        def policy(state):
            logits,_=infer(state)
            return {a:float(logits[j]) for j,a in enumerate(ACTION_ORDER)
                    if a in board.legal_actions(state)}
        def heuristic(state):
            return args.value_weight*infer(state)[1]
        for width in args.widths:
            result=beam_search(board,root,policy,width=width,horizon=args.horizon,
                               max_expanded=args.node_budget,
                               estimated_remaining=heuristic)
            current=root
            for action in result.actions:
                current=board.next_state(current,action)
            if result.won and not board.won(current):
                raise RuntimeError("beam claimed win but simulator denies it")
            results.append({"level_id":level["level_id"],"width":width,
                            "won":result.won,"steps":len(result.actions),
                            "expanded_nodes":result.expanded,
                            "node_budget":args.node_budget})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({
      "results":results,"source":str(args.maps),"checkpoint":str(args.checkpoint),
      "uses_exact_simulator":True,"valid_arc3_eval":False
    },indent=2)+"\n")
    print(json.dumps({"levels":len(levels),"runs":len(results),"output":str(args.output)}))

if __name__=="__main__":main()
