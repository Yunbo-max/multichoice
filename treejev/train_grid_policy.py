#!/usr/bin/env python3
"""Train a tiny grid policy/value baseline from certified shortest-path states."""
import argparse
import json
import random
from pathlib import Path
from treejev.grid_model import grid_tensor, make_model, training_examples, loss_terms

def evaluate(model,rows,batch_size,max_side,device):
    import torch
    import torch.nn.functional as F
    model.eval();correct=0;mae=0.;nll=0.
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            part=rows[start:start+batch_size]
            x=grid_tensor([r["state"] for r in part],max_side).to(device)
            logits,values=model(x)
            gold=torch.tensor([r["target"] for r in part],device=device,dtype=torch.float32)
            probs=gold/gold.sum(-1,keepdim=True)
            predicted=logits.argmax(-1)
            correct+=float(gold[torch.arange(len(part),device=device),predicted].sum())
            nll+=float((-(probs*F.log_softmax(logits,-1)).sum(-1)).sum())
            d=torch.tensor([r["distance_to_goal"]/10 for r in part],device=device)
            mae+=float((values-d).abs().sum())*10
    return {"optimal_action_accuracy":correct/len(rows),
            "action_ce":nll/len(rows),"distance_mae":mae/len(rows),
            "n":len(rows)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--max-side",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    if args.output.exists():p.error(f"refusing overwrite: {args.output}")
    if args.epochs<1 or args.width<8 or args.batch_size<1 or args.lr<=0:
        p.error("invalid train setting")
    import torch
    torch.manual_seed(args.seed);random.seed(args.seed)
    train=training_examples(args.train);dev=training_examples(args.dev)
    if {r["group_id"] for r in train}&{r["group_id"] for r in dev}:
        raise ValueError("train/dev layout leakage")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model=make_model(args.width).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    args.output.mkdir(parents=True)
    history=[];best=(-1.,float("inf"))
    for epoch in range(args.epochs):
        model.train();random.Random(args.seed+epoch).shuffle(train)
        training_loss=0.;count=0
        for start in range(0,len(train),args.batch_size):
            part=train[start:start+args.batch_size]
            x=grid_tensor([r["state"] for r in part],args.max_side).to(args.device)
            logits,values=model(x)
            loss,_,_=loss_terms(logits,values,part)
            if not torch.isfinite(loss).item():raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            training_loss+=float(loss)*len(part);count+=len(part)
        dev_stats=evaluate(model,dev,args.batch_size,args.max_side,args.device)
        dev_stats.update({"epoch":epoch+1,"train_loss":training_loss/count})
        history.append(dev_stats)
        key=(dev_stats["optimal_action_accuracy"],dev_stats["action_ce"])
        if key[0]>best[0] or (key[0]==best[0] and key[1]<best[1]):
            best=key
            torch.save({"model":model.state_dict(),"width":args.width,
                        "max_side":args.max_side,"seed":args.seed,
                        "dev":dev_stats,"source":"oracle_sokoban_train_only"},
                        args.output/"best.pt")
    (args.output/"training.json").write_text(json.dumps({
        "status":"FINISHED","history":history,"best_dev":best,
        "train":str(args.train),"dev":str(args.dev),
        "official_test_touched":False,"model_kind":"symbolic_grid_control",
        "warning":"The state values are learned only along oracle-labeled states; "
                  "the full off-policy branch-recovery value is not validated."
    },indent=2)+"\n")
    print(json.dumps({"best_dev":best,"weights":str(args.output/"best.pt")}))

if __name__=="__main__":main()
