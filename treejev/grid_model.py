"""Small symbolic-grid action/value baseline for testing learned search cheaply.

This is an AUDITED CONTROL, not the proposed 4B latent world model. It sees
only the current visible ASCII grid and learns tied-optimal action policy and
remaining-step estimates from the oracle training split.
"""
from pathlib import Path
CHARS = {" ":0,"-":0,"#":1,"@":2,"$":3,".":4,"+":5,"*":6}
PAD=7

def grid_tensor(boards,max_side=16):
    import torch
    tensor=torch.full((len(boards),max_side,max_side),PAD,dtype=torch.long)
    for b,board in enumerate(boards):
        lines=board.splitlines()
        if not lines or max(len(line) for line in lines)>max_side or len(lines)>max_side:
            raise ValueError("invalid or oversized ASCII board")
        for r,line in enumerate(lines):
            for c,ch in enumerate(line):
                if ch not in CHARS:raise ValueError(f"unknown board glyph {ch!r}")
                tensor[b,r,c]=CHARS[ch]
    return tensor

def make_model(width=64):
    import torch
    from torch import nn
    class GridPolicyValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=nn.Embedding(8,32)
            self.cnn=nn.Sequential(
                nn.Conv2d(32,width,3,padding=1),nn.ReLU(),
                nn.Conv2d(width,width,3,padding=1),nn.ReLU(),
                nn.Conv2d(width,width,3,padding=1),nn.ReLU())
            self.policy=nn.Linear(width,4)
            self.value=nn.Linear(width,1)
        def forward(self,x):
            mask=(x!=PAD).float()
            y=self.embedding(x).permute(0,3,1,2)
            y=self.cnn(y)
            pooled=(y*mask[:,None]).sum((2,3))/mask.sum((1,2)).clamp_min(1)[:,None]
            return self.policy(pooled),torch.nn.functional.softplus(self.value(pooled)[:,0])
    return GridPolicyValue()

def training_examples(jsonl):
    from treejev.data import read_jsonl
    rows=list(read_jsonl(jsonl))
    if not rows:raise ValueError("empty oracle state data")
    for r in rows:
        if r.get("source")!="sokoban_oracle":
            raise ValueError("grid baseline requires certified oracle state records")
        if not isinstance(r.get("distance_to_goal"),int) or r["distance_to_goal"]<1:
            raise ValueError("remaining distance label missing")
        if len(r["target"])!=4 or sum(r["target"])<1:
            raise ValueError("invalid optimal-action target")
    return rows

def loss_terms(logits,values,rows,distance_scale=10.0,value_weight=0.25):
    import torch
    import torch.nn.functional as F
    target=torch.tensor([r["target"] for r in rows],dtype=torch.float32,device=logits.device)
    soft=target/target.sum(-1,keepdim=True)
    target_d=torch.tensor([r["distance_to_goal"]/distance_scale for r in rows],
                          dtype=torch.float32,device=logits.device)
    policy=-(soft*F.log_softmax(logits,dim=-1)).sum(-1)
    value=F.smooth_l1_loss(values,target_d,reduction="none")
    return (policy+value_weight*value).mean(),policy.mean(),value.mean()

def load_checkpoint(path,device="cpu"):
    import torch
    saved=torch.load(path,map_location=device,weights_only=True)
    model=make_model(saved["width"]).to(device)
    model.load_state_dict(saved["model"],strict=True);model.eval()
    return model,saved
