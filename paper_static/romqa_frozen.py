#!/usr/bin/env python3
"""RoMQA Closed: shared frozen RoMQA-BCE features and matched Count Head controls.

The COMMON RoMQA-trained BCE checkpoint must be obtained via the EXISTING
harness/train_multiselect.py. Never reuse the EUR-LEX/CMB BCE scores as a
RoMQA-trained baseline or pretend no-evidence prompts match DPR evidence.

extract: frozen Qwen slot hidden, candidate logits, log Z, no gradient.
train: ONLY K+1 linear head, mode anchor or independent; prior from TRAIN.
predict: BCE, Anchor, Independent Count on the SAME frozen candidate scores.
All actions accept TRAIN/DEV only, never Test; no actual experiment launched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from harness.multiselect_metrics import read_dataset
from harness.set_distillation import input_digest, structured_prediction
from paper_static.eurlex_frozen_count import (
    centered_count_logz, count_log_probabilities, _feature_norm, _eval_count_nll
)


SCHEMA="romqa_closed_frozen_bce_v1"
SOURCE="romqa_closed"


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):sha.update(x)
    return sha.hexdigest()


def _check_rows(rows, split):
    if split not in ("train","dev") or not rows:
        raise ValueError("only official RoMQA Train/DEV are legal")
    k=len(rows[0]["options"])
    if k<2 or k>255 or any(
        row.get("source")!=SOURCE or row.get("split")!=split or
        row.get("selection_mode")!="multi" or row.get("allow_empty") is not True or
        len(row["options"])!=k or len(row["target"])!=k
        for row in rows):
        raise ValueError("frozen experiment requires fixed-K RoMQA Closed multi-select")
    return k


def check_cache(cache, rows, split, data_sha):
    import torch
    k=_check_rows(rows,split)
    if cache.get("schema")!=SCHEMA or cache.get("split")!=split:
        raise ValueError("cache source schema or split mismatch")
    if cache.get("k")!=k or cache.get("data_sha256")!=data_sha:
        raise ValueError("cache candidate count or dataset hash changed")
    n=len(rows)
    if any(key not in cache for key in (
            "hidden","scores","logz","ids","input_digests","label_counts",
            "checkpoint_identity","model_config_sha256","label_token_ids")):
        raise ValueError("incomplete frozen feature cache")
    hidden,scores,logz=cache["hidden"],cache["scores"],cache["logz"]
    if any(not isinstance(t,torch.Tensor) or not torch.isfinite(t).all()
           for t in (hidden,scores,logz)):
        raise ValueError("nonfinite or missing cached tensor")
    if (tuple(scores.shape)!=(n,k) or tuple(logz.shape)!=(n,k+1)
        or len(hidden.shape)!=2 or hidden.shape[0]!=n or hidden.shape[1]<64):
        raise ValueError("cache tensor shape mismatch")
    if len(cache["label_token_ids"])!=k:
        raise ValueError("wrong readout label token count")
    if (len(cache["ids"])!=n or len(cache["input_digests"])!=n
        or len(cache["label_counts"])!=n):
        raise ValueError("cache row count mismatch")
    if any(cache["ids"][i]!=row["id"] or
           cache["input_digests"][i]!=input_digest(row) or
           cache["label_counts"][i]!=sum(row["target"])
           for i,row in enumerate(rows)):
        raise ValueError("cache row identities or labels no longer match")
    return True


def load_cache(path):
    import torch
    data=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(data,dict) or data.get("schema")!=SCHEMA:
        raise ValueError("wrong/untrusted feature cache schema")
    return data


def check_pair(t,d,trows,drows,tsha,dsha):
    check_cache(t,trows,"train",tsha)
    check_cache(d,drows,"dev",dsha)
    if t["k"]!=d["k"] or t["hidden"].shape[1]!=d["hidden"].shape[1]:
        raise ValueError("Train/DEV feature dimension differs")
    for field in ("checkpoint_identity","model_config_sha256","label_token_ids"):
        if t[field]!=d[field]:
            raise ValueError(f"unpaired Train/DEV BCE checkpoint {field}")
    if set(t["ids"])&set(d["ids"]) or set(t["input_digests"])&set(d["input_digests"]):
        raise ValueError("Train/DEV overlapping questions/prompts")
    if set(r["group_id"] for r in trows)&set(r["group_id"] for r in drows):
        raise ValueError("RoMQA question-cluster split leakage")
    return True


def extract(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from harness.eval_multiselect import _prepare_prompts,validate_right_padding,check_export

    if args.output.exists():
        raise ValueError("immutable feature cache already exists")
    if args.split not in ("train","dev"):
        raise ValueError("never open RoMQA Test during feature extraction")
    rows=read_dataset(args.gold,expected_split=args.split)
    k=_check_rows(rows,args.split)
    if check_export(args.model)["objective"]!="bce":
        raise ValueError("freeze a RoMQA TRAINED BCE checkpoint, not a structured head")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU available")
    if args.device=="cpu" and args.dtype=="bfloat16":
        raise ValueError("use float32 on CPU")
    trainer,tok,labels,label_ids,prompts,lengths,batches=_prepare_prompts(args,rows)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    cls=Qwen3_5ForConditionalGeneration if config.model_type=="qwen3_5" else Qwen3_5ForCausalLM
    base,loading=cls.from_pretrained(
        args.model,dtype=getattr(torch,args.dtype),attn_implementation="sdpa",
        local_files_only=True,output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("BCE export is incomplete")
    base.requires_grad_(False)
    base=base.to(args.device).eval()
    backbone=getattr(base.model,"language_model",base.model)
    label_tensor=torch.tensor(label_ids[:k],device=args.device,dtype=torch.long)
    weights=base.get_output_embeddings().weight.index_select(0,label_tensor)
    hidden,scores,logz=[None]*len(rows),[None]*len(rows),[None]*len(rows)
    with torch.inference_mode():
        for batch in batches:
            encoded=tok([prompts[i] for i in batch],padding=True,
                        truncation=False,return_tensors="pt",add_special_tokens=False)
            validate_right_padding(encoded["attention_mask"].tolist(),
                                   [lengths[i] for i in batch])
            inputs={name:value.to(args.device) for name,value in encoded.items()}
            out=backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state
            offsets=inputs["attention_mask"].sum(-1)-1
            h=out[torch.arange(len(batch),device=args.device),offsets]
            logits=F.linear(h,weights).float()
            for j,original in enumerate(batch):
                s=logits[j].cpu().tolist()
                hidden[original]=h[j].float().cpu()
                scores[original]=logits[j].cpu()
                logz[original]=torch.tensor(centered_count_logz(s),dtype=torch.float32)
    if any(val is None for val in hidden+scores+logz):
        raise RuntimeError("cache extraction missing source questions")
    cache={
        "schema":SCHEMA,"split":args.split,"k":k,"data_sha256":digest(args.gold),
        "checkpoint_identity":str(args.model.resolve()),
        "model_config_sha256":digest(args.model/"config.json"),
        "label_token_ids":label_ids[:k],
        "hidden":torch.stack(hidden),"scores":torch.stack(scores),
        "logz":torch.stack(logz),
        "ids":[row["id"] for row in rows],
        "input_digests":[input_digest(row) for row in rows],
        "label_counts":[sum(row["target"]) for row in rows],
        "input_tokens":lengths,
        "parameter_weights_sha256_checked":False,
        "test_opened":False,
    }
    check_cache(cache,rows,args.split,digest(args.gold))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,args.output)
    print(json.dumps({"status":"ROMQA_FROZEN_FEATURES_EXTRACTED_NOT_TRAINED",
                      "n":len(rows),"k":k,"split":args.split,
                      "output":str(args.output),"test_opened":False},indent=2))


def train(args):
    import torch
    import torch.nn.functional as F
    if args.mode not in ("anchor","independent") or args.init not in ("zero","prior"):
        raise ValueError("unsupported count head")
    if args.mode=="anchor" and args.init=="prior":
        raise ValueError("prior initialization is the independent strong control")
    if args.output.exists():
        raise ValueError("immutable head directory already exists")
    if (args.steps<1 or args.batch<1 or args.eval_every<1 or
        not 0<args.lr<1 or args.device not in ("cpu","cuda")):
        raise ValueError("invalid training parameters")
    trows=read_dataset(args.train_gold,expected_split="train")
    drows=read_dataset(args.dev_gold,expected_split="dev")
    tc,dc=load_cache(args.train_cache),load_cache(args.dev_cache)
    check_pair(tc,dc,trows,drows,digest(args.train_gold),digest(args.dev_gold))
    torch.manual_seed(args.seed)
    hidden_t,hidden_d=tc["hidden"].float(),dc["hidden"].float()
    mu,std=_feature_norm(hidden_t,hidden_d)
    Xt,Xd=(hidden_t-mu)/std,(hidden_d-mu)/std
    yt,yd=(torch.tensor(tc["label_counts"],dtype=torch.long),
           torch.tensor(dc["label_counts"],dtype=torch.long))
    zt,zd=tc["logz"].float(),dc["logz"].float()
    k=tc["k"]
    head=torch.nn.Linear(Xt.shape[1],k+1).float()
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    if args.init=="prior":
        hist=torch.bincount(yt,minlength=k+1).float()+1
        head.bias.data.copy_(hist.log())
    device=torch.device(args.device)
    head=head.to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=0)
    indices=list(range(len(Xt)))
    rng=random.Random(args.seed)
    position=len(indices)
    history=[]
    best=(math.inf,None,None)
    initial_nll=_eval_count_nll(head,Xd,zd,yd,args.mode,device)
    for step in range(1,args.steps+1):
        if position+args.batch>len(indices):
            rng.shuffle(indices)
            position=0
        chosen=indices[position:position+args.batch]
        position+=len(chosen)
        if not chosen:raise RuntimeError("empty head train batch")
        raw=head(Xt[chosen].to(device))
        logits=raw+zt[chosen].to(device) if args.mode=="anchor" else raw
        loss=F.cross_entropy(logits,yt[chosen].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite proper Count CE")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),max_norm=5)
        optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            dev_nll=_eval_count_nll(head,Xd,zd,yd,args.mode,device)
            history.append({"step":step,"dev_count_nll":dev_nll})
            if dev_nll<best[0]:
                best=(dev_nll,step,{n:v.detach().float().cpu().clone()
                                     for n,v in head.state_dict().items()})
    dev_nll,step,weights=best
    if weights is None:
        raise RuntimeError("no validated count head")
    args.output.mkdir(parents=True)
    torch.save({
        "weight":weights["weight"],"bias":weights["bias"],
        "norm_mean":mu,"norm_scale":std,
    },args.output/"head.pt")
    provenance={
        "status":"ROMQA_FROZEN_HEAD_TRAINED_NO_TEST","schema":"romqa_count_head_v1",
        "mode":args.mode,"init":args.init,"seed":args.seed,"lr":args.lr,
        "steps":args.steps,"batch":args.batch,"eval_every":args.eval_every,
        "best_step":step,"initial_dev_count_nll":initial_nll,
        "best_dev_count_nll":dev_nll,
        "checkpoint_identity":tc["checkpoint_identity"],
        "model_config_sha256":tc["model_config_sha256"],
        "train_gold_sha256":digest(args.train_gold),
        "dev_gold_sha256":digest(args.dev_gold),
        "train_cache_sha256":digest(args.train_cache),
        "dev_cache_sha256":digest(args.dev_cache),
        "head_file_sha256":digest(args.output/"head.pt"),
        "frozen_backbone_and_candidate_scores":True,
        "selection_criterion":"DEV proper Count NLL only",
        "history":history,"test_opened":False,
    }
    (args.output/"training_summary.json").write_text(
        json.dumps(provenance,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":provenance["status"],"mode":args.mode,
                      "init":args.init,"best_dev_count_nll":dev_nll,
                      "output":str(args.output)},indent=2))


def predict(args):
    import torch
    if args.output.exists():
        raise ValueError("immutable predictions directory already exists")
    if args.mode not in ("bce","anchor","independent"):
        raise ValueError("unsupported inference mode")
    rows=read_dataset(args.gold,expected_split="dev")
    k=_check_rows(rows,"dev")
    cache=load_cache(args.cache)
    check_cache(cache,rows,"dev",digest(args.gold))
    head=None
    if args.mode!="bce":
        if args.head is None:
            raise ValueError("structured mode requires trained head")
        m=json.loads((args.head/"training_summary.json").read_text())
        if m.get("schema")!="romqa_count_head_v1" or m.get("mode")!=args.mode:
            raise ValueError("head family identity mismatch")
        if (m.get("checkpoint_identity")!=cache["checkpoint_identity"]
            or m.get("model_config_sha256")!=cache["model_config_sha256"]
            or m.get("dev_gold_sha256")!=digest(args.gold)
            or m.get("dev_cache_sha256")!=digest(args.cache)
            or m.get("head_file_sha256")!=digest(args.head/"head.pt")):
            raise ValueError("frozen head does not match this DEV cache and model")
        head=torch.load(args.head/"head.pt",map_location="cpu",weights_only=True)
        if tuple(head["weight"].shape)!=(k+1,cache["hidden"].shape[1]):
            raise ValueError("count head parameter dimensionality mismatch")
        identifier=f"romqa|frozen|{args.mode}|{m['init']}|seed{m['seed']}|step{m['best_step']}"
    else:
        identifier="romqa|frozen|bce|"+cache["checkpoint_identity"]
    output=[]
    for i,row in enumerate(rows):
        scores=cache["scores"][i].tolist()
        if head is None:
            p=[1/(1+math.exp(-s)) if s>=0 else math.exp(s)/(1+math.exp(s))
               for s in scores]
            item={"p":p,"logits":scores}
        else:
            h=cache["hidden"][i].float()
            normed=(h-head["norm_mean"])/head["norm_scale"]
            raw=(head["weight"]@normed+head["bias"]).tolist()
            item=structured_prediction(scores,raw,row,
                                       count_anchor=(args.mode=="anchor"))
            item["candidate_logits"]=scores
            item["count_residual_logits"]=raw
        item.update({
            "id":row["id"],
            "option_keys":[o["key"] for o in row["options"]],
            "input_digest":input_digest(row),
            "checkpoint_id":identifier,
            "provenance":{"model":cache["checkpoint_identity"],
                          "model_config_sha256":cache["model_config_sha256"],
                          "head":str(args.head) if args.head else None,
                          "frozen_candidate_scores":True},
        })
        output.append(item)
    args.output.mkdir(parents=True)
    with (args.output/"predictions.jsonl").open("x",encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
    (args.output/"provenance.json").write_text(
        json.dumps({"status":"ROMQA_DEV_FROZEN_PROBABILITIES_NO_GPU",
                    "mode":args.mode,"gold_sha256":digest(args.gold),
                    "cache_sha256":digest(args.cache),
                    "checkpoint_id":identifier,
                    "head":str(args.head) if args.head else None,
                    "test_opened":False},indent=2)+"\n")
    print(json.dumps({"status":"ROMQA_DEV_FROZEN_PROBABILITIES_NO_GPU",
                      "mode":args.mode,"n":len(output),
                      "predictions":str(args.output/"predictions.jsonl")},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    e=sub.add_parser("extract",help="one frozen BCE forward per RoMQA Train/DEV row")
    e.add_argument("--gold",type=Path,required=True)
    e.add_argument("--split",choices=["train","dev"],required=True)
    e.add_argument("--model",type=Path,required=True)
    e.add_argument("--output",type=Path,required=True)
    e.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    e.add_argument("--dtype",choices=["float32","bfloat16"],default="bfloat16")
    e.add_argument("--batch-size",type=int,default=1)
    e.add_argument("--max-input-tokens",type=int,default=16384)
    e.add_argument("--max-batch-tokens",type=int,default=16384)
    t=sub.add_parser("train",help="only train a K+1 linear count head")
    for name in ("train","dev"):
        t.add_argument(f"--{name}-gold",type=Path,required=True)
        t.add_argument(f"--{name}-cache",type=Path,required=True)
    t.add_argument("--mode",choices=["anchor","independent"],required=True)
    t.add_argument("--init",choices=["zero","prior"],default="zero")
    t.add_argument("--output",type=Path,required=True)
    t.add_argument("--seed",type=int,default=43)
    t.add_argument("--lr",type=float,default=.001)
    t.add_argument("--steps",type=int,default=600)
    t.add_argument("--batch",type=int,default=64)
    t.add_argument("--eval-every",type=int,default=50)
    t.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    q=sub.add_parser("predict",help="replay same frozen RoMQA BCE scores on DEV")
    q.add_argument("--gold",type=Path,required=True)
    q.add_argument("--cache",type=Path,required=True)
    q.add_argument("--mode",choices=["bce","anchor","independent"],required=True)
    q.add_argument("--head",type=Path)
    q.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    {"extract":extract,"train":train,"predict":predict}[args.action](args)


if __name__=="__main__":
    main()
