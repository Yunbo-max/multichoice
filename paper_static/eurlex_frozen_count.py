#!/usr/bin/env python3
"""P1 frozen-BCE experiment, from model weights to reproducible count-only heads.

Workflow:
  extract --gold TRAIN|DEV --model COMMON_BCE_CHECKPOINT --output train.pt|dev.pt
  train --train-cache train.pt --dev-cache dev.pt --train-gold TRAIN
        --dev-gold DEV --mode anchor|independent --output NEW/head_dir
  predict --cache dev.pt --gold DEV --head NEW/head_dir --output NEW/predict_dir
  predict --cache dev.pt --gold DEV --mode bce --output NEW/bce_predict_dir

No full-Qwen gradients: extraction calls the frozen Qwen Readout-compatible
backbone exactly once per sample and saves (slot hidden, candidate scores,
full shifted log Z_m). Train the two matched linear heads on the SAME inputs,
gold counts, update budget, and (seed-specific) shuffled batches. Head loss:
  anchor:      CE(m, softmax(W h+b + log Z_m(s_BCE)))
  independent: CE(m, softmax(W h+b))
At the unrestricted probability-family level, both count laws admit
reparameterization c_m(x)=a_m(x)+log Z_m(s_BCE(x)). With LINEAR heads
on a frozen h(x), log Z_m(s_BCE(x)) is generally NOT linear in h, so
the actual restricted architectures have different inductive biases.
The comparison tests this prior, optimization and calibration; it does NOT
establish a new unrestricted probability family. P1 never touches TEST.

Requires installed torch for training and a local Qwen3.5 checkpoint for
extract. Export and readout can use CPU or GPU. Torch-only operations import
lazily so offline schema and numerical-contract tests run without torch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from harness.multiselect_metrics import read_dataset
from harness.set_distillation import (
    allowed_counts, input_digest, log_partitions, structured_prediction,
)


MODES=("anchor","independent","bce")
SCHEMA="eurlex_frozen_bce_features_v1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1<<20),b""):
            h.update(piece)
    return h.hexdigest()


def centered_count_logz(scores):
    """Stable log Z_m for the RAW logits; return all K+1 counts."""
    if not scores or any(type(s) not in (float,int) or not math.isfinite(s)
                         for s in scores):
        raise ValueError("finite raw logits required")
    center=max(scores)
    z=log_partitions([s-center for s in scores])
    return [float(v+m*center) for m,v in enumerate(z)]


def verify_cache(cache, rows, *, split, data_sha=None):
    if cache.get("schema")!=SCHEMA or cache.get("split")!=split:
        raise ValueError("cache version/split mismatch")
    if split not in ("train","dev"):
        raise ValueError("frozen head only permits Train or DEV")
    if not rows or any(r.get("split")!=split or len(r["target"])!=100
                       or r.get("source")!="lexglue_eurlex" for r in rows):
        raise ValueError("expected 100-option EUR-LEX Train/DEV")
    n=len(rows)
    keys=["hidden","scores","logz","ids","input_digests","label_counts"]
    if any(k not in cache for k in keys):
        raise ValueError("incomplete frozen feature cache")
    hidden,scores,logz=cache["hidden"],cache["scores"],cache["logz"]
    if len(hidden)!=n or tuple(scores.shape)!=(n,100) or tuple(logz.shape)!=(n,101):
        raise ValueError("feature/score/count normalizer shape mismatch")
    if len(hidden.shape)!=2 or hidden.shape[1]<64:
        raise ValueError("hidden dimension invalid")
    if len(cache["ids"])!=n or len(cache["input_digests"])!=n or len(cache["label_counts"])!=n:
        raise ValueError("feature cache identity count mismatch")
    if any(cache["ids"][i]!=r["id"] or
           cache["input_digests"][i]!=input_digest(r) or
           cache["label_counts"][i]!=sum(r["target"])
           for i,r in enumerate(rows)):
        raise ValueError("features differ from the exact corresponding gold rows")
    if cache.get("data_sha256")!=data_sha and data_sha is not None:
        raise ValueError("Gold dataset SHA differs from frozen cache")
    if not cache.get("checkpoint_identity") or not cache.get("label_token_ids"):
        raise ValueError("frozen model identity/answer token IDs missing")
    return True


def load_cache(path):
    import torch
    # Only tensors and primitive metadata; no arbitrary pickled Python objects.
    cache=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(cache,dict) or cache.get("schema")!=SCHEMA:
        raise ValueError("unknown untrusted or unsupported cache format")
    for key in ("hidden","scores","logz"):
        if not isinstance(cache.get(key),torch.Tensor) or not torch.isfinite(cache[key]).all():
            raise ValueError(f"cache has nonfinite or missing {key}")
    return cache


def validate_pair(train,dev,train_rows,dev_rows,train_sha,dev_sha):
    verify_cache(train,train_rows,split="train",data_sha=train_sha)
    verify_cache(dev,dev_rows,split="dev",data_sha=dev_sha)
    for field in ("checkpoint_identity","label_token_ids","model_config_sha256"):
        if train.get(field)!=dev.get(field):
            raise ValueError(f"Train and DEV features originate from different {field}")
    if set(train["ids"]) & set(dev["ids"]):
        raise ValueError("Train/DEV feature caches share document IDs")
    if set(train["input_digests"]) & set(dev["input_digests"]):
        raise ValueError("Train/DEV feature caches share identical prompts")
    if train["hidden"].shape[1]!=dev["hidden"].shape[1]:
        raise ValueError("hidden dimensionality mismatch")
    return True


def _feature_norm(train,dev):
    import torch
    if train.ndim!=2 or dev.ndim!=2 or train.shape[1]!=dev.shape[1]:
        raise ValueError("invalid hidden shape")
    mean=train.float().mean(dim=0)
    std=train.float().std(dim=0,unbiased=False).clamp(min=.05)
    return mean,std


def count_log_probabilities(raw_head,logz,mode):
    """Pure reference: COUNT-only law conditioned on the public 0..K support."""
    if mode not in ("anchor","independent"):
        raise ValueError("mode must be anchor or independent")
    if len(raw_head)!=len(logz) or any(not math.isfinite(x) for x in raw_head+logz):
        raise ValueError("invalid count scores")
    z=[(raw_head[m]+logz[m] if mode=="anchor" else raw_head[m])
       for m in range(len(raw_head))]
    hi=max(z)
    norm=hi+math.log(sum(math.exp(v-hi) for v in z))
    return [v-norm for v in z]


def extract(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from harness.eval_multiselect import _prepare_prompts, validate_right_padding, check_export

    if args.split not in ("train","dev"):
        raise ValueError("TEST extraction is deliberately not supported")
    if args.output.exists():
        raise ValueError("output cache already exists")
    rows=read_dataset(args.gold,expected_split=args.split)
    if any(r.get("source")!="lexglue_eurlex" or len(r["options"])!=100 for r in rows):
        raise ValueError("frozen features require complete 100-label EUR-LEX")
    info=check_export(args.model)
    if info["objective"]!="bce":
        raise ValueError("freeze a BCE checkpoint, NOT a structured checkpoint")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA without GPU")
    if args.dtype=="bfloat16" and args.device=="cpu":
        raise ValueError("use float32 for CPU extraction")
    trainer,tokenizer,labels,label_ids,prompts,lengths,batches=_prepare_prompts(args,rows)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    cls=Qwen3_5ForConditionalGeneration if config.model_type=="qwen3_5" else Qwen3_5ForCausalLM
    base,missing=cls.from_pretrained(args.model,dtype=getattr(torch,args.dtype),
                                     attn_implementation="sdpa",local_files_only=True,
                                     output_loading_info=True)
    if missing.get("missing_keys") or missing.get("mismatched_keys"):
        raise RuntimeError("frozen BCE checkpoint lacks required model weights")
    base.requires_grad_(False)
    base=base.to(args.device).eval()
    model=getattr(base.model,"language_model",base.model)
    label_tensor=torch.tensor(label_ids[:100],dtype=torch.long,device=args.device)
    weight=base.get_output_embeddings().weight.index_select(0,label_tensor)
    hidden,score_data,logz_data=[None]*len(rows),[None]*len(rows),[None]*len(rows)
    with torch.inference_mode():
        for batch in batches:
            encoded=tokenizer([prompts[i] for i in batch],padding=True,
                              truncation=False,return_tensors="pt",
                              add_special_tokens=False)
            validate_right_padding(encoded["attention_mask"].tolist(),
                                   [lengths[i] for i in batch])
            inputs={key:t.to(args.device) for key,t in encoded.items()}
            states=model(**inputs,use_cache=False,return_dict=True).last_hidden_state
            lengths_t=inputs["attention_mask"].sum(-1)-1
            slot=states[torch.arange(len(batch),device=args.device),lengths_t]
            scores=F.linear(slot,weight).float()
            for index,position in enumerate(batch):
                values=scores[index].cpu().tolist()
                hidden[position]=slot[index].float().cpu()
                score_data[position]=scores[index].cpu()
                logz_data[position]=torch.tensor(centered_count_logz(values),
                                                  dtype=torch.float32)
    if any(x is None for x in hidden+score_data+logz_data):
        raise RuntimeError("feature extraction did not cover every row")
    cache={
        "schema":SCHEMA,"split":args.split,
        "data_sha256":digest(args.gold),
        "checkpoint_identity":str(args.model.resolve()),
        "model_config_sha256":digest(args.model/"config.json"),
        "label_token_ids":label_ids[:100],
        "hidden":torch.stack(hidden).to(dtype=torch.float32),
        "scores":torch.stack(score_data).float(),
        "logz":torch.stack(logz_data).float(),
        "ids":[r["id"] for r in rows],
        "input_digests":[input_digest(r) for r in rows],
        "label_counts":[sum(r["target"]) for r in rows],
        "input_tokens":lengths,
        "parameter_weights_sha256_checked":False,
    }
    verify_cache(cache,rows,split=args.split,data_sha=digest(args.gold))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,args.output)
    print(json.dumps({"status":"FROZEN_BCE_FEATURES_EXPORTED",
                      "split":args.split,"n":len(rows),
                      "k":100,"hidden_size":int(cache["hidden"].shape[1]),
                      "cache":str(args.output),"test_read":False},indent=2))


def _eval_count_nll(head,features,logz,gold,mode,device):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        output=head(features.to(device))
        if mode=="anchor":
            output=output+logz.to(device)
        nll=F.cross_entropy(output,gold.to(device),reduction="mean")
        return nll.item()


def train(args):
    import torch
    import torch.nn.functional as F
    if args.mode not in ("anchor","independent"):
        raise ValueError("frozen training requires anchor or independent mode")
    if args.output.exists():
        raise ValueError("output directory already exists")
    if args.steps<1 or args.batch<1 or args.eval_every<1 or not 0<args.lr<1:
        raise ValueError("invalid training hyperparameters")
    if args.init=="prior" and args.mode!="independent":
        raise ValueError("train-prior initialization is an independent-count baseline only")
    train_rows=read_dataset(args.train_gold,expected_split="train")
    dev_rows=read_dataset(args.dev_gold,expected_split="dev")
    tcache,dcache=load_cache(args.train_cache),load_cache(args.dev_cache)
    validate_pair(tcache,dcache,train_rows,dev_rows,
                  digest(args.train_gold),digest(args.dev_gold))
    torch.manual_seed(args.seed)
    hidden_t=tcache["hidden"].float()
    hidden_d=dcache["hidden"].float()
    mu,scale=_feature_norm(hidden_t,hidden_d)
    X_t=(hidden_t-mu)/scale
    X_d=(hidden_d-mu)/scale
    m_t=torch.tensor(tcache["label_counts"],dtype=torch.long)
    m_d=torch.tensor(dcache["label_counts"],dtype=torch.long)
    z_t=tcache["logz"].float()
    z_d=dcache["logz"].float()
    if any(row["allow_empty"] is False for row in train_rows+dev_rows):
        # The current EUR-LEX protocol uses all counts 0..100.
        raise ValueError("this code requires the same public all-count support for EUR-LEX")
    d=X_t.shape[1]
    head=torch.nn.Linear(d,101).float()
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    if args.init=="prior":
        histogram=torch.bincount(m_t,minlength=101).float()+1.0
        head.bias.data.copy_(torch.log(histogram/histogram.sum()))
    device=torch.device(args.device)
    head=head.to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=0.0)
    rng=random.Random(args.seed)
    indices=list(range(len(X_t)))
    position=len(indices)
    history=[]
    best_nll=math.inf
    best_state=None
    best_step=None
    initial_nll=_eval_count_nll(head,X_d,z_d,m_d,args.mode,device)
    for step in range(1,args.steps+1):
        if position+args.batch>len(indices):
            rng.shuffle(indices)
            position=0
        selection=indices[position:position+args.batch]
        position+=args.batch
        if not selection:
            raise RuntimeError("empty training batch")
        values=head(X_t[selection].to(device))
        if args.mode=="anchor":
            values=values+z_t[selection].to(device)
        loss=F.cross_entropy(values,m_t[selection].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite count loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),max_norm=5.0)
        optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            nll=_eval_count_nll(head,X_d,z_d,m_d,args.mode,device)
            history.append({"step":step,"train_batch_count_nll":float(loss.detach().cpu()),
                            "dev_count_nll":nll})
            if nll<best_nll:
                best_nll=nll
                best_step=step
                best_state={name:val.detach().float().cpu().clone()
                            for name,val in head.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validated trained count head")
    args.output.mkdir(parents=True)
    torch.save({"weight":best_state["weight"],"bias":best_state["bias"],
                "norm_mean":mu,"norm_scale":scale},args.output/"head.pt")
    meta={
        "schema":"eurlex_frozen_count_head_v1","mode":args.mode,
        "init":args.init,"seed":args.seed,"lr":args.lr,
        "steps":args.steps,"batch":args.batch,"eval_every":args.eval_every,
        "best_step":best_step,"best_dev_count_nll":best_nll,
        "initial_dev_count_nll":initial_nll,
        "train_gold_sha256":digest(args.train_gold),
        "dev_gold_sha256":digest(args.dev_gold),
        "train_cache_sha256":digest(args.train_cache),
        "dev_cache_sha256":digest(args.dev_cache),
        "checkpoint_identity":tcache["checkpoint_identity"],
        "model_config_sha256":tcache["model_config_sha256"],
        "frozen_backbone_and_candidate_scores":True,
        "no_test_access":True,
        "head_file_sha256":digest(args.output/"head.pt"),
        "criterion":"proper count negative log likelihood on DEV",
        "history":history,
    }
    (args.output/"training_summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"status":"FROZEN_COUNT_HEAD_TRAINED",
                      "mode":args.mode,"best_step":best_step,
                      "initial_dev_count_nll":initial_nll,
                      "best_dev_count_nll":best_nll,
                      "output":str(args.output)},indent=2))


def predict(args):
    import torch
    from harness.set_distillation import input_digest
    from paper_static.eurlex_metrics import score as official_score
    if args.output.exists():
        raise ValueError("output already exists; do not replace predictions")
    if args.mode not in MODES:
        raise ValueError("unsupported prediction mode")
    rows=read_dataset(args.gold,expected_split="dev")
    cache=load_cache(args.cache)
    verify_cache(cache,rows,split="dev",data_sha=digest(args.gold))
    if args.mode=="bce":
        output_head=None
        mode="bce"
        identifier=f"paper1|eurlex|frozen|bce|{cache['checkpoint_identity']}"
    else:
        if args.head is None:
            raise ValueError("--head directory required for count modes")
        meta=json.loads((args.head/"training_summary.json").read_text())
        if meta.get("mode")!=args.mode:
            raise ValueError("head mode mismatch")
        if meta.get("checkpoint_identity")!=cache["checkpoint_identity"] or \
           meta.get("model_config_sha256")!=cache["model_config_sha256"]:
            raise ValueError("head was trained against a different frozen BCE model")
        if meta.get("dev_gold_sha256")!=digest(args.gold) or \
           meta.get("dev_cache_sha256")!=digest(args.cache):
            raise ValueError("head DEV feature cache identity mismatch")
        if meta.get("head_file_sha256")!=digest(args.head/"head.pt"):
            raise ValueError("count head changed after training")
        output_head=torch.load(args.head/"head.pt",weights_only=True,map_location="cpu")
        if tuple(output_head["weight"].shape)!=(101,cache["hidden"].shape[1]):
            raise ValueError("wrong count-head dimensions")
        mode=args.mode
        identifier=f"paper1|eurlex|frozen|{mode}|seed_{meta['seed']}|step_{meta['best_step']}"
    predictions=[]
    for i,row in enumerate(rows):
        scores=cache["scores"][i].tolist()
        if mode=="bce":
            p=[1/(1+math.exp(-s)) if s>=0 else math.exp(s)/(1+math.exp(s))
               for s in scores]
            item={"p":p,"logits":scores}
        else:
            h=cache["hidden"][i].float()
            v=(h-output_head["norm_mean"])/output_head["norm_scale"]
            raw=(output_head["weight"]@v+output_head["bias"]).tolist()
            item=structured_prediction(scores,raw,row,count_anchor=(mode=="anchor"))
            item["candidate_logits"]=scores
            item["count_residual_logits"]=raw
        item.update({"id":row["id"],"option_keys":[o["key"] for o in row["options"]],
                     "input_digest":input_digest(row),
                     "checkpoint_id":identifier,
                     "provenance":{
                         "frozen_bce_checkpoint":cache["checkpoint_identity"],
                         "model_config_sha256":cache["model_config_sha256"],
                         "no_qwen_forward_at_prediction":True,
                         "training_model_backbone_frozen":True,
                     }})
        predictions.append(item)
    metrics=official_score(rows,predictions,threshold=.5)
    args.output.mkdir(parents=True)
    with (args.output/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
    (args.output/"eurlex_paper_metrics.json").write_text(
        json.dumps(metrics,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (args.output/"prediction_provenance.json").write_text(
        json.dumps({"status":"DEV_FROZEN_BCE_SCORE_HEAD_ONLY",
                    "mode":mode,
                    "gold_sha256":digest(args.gold),
                    "cache_sha256":digest(args.cache),
                    "head":str(args.head) if args.head else None,
                    "checkpoint_identity":identifier},indent=2)+"\n")
    print(json.dumps({"status":"DEV_FROZEN_BCE_SCORE_HEAD_ONLY",
                      "mode":mode,"count_nll":metrics["count_nll"],
                      "proper_joint_set_nll":metrics["proper_joint_set_nll"],
                      "micro_f1_map":metrics["official_micro_f1"],
                      "predictions":str(args.output/"predictions.jsonl")},indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("extract",help="frozen Qwen forward once per DEV/TRAIN sample")
    a.add_argument("--gold",type=Path,required=True)
    a.add_argument("--split",choices=["train","dev"],required=True)
    a.add_argument("--model",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    a.add_argument("--dtype",choices=["float32","bfloat16"],default="bfloat16")
    a.add_argument("--batch-size",type=int,default=1)
    a.add_argument("--max-input-tokens",type=int,default=8192)
    a.add_argument("--max-batch-tokens",type=int,default=8192)
    b=sub.add_parser("train",help="fit frozen-feature CountHead on CPU or GPU")
    b.add_argument("--train-gold",type=Path,required=True)
    b.add_argument("--dev-gold",type=Path,required=True)
    b.add_argument("--train-cache",type=Path,required=True)
    b.add_argument("--dev-cache",type=Path,required=True)
    b.add_argument("--mode",choices=["anchor","independent"],required=True)
    b.add_argument("--init",choices=["zero","prior"],default="zero")
    b.add_argument("--seed",type=int,default=43)
    b.add_argument("--steps",type=int,default=600)
    b.add_argument("--batch",type=int,default=64)
    b.add_argument("--eval-every",type=int,default=50)
    b.add_argument("--lr",type=float,default=1e-3)
    b.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    b.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("predict",help="replay saved frozen features through trained count head")
    c.add_argument("--gold",type=Path,required=True)
    c.add_argument("--cache",type=Path,required=True)
    c.add_argument("--mode",choices=MODES,required=True)
    c.add_argument("--head",type=Path)
    c.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    {"extract":extract,"train":train,"predict":predict}[args.action](args)


if __name__=="__main__":
    main()
