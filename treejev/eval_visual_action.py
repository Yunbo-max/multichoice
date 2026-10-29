#!/usr/bin/env python3
"""Re-evaluate an existing Qwen3.5-4B visual LoRA adapter on ALL Valen DEV rows.

Never uses official eval_sokoban. Mixed-batch/singleton agreement is a blocking
preflight for image-padding correctness; metrics are saved for provenance.
This does not modify the trained adapter or create new training outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1 << 20),b""):
            h.update(chunk)
    return h.hexdigest()


def eval_logits(model,processor,rows,device,batch_size):
    import torch
    from treejev.train_visual_action import first_token_logits_batch
    results=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            part=rows[start:start+batch_size]
            x=first_token_logits_batch(model,processor,part,device,bf16=True)
            results.append(x.detach().cpu())
            if (start//batch_size)%25==0:
                print(f"evaluated {start+len(part)}/{len(rows)}",flush=True)
    return torch.cat(results,dim=0)


def check_batch_invariance(model,processor,rows,device,n=8,max_prob_diff=0.05):
    import torch
    from PIL import Image
    if n<2:return {"status":"SKIPPED","n":0}
    # Cover different source image sizes to stress mixed-length multimodal pads.
    stride=max(1,len(rows)//64)
    candidates=rows[::stride][:64]
    dims=[]
    for row in candidates:
        with Image.open(row["image_path"]) as f:
            dims.append((f.width*f.height,row))
    ranked=sorted(dims,key=lambda x:x[0])
    lo=ranked[:max(1,n//2)]
    hi=ranked[-max(1,n-len(lo)):]
    selected=[r for _,r in lo+hi]
    with torch.inference_mode():
        batch=eval_logits(model,processor,selected,device,len(selected))
        single=eval_logits(model,processor,selected,device,1)
    p1=batch.softmax(dim=-1)
    p2=single.softmax(dim=-1)
    diff=(p1-p2).abs().max().item()
    flips=int((batch.argmax(dim=-1)!=single.argmax(dim=-1)).sum())
    report={"status":"PASS" if diff<=max_prob_diff and flips==0 else "FAIL",
            "n":len(selected),"max_abs_prob_delta":diff,
            "argmax_disagreements":flips,
            "threshold_max_probability_delta":max_prob_diff}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--train-manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--batch-check",type=int,default=8)
    p.add_argument("--max-prob-diff",type=float,default=.05)
    p.add_argument("--limit",type=int,default=0,
                   help="0 means full DEV; nonzero is smoke only")
    args=p.parse_args()
    if args.output.exists():p.error(f"refusing to overwrite {args.output}")
    if args.batch_size<1 or args.batch_check<0 or args.limit<0 or args.max_prob_diff<0:
        p.error("invalid batch settings")
    if "eval_sokoban" in args.dev.parts:
        p.error("official eval_sokoban is locked TEST and may not be used here")
    if not args.adapter.is_dir() or not (args.adapter/"adapter_config.json").is_file():
        p.error("PEFT adapter_config.json is missing; use a real saved LoRA adapter")
    import torch
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from treejev.train_visual_action import get_rows
    from treejev.visual_eval_metrics import summarize_visual_actions
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen4B visual DEV eval needs CUDA")
    rows=get_rows(args.dev,args.limit)
    if args.train_manifest:
        from treejev.data import read_jsonl
        train_groups={r["group_id"] for r in read_jsonl(args.train_manifest)}
        overlap=train_groups & {r["group_id"] for r in rows}
        if overlap:raise ValueError(f"train/dev layout group leakage: {len(overlap)}")
    processor=AutoProcessor.from_pretrained(args.base_model,local_files_only=True)
    base=Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,torch_dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,str(args.adapter),is_trainable=False)
    model.to("cuda");model.eval()
    batch_report=check_batch_invariance(model,processor,rows,"cuda",
                                        n=args.batch_check,
                                        max_prob_diff=args.max_prob_diff)
    if batch_report["status"]=="FAIL":
        raise RuntimeError(f"batch=1/8 disagreement: {batch_report}. "
                           "Refusing to report corrupted model scores.")
    logits=eval_logits(model,processor,rows,"cuda",args.batch_size)
    metrics=summarize_visual_actions(rows,logits)
    metadata={
      "status":"COMPLETE","checkpoint_kind":"pre_fix_training_post_fix_evaluation",
      "data_role":"DEV_NOT_TEST","dev_sha256":sha256_file(args.dev),
      "train_manifest_sha256":sha256_file(args.train_manifest)
          if args.train_manifest else None,
      "adapter_config_sha256":sha256_file(args.adapter/"adapter_config.json"),
      "base_model":str(args.base_model),"adapter":str(args.adapter),
      "dev":str(args.dev),"batch_size":args.batch_size,
      "full_dev":args.limit==0,"batch_check":batch_report,
      "official_test_touched":False,"metrics":metrics,
      "warning":"Metrics use existing trained weights, not new post-fix retraining."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,
                                      allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"metrics":metrics,
                      "batch_check":batch_report},indent=2))

if __name__=="__main__":
    main()
