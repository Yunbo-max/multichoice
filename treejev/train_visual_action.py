#!/usr/bin/env python3
"""Stage 1: Qwen3.5-4B visual action SFT on Valen tied-optimal action distributions.

Actual next-token restricted-softmax *soft* CE; never accesses answer labels in
the prompt, symbolic states, search distances, official eval games or oracle.
Requires torch, transformers, peft, pillow; run on GPU with fresh directories.
No claim that this code or the VLM pathway has already passed GPU validation.
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from treejev.data import ACTION_ORDER, read_jsonl

PROMPT = (
    "You are playing Sokoban. Walls block movement. Push boxes onto targets. "
    "You cannot pull boxes or push two boxes at once. "
    "Choose one best next move for the pictured board.\n"
    "Options: (A) up; (B) right; (C) down; (D) left.\n"
    "Output only the option letter."
)

def get_rows(filename, limit=0):
    rows=list(read_jsonl(filename))
    if limit: rows=rows[:limit]
    if not rows: raise ValueError(f"empty {filename}")
    if any(r.get("source")!="valen_sokoban" or r.get("input_modality")!="image" for r in rows):
        raise ValueError("expected Valen image manifest rows from treejev.prepare")
    if any(not Path(r["image_path"]).is_file() for r in rows):
        raise FileNotFoundError("missing image; extract assets.zip from dataset")
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("duplicate input rows")
    return rows

def _build_batch_features(processor, rows):
    from PIL import Image
    import torch
    # Only the visible image and public game rules enter this forward.
    messages=[]
    for row in rows:
        with Image.open(row["image_path"]) as raw:
            image=raw.convert("RGB")
        messages.append([{"role":"user","content":[
            {"type":"image","image":image},{"type":"text","text":PROMPT}]}])
    # Follow the official Qwen3.5 multimodal processor path. The PIL image
    # becomes pixel_values and image_grid_thw within the returned batch.
    features=processor.apply_chat_template(
        messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors="pt",enable_thinking=False,
        padding=True)
    # Append a known answer cue after the multimodal template so that next
    # token logits score A/B/C/D directly without generating textual CoT.
    cue=processor.tokenizer("The correct answer is (",
            add_special_tokens=False,return_tensors="pt")["input_ids"]
    cue=cue.expand(features["input_ids"].shape[0],-1)
    features["input_ids"]=torch.cat((features["input_ids"],cue),dim=1)
    features["attention_mask"]=torch.cat((features["attention_mask"],
                             torch.ones_like(cue)),dim=1)
    if "mm_token_type_ids" in features:
        features["mm_token_type_ids"]=torch.cat((features["mm_token_type_ids"],
                             torch.zeros_like(cue)),dim=1)
    return features

def _forward_features(model, processor, features, device, *, bf16):
    import torch
    features={k:(v.to(device=device,dtype=torch.bfloat16 if bf16 else torch.float32)
                 if v.is_floating_point() else v.to(device)) for k,v in features.items()}
    logits=model(**features,use_cache=False).logits  # [batch, seq, vocab]
    last_idx=features["attention_mask"].sum(dim=1)-1  # last valid token per row
    batch_idx=torch.arange(logits.shape[0],device=logits.device)
    last_logits=logits[batch_idx,last_idx]  # [batch, vocab]
    token_ids=[]
    for letter in "ABCD":
        codes=processor.tokenizer.encode(letter,add_special_tokens=False)
        if len(codes)!=1:
            raise ValueError(f"non-single-token answer letter: {letter}")
        token_ids.append(codes[0])
    token_ids=torch.tensor(token_ids,device=logits.device)
    return last_logits[:,token_ids].float()  # [batch, 4]

def first_token_logits_batch(model, processor, rows, device, *, bf16):
    features=_build_batch_features(processor,rows)
    return _forward_features(model,processor,features,device,bf16=bf16)

class FeatureCache:
    """Load pre-cached processor features so training skips CPU preprocessing."""
    def __init__(self, cache_dir):
        import json
        self.dir=Path(cache_dir)
        manifest=json.loads((self.dir/"manifest.json").read_text())
        self.chunks=[]
        self.row_index=[]  # row position -> (chunk_idx, offset_in_chunk)
        for ci,m in enumerate(manifest):
            self.chunks.append((self.dir/m["chunk"], m["n"]))
            for oi in range(m["n"]):
                self.row_index.append((ci,oi))
        self.loaded={}

    def _chunk(self, ci):
        import torch
        if ci not in self.loaded:
            path,n=self.chunks[ci]
            self.loaded[ci]=torch.load(path,map_location="cpu",weights_only=False)
        return self.loaded[ci]

    def get(self, indices):
        import torch
        groups={}
        for idx in indices:
            ci,oi=self.row_index[idx]
            groups.setdefault(ci,[]).append(oi)
        pixel_values=[];grid_thw=[];input_ids=[];attention_mask=[];mm=[]
        for ci,offs in groups.items():
            c=self._chunk(ci)
            for oi in offs:
                pixel_values.append(c["pixel_values"][oi])
                grid_thw.append(c["image_grid_thw"][oi])
                input_ids.append(c["input_ids"][oi])
                attention_mask.append(c["attention_mask"][oi])
                mm.append(c["mm_token_type_ids"][oi])
        # re-pad text tensors to the batch max length (they were padded per-chunk)
        max_len=max(t.shape[0] for t in input_ids)
        pad=lambda t: torch.nn.functional.pad(t,(0,max_len-t.shape[0]),value=0)
        return {
            "pixel_values": torch.cat(pixel_values,dim=0),  # [total_patches, 1536]
            "image_grid_thw": torch.stack(grid_thw),        # [n, 3]
            "input_ids": torch.stack([pad(t) for t in input_ids]),
            "attention_mask": torch.stack([pad(t) for t in attention_mask]),
            "mm_token_type_ids": torch.stack([pad(t) for t in mm]),
        }

def action_loss_batch(logits, rows):
    import torch
    import torch.nn.functional as F
    targets=torch.tensor([[r["target_probs"][a] for a in ACTION_ORDER] for r in rows],
                         device=logits.device,dtype=torch.float32)
    if not torch.isfinite(targets).all() or torch.any(torch.abs(targets.sum(-1)-1)>1e-5):
        raise ValueError("bad soft action label")
    return -(targets*F.log_softmax(logits,dim=-1)).sum(-1).mean()

def evaluate(model, processor, rows, device, bf16, batch_size=8):
    import torch
    model.eval()
    loss_sum=0.; hits=0; conf_sum=0.
    with torch.no_grad():
        for i in range(0,len(rows),batch_size):
            batch_rows=rows[i:i+batch_size]
            logits=first_token_logits_batch(model,processor,batch_rows,device,bf16=bf16)
            probs=logits.softmax(-1)
            for j,row in enumerate(batch_rows):
                loss_sum+=float(-(torch.tensor(
                    [row["target_probs"][a] for a in ACTION_ORDER],
                    device=logits.device,dtype=torch.float32)
                    *torch.log_softmax(logits[j],dim=-1)).sum().item())
                picked=ACTION_ORDER[int(torch.argmax(probs[j]))]
                hits+=int(row["target_probs"][picked]>0)
                conf_sum+=float(sum(probs[j][k].item() for k,a in enumerate(ACTION_ORDER)
                                    if row["target_probs"][a]>0))
    return {"action_optimal_accuracy":hits/len(rows),
            "soft_cross_entropy":loss_sum/len(rows),
            "mass_on_optimal_actions":conf_sum/len(rows),"n":len(rows)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3.5-4B")
    p.add_argument("--modules",type=Path,default=Path("innerjev/configs/4B/modules.json"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--lora-r",type=int,default=16)
    p.add_argument("--lora-alpha",type=int,default=32)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--grad-accum",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--dev-limit",type=int,default=64)
    p.add_argument("--max-steps",type=int,default=250)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--local-files-only",action="store_true")
    p.add_argument("--cache-dir",type=Path,default=None,
                   help="pre-cached processor features from treejev.cache_valen_features")
    args=p.parse_args()
    if args.grad_accum<1 or args.epochs<1 or args.max_steps<1 or args.batch_size<1:
        p.error("invalid accumulation/epoch/steps/batch")
    if args.output.exists():
        p.error(f"refusing to overwrite {args.output}")
    if "eval_sokoban" in args.train.parts or "eval_sokoban" in args.dev.parts:
        p.error("never optimize on official Valen test")
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("Stage1 VLM LoRA needs CUDA; CPU-only tests use treejev oracle")
    local_rank=int(os.environ.get("LOCAL_RANK",-1))
    world_size=int(os.environ.get("WORLD_SIZE",1))
    distributed=local_rank>=0 and world_size>1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device=f"cuda:{local_rank}" if distributed else "cuda"
    random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train=get_rows(args.train,args.limit);dev=get_rows(args.dev,args.dev_limit)
    if distributed:
        train=train[local_rank::world_size]
    if {r["group_id"] for r in train}&{r["group_id"] for r in dev}:
        raise ValueError("train/dev board leakage")
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=args.local_files_only)
    qwen=Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,torch_dtype=torch.bfloat16,local_files_only=args.local_files_only)
    targets=json.loads(args.modules.read_text(encoding="utf-8"))
    available=set(dict(qwen.named_modules()))
    absent=[m for m in targets if m not in available]
    if absent: raise ValueError(f"module manifest does not match VLM: {absent[:5]}")
    model=get_peft_model(qwen,LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,
                                    target_modules=targets,lora_dropout=0,bias="none"))
    model.to(device);model.config.use_cache=False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant":False})
    if distributed:
        model=DDP(model,device_ids=[local_rank],find_unused_parameters=False)
    params=[v for v in model.parameters() if v.requires_grad]
    if not params: raise RuntimeError("no LoRA parameters")
    optim=torch.optim.AdamW(params,lr=args.lr)
    if not distributed or local_rank==0:
        args.output.mkdir(parents=True)
    history=[];best=(-1.,float("inf"));steps=0
    is_main=not distributed or local_rank==0
    cache=FeatureCache(args.cache_dir) if args.cache_dir else None
    for epoch in range(args.epochs):
        model.train()
        indices=list(range(len(train)))
        random.Random(args.seed+epoch).shuffle(indices)
        optim.zero_grad(set_to_none=True);total_loss=0.;pending=0
        for i in range(0,len(indices),args.batch_size):
            batch_idx=indices[i:i+args.batch_size]
            batch_rows=[train[k] for k in batch_idx]
            if cache is not None:
                features=cache.get(batch_idx)
                logits=_forward_features(model,processor,features,device,bf16=True)
            else:
                logits=first_token_logits_batch(model,processor,batch_rows,device,bf16=True)
            loss=action_loss_batch(logits,batch_rows)
            if not torch.isfinite(loss).item(): raise FloatingPointError("nonfinite action loss")
            (loss/args.grad_accum).backward()
            if steps==0 and pending==0 and not any(p.grad is not None and
                torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0
                for p in params):
                raise RuntimeError("LoRA adapter received no nonzero gradient")
            pending+=1;total_loss+=float(loss.detach())
            if pending>=args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params,1.0)
                optim.step();optim.zero_grad(set_to_none=True)
                pending=0;steps+=1
                if steps>=args.max_steps: break
        if pending:
            # Rescale the short final accumulation to match full group average.
            scale=args.grad_accum/pending
            for p_ in params:
                if p_.grad is not None:p_.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optim.step();optim.zero_grad(set_to_none=True);steps+=1
        if is_main:
            underlying=model.module if distributed else model
            metrics=evaluate(underlying,processor,dev,device,True)
            metrics.update({"epoch":epoch+1,"steps":steps,"train_loss_sum":total_loss})
            history.append(metrics)
            key=(metrics["action_optimal_accuracy"],metrics["soft_cross_entropy"])
            if key[0]>best[0] or (key[0]==best[0] and key[1]<best[1]):
                best=key;folder=args.output/"adapter_best"
                if folder.exists():shutil.rmtree(folder)
                underlying.save_pretrained(folder,safe_serialization=True)
                processor.save_pretrained(folder)
                (args.output/"best.json").write_text(json.dumps(metrics,indent=2)+"\n")
        if steps>=args.max_steps:break
    if is_main:
        source_bytes=args.modules.read_bytes()
        (args.output/"training.json").write_text(json.dumps({
          "status":"FINISHED","train":str(args.train),"dev":str(args.dev),
          "model":args.model,"model_weight_hash_verified":False,
          "module_manifest_sha256":hashlib.sha256(source_bytes).hexdigest(),
          "max_steps":args.max_steps,"optimizer_steps":steps,
          "world_size":world_size,
          "history":history,"best_dev":best,
          "official_test_touched":False,
          "note":"Check exact base model and adapter reload before held-out evaluation."
        },indent=2)+"\n")
        print(json.dumps({"training_steps":steps,"best_dev":best,
                          "checkpoint":str(args.output/"adapter_best")},indent=2))
    if distributed:
        dist.destroy_process_group()

if __name__=="__main__":
    main()
