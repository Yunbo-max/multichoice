#!/usr/bin/env python3
"""Stage exactly two one-GPU P1 frozen-score extraction jobs: Pilot Train + DEV.

Both use ONE standalone BCE checkpoint, identical Qwen3.5 native readout,
100 answer-slot identities and the original EUR-LEX prompt. Head training
then runs on saved features on CPU. No Test, no destructive deletion,
no automatic QZ submission. For full 55k, plan chunked resumable inference
separately; do not pretend a 55k batch fits a 3h single-GPU job.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex


def payloads(template,manifest, *,root,data,model,output,batch_size=1):
    if manifest.get("dataset")!="coastalcph/lex_glue" or \
       manifest.get("test_loaded") or len(manifest.get("label_ids",[]))!=100:
        raise ValueError("Pilot requires verified TRAIN/DEV-only EUR-LEX manifest")
    if not 1<=batch_size<=8:
        raise ValueError("frozen readout extraction batch must be 1..8")
    root,data,model,output=map(Path,(root,data,model,output))
    if any(not path.is_absolute() for path in (root,data,model,output)):
        raise ValueError("remote QZ paths must be absolute")
    entries=[]
    for split,reference in (("train","train"),("dev","dev")):
        hashes=manifest.get("pilot_files",{}).get(reference,{})
        if not isinstance(hashes.get("sha256"),str) or len(hashes["sha256"])!=64:
            raise ValueError("Pilot data SHA256 missing")
        gold=data/"pilot"/f"{split}.jsonl"
        dest=output/f"{split}.pt"
        log=root/"logs/qz/eurlex"/f"p1-frozen-extract-{split}.log"
        verify=(
            "import pathlib,hashlib,json;"
            f"manifest=json.loads(pathlib.Path({str(data/'manifest.json')!r}).read_text());"
            "assert not manifest['test_loaded'];"
            "sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();"
            f"assert sha({str(gold)!r})=={hashes['sha256']!r};"
        )
        exe=root/".venv-sii/bin/python"
        args=[str(exe),"-m","paper_static.eurlex_frozen_count","extract",
              "--gold",str(gold),"--split",split,
              "--model",str(model),"--output",str(dest),
              "--device","cuda","--dtype","bfloat16",
              "--batch-size",str(batch_size),
              "--max-input-tokens",str(manifest["max_input_tokens"]),
              "--max-batch-tokens",str(manifest["max_input_tokens"])]
        lines=["set -euo pipefail",
               "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
               "export PYTHONPATH="+shlex.quote(str(root)),
               "test -s "+shlex.quote(str(gold)),
               "test -s "+shlex.quote(str(model/"config.json")),
               "test ! -e "+shlex.quote(str(dest)),
               "mkdir -p "+shlex.quote(str(dest.parent)),
               "mkdir -p "+shlex.quote(str(log.parent)),
               "cd "+shlex.quote(str(root)),
               shlex.quote(str(exe))+" -c "+shlex.quote(verify),
               shlex.join(args)+" > "+shlex.quote(str(log))+" 2>&1"]
        if any(x in "\n".join(lines) for x in ("rm -rf","/test.jsonl","--unlock-test")):
            raise ValueError("unsafe extraction payload")
        payload=dict(template)
        payload["name"]=f"p1-frozen-bce-extract-{split}"
        payload["description"]="P1 one-forward frozen BCE head feature export (DEV/TRAIN only), UNRUN"
        payload["command"]="\n".join(lines)+"\n"
        entries.append(payload)
    return entries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template",type=Path,default=Path("qz_c64/c64-warm-bce.json"))
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--bce-checkpoint",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=1)
    args=p.parse_args(argv)
    if args.output_dir.exists():
        p.error("refusing to overwrite staged jobs")
    tpl=json.loads(args.template.read_text())
    manifest=json.loads(args.manifest.read_text())
    jobs=payloads(tpl,manifest,root=args.repo_root,data=args.data_root,
                  model=args.bce_checkpoint,output=args.cache_root,
                  batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True)
    for job in jobs:
        (args.output_dir/(job["name"]+".json")).write_text(json.dumps(job,indent=2)+"\n")
    print(json.dumps({"status":"STAGED_NOT_SUBMITTED","jobs":[j["name"] for j in jobs]},
                     indent=2))


if __name__=="__main__":
    main()
