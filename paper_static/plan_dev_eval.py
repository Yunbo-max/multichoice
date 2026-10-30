#!/usr/bin/env python3
"""Regenerate six safe QZ DEV inference payloads from existing frozen TEST templates.

No GPU submission, training, test-set access, or destructive cleanup.
The source templates already specify local checkpoints and QZ workspace IDs.
Run this from the Paper 1 branch after checking out the same checkpoints.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ARMS=("bce_continue","set_ce_anchor")
BRANCH="research/static-setjev-paper-20261027"


def convert(template,arm,seed):
    cmd=template.get("command")
    if not isinstance(cmd,str):
        raise ValueError("QZ template has no command")
    orig=f"paper1-eval-{arm}-s{seed}"
    if template.get("name")!=orig:
        raise ValueError("unexpected arm/seed in QZ template")
    if "--expected-split test" not in cmd or "/test.jsonl" not in cmd:
        raise ValueError("source template must evaluate test before conversion")
    old_path_match=re.search(r'--output\s+"([^"]+)"',cmd)
    if not old_path_match:
        raise ValueError("missing explicit --output path")
    old_out=old_path_match.group(1)
    if "/runs/paper_static_eval/" not in old_out:
        raise ValueError("unexpected original output root")
    new_out=old_out.replace("/runs/paper_static_eval/","/runs/paper_static_dev_eval/")
    expected_rm=f'rm -rf "{old_out}"'
    if cmd.count("rm -rf")!=1 or expected_rm not in cmd:
        raise ValueError("unexpected or unrecognized destructive QZ command")
    cmd=cmd.replace("set -uo pipefail", "set -euo pipefail", 1)
    if not cmd.startswith("set -euo pipefail"):
        raise ValueError("QZ job must stop on failed branch/output preflight")
    cmd=cmd.replace(expected_rm,f'test ! -e "{new_out}"')
    cmd=cmd.replace("--expected-split test","--expected-split dev")
    cmd=cmd.replace("/data/multiselect_grouped_multi/test.jsonl",
                    "/data/multiselect_grouped_multi/dev.jsonl")
    cmd=cmd.replace(old_out,new_out)
    cmd=cmd.replace("/runs/paper_static_eval\"","/runs/paper_static_dev_eval\"")
    cmd=cmd.replace(f"{orig}.log",f"paper1-dev-{arm}-s{seed}.log")
    if "rm -rf" in cmd or "--expected-split test" in cmd or "/test.jsonl" in cmd:
        raise ValueError("unsafe conversion produced test/destructive command")
    # Banish stale branch inference when a cluster checkout serves both papers.
    gate='test "$(git rev-parse --abbrev-ref HEAD)" = "'+BRANCH+'"'
    if '\ncd "' not in cmd:
        raise ValueError("could not find project checkout invocation")
    cmd=re.sub(r'(?m)^(cd "[^"]+")$',r'\1\n'+gate,cmd,count=1)
    if gate not in cmd:
        raise ValueError("failed to inject branch preflight")
    new=dict(template)
    new["name"]=f"paper1-dev-{arm}-s{seed}"
    new["description"]="Paper 1 DEV-only saved predictions; frozen checkpoint; no TEST, no training"
    new["command"]=cmd
    return new


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template-root",type=Path,default=Path("qz_static_eval"))
    ap.add_argument("--output-dir",type=Path,default=Path("qz_static_count_dev"))
    ap.add_argument("--seeds",nargs="+",type=int,default=[43,44,45])
    args=ap.parse_args(argv)
    if len(set(args.seeds))!=len(args.seeds):
        ap.error("duplicate seeds")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    files=[]
    for seed in args.seeds:
        for arm in ARMS:
            source=args.template_root/f"paper1-eval-{arm}-s{seed}.json"
            payload=convert(json.loads(source.read_text(encoding="utf-8")),arm,seed)
            dest=args.output_dir/f"paper1-dev-{arm}-s{seed}.json"
            if dest.exists():
                ap.error(f"refusing overwrite: {dest}")
            dest.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            files.append(str(dest))
    summary={"status":"STAGED_NOT_SUBMITTED","branch":BRANCH,"jobs":files,
             "run_after_all_succeed":(
             "python -m paper_static.paired_count_audit "
             "--gold data/multiselect_grouped_multi/dev.jsonl "
             "--predictions-root runs/paper_static_dev_eval/qwen3.5-4b-bce-s44 "
             "--seeds 43 44 45 --bootstrap 1000 "
             "--output runs/paper1_count_audit/paired_count_dev.json "
             "--per-question runs/paper1_count_audit/paired_count_dev.jsonl")}
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
