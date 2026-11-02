#!/usr/bin/env python3
"""Read the ORIGINAL RoMQA CLOSED files and measure dataset suitability.

Official source: https://github.com/facebookresearch/romqa
Official files inside its third-party mirror archive:
  data/closed/top_20.train.json.bz2
  data/closed/top_20.dev.json.bz2
The "top_20" name refers to retrieval evidence, not necessarily 20 answer
candidates. This code reports the ACTUAL count; it does not assume 100.
Only TRAIN and DEV are ever opened. TEST is intentionally unsupported.

This is a DATA AUDIT, not a model result or a measured BCE count gap.
No candidate is inserted from gold, no per-example quota is leaked to
a model, and no benchmark choice is made using a sealed TEST set.

Usage:
 python -m paper_static.romqa_data_screen \
   --archive /path/to/romqa_data.zip --output reports/romqa_screen.json
 OR
 python -m paper_static.romqa_data_screen \
   --data-dir /path/to/romqa/data/closed --output reports/romqa_screen.json

Dependency-free: Python standard library only.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


SPLITS=("train","dev")
BASENAME={s:f"top_20.{s}.json.bz2" for s in SPLITS}
REQUIRED=("id","cluster_id","candidates","question")


def load_zip(archive):
    """Do not decompress/open any blind TEST file."""
    output={}
    with zipfile.ZipFile(archive) as z:
        files=z.namelist()
        for split in SPLITS:
            matches=[p for p in files if
                     p.replace("\\","/").endswith("/closed/"+BASENAME[split])]
            if len(matches)!=1:
                raise ValueError(f"zip must contain exactly one closed/{BASENAME[split]}")
            data=z.read(matches[0])
            try:
                obj=json.loads(bz2.decompress(data).decode("utf-8"))
            except (ValueError,EOFError,OSError,UnicodeDecodeError) as e:
                raise ValueError(f"corrupt {split} closed JSON/bzip2 data") from e
            output[split]=obj
    return output


def load_dir(data_dir):
    output={}
    for split in SPLITS:
        path=Path(data_dir)/BASENAME[split]
        if not path.is_file():
            raise ValueError(f"missing official RoMQA {split} dataset: {path}")
        try:
            with bz2.open(path,"rt",encoding="utf-8") as f:
                output[split]=json.load(f)
        except (ValueError,EOFError,OSError,UnicodeDecodeError) as e:
            raise ValueError(f"bad official closed dataset: {path}") from e
    return output


def _quantile(v,q):
    values=sorted(v)
    return values[min(len(values)-1,round(q*(len(values)-1)))]


def screen(splits):
    if set(splits)!=set(SPLITS):
        raise ValueError("ONLY train and dev are accepted: no TEST inspection")
    report={"status":"RO_MQA_TRAIN_DEV_DATA_ONLY_NO_MODEL_RESULTS",
            "dataset":"facebookresearch/romqa, closed setting",
            "splits":{},"issues":[],
            "test_opened":False}
    global_ids=set()
    cluster_owner={}
    for split in SPLITS:
        examples=splits[split]
        if not isinstance(examples,list) or not examples:
            raise ValueError(f"{split} must be a nonempty JSON list")
        ids=set()
        per_cluster=defaultdict(list)
        histogram=Counter()
        evidence_stats=Counter()
        candidate_lengths=[]
        counts=[]
        qlengths=[]
        out_of_candidates=[]
        duplicate_candidates=0
        constraints_lengths=[]
        nonzero=[]
        for i,ex in enumerate(examples):
            if not isinstance(ex,dict) or any(k not in ex for k in REQUIRED):
                raise ValueError(f"{split} row {i}: missing mandatory keys")
            key=str(ex["id"])
            cluster=str(ex["cluster_id"])
            if not key or key in ids or key in global_ids:
                raise ValueError(f"{split}: missing or duplicate sample ID {key}")
            if cluster in cluster_owner and cluster_owner[cluster]!=split:
                raise ValueError(f"cluster {cluster!r} leaks across train and DEV")
            cluster_owner[cluster]=split
            ids.add(key)
            candidates=ex["candidates"]
            if not isinstance(candidates,list) or not candidates:
                raise ValueError(f"{split}: empty or missing candidates on {key}")
            if not isinstance(ex["question"],str) or not ex["question"].strip():
                raise ValueError(f"{split}: invalid question")
            texts=[]
            missing_labels=0
            for c in candidates:
                if not isinstance(c,dict) or not isinstance(c.get("text"),str):
                    raise ValueError(f"{split}: candidate without text on {key}")
                if "is_answer" not in c or type(c["is_answer"]) is not bool:
                    missing_labels+=1
                texts.append(c["text"])
                if c.get("dpr"):
                    evidence_stats["has_dpr_candidates"]+=1
                if c.get("evidence"):
                    evidence_stats["has_oracle_evidence_candidates"]+=1
            if missing_labels:
                raise ValueError(f"{split}: unlabeled candidate is not valid for Train/DEV")
            if len(set(texts))!=len(texts):
                duplicate_candidates+=1
            count=sum(c["is_answer"] for c in candidates)
            counts.append(count)
            nonzero.append(count>0)
            histogram[count]+=1
            candidate_lengths.append(len(candidates))
            qlengths.append(len(ex["question"].split()))
            per_cluster[cluster].append(count)
            constraints=ex.get("constraints")
            if isinstance(constraints,list):
                constraints_lengths.append(len(constraints))
            complete=ex.get("complete_answer")
            if isinstance(complete,list) and complete:
                full={c["text"] for c in complete if isinstance(c,dict) and
                      isinstance(c.get("text"),str)}
                if full:
                    out_of_candidates.append(len(full-set(texts))/len(full))
        global_ids.update(ids)
        varying=sum(len(set(c))>1 for c in per_cluster.values() if len(c)>=2)
        multiple=sum(len(c)>=2 for c in per_cluster.values())
        report["splits"][split]={
            "questions":len(examples),"clusters":len(per_cluster),
            "clusters_with_2plus_questions":multiple,
            "clusters_with_varying_answer_count":varying,
            "fraction_contrast_clusters_with_count_change":
                varying/multiple if multiple else None,
            "candidate_K_histogram":dict(sorted(Counter(candidate_lengths).items())),
            "candidate_K_min":min(candidate_lengths),
            "candidate_K_max":max(candidate_lengths),
            "positive_count_histogram":dict(sorted(histogram.items())),
            "mean_gold_count":statistics.mean(counts),
            "std_gold_count":statistics.pstdev(counts),
            "count_min":min(counts),"count_max":max(counts),
            "empty_answer_fraction":sum(not x for x in nonzero)/len(nonzero),
            "multi_answer_fraction":sum(c>=2 for c in counts)/len(counts),
            "median_question_words":statistics.median(qlengths),
            "clusters_not_independent_samples":True,
            "duplicate_candidate_text_question_count":duplicate_candidates,
            "mean_fraction_complete_answers_outside_candidates":
                statistics.mean(out_of_candidates) if out_of_candidates else None,
            "full_answer_coverage_audited_questions":len(out_of_candidates),
            "constraint_count_histogram":
                dict(sorted(Counter(constraints_lengths).items())),
            "evidence_availability":dict(evidence_stats),
        }
        if any(k!=100 for k in candidate_lengths):
            report["issues"].append(
                f"{split}: not every question offers exactly 100 options; report this "
                "rather than pretending all have K=100")
        if duplicate_candidates:
            report["issues"].append(
                f"{split}: {duplicate_candidates} questions have duplicate candidate "
                "texts; fix identity before fitting a set classifier")
        if len(out_of_candidates)>0 and statistics.mean(out_of_candidates)>.2:
            report["issues"].append(
                f"{split}: candidate pool misses many complete-answer entities; "
                "must report candidate-recall ceiling, not gold-complete EM")
    report["decision_gate"]={
       "data_ready_for_model":not report["issues"],
       "count_gap_measured":False,
       "required_next":"Train candidate BCE and a frozen independent count head with TRAIN count-prior init on TRAIN; "
                       "compare paired DEV count NLL, joint NLL and constraint-cluster worst-case F1.",
       "must_never_use":"per-example gold count or complete answer to fill candidate lists",
       "recommended_model_input":"question and candidate entities with identical fixed-rule retrieval "
                                 "context for all arms; do not expose gold evidence at Test",
    }
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive",type=Path)
    source.add_argument("--data-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite previous data audit")
    if args.archive and not args.archive.is_file():
        p.error("official third-party data archive not found; cannot claim data verified")
    result=screen(load_zip(args.archive) if args.archive else load_dir(args.data_dir))
    result["source_kind"]="third_party_zip" if args.archive else "local_bz2_dir"
    result["source_path"]=str(args.archive or args.data_dir)
    if args.archive:
        source_digest=hashlib.sha256()
        with args.archive.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1<<20),b""):
                source_digest.update(chunk)
        result["source_sha256"]=source_digest.hexdigest()
    else:
        result["source_sha256"]=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
