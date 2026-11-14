#!/usr/bin/env python3
"""Matched RoMQA Closed Set/Count evaluation, entirely offline on DEV.

This validates the EXISTING Qwen eval_multiselect prediction JSONL format.
No TEST labels or submissions. No count oracle, no gold-injected candidates.
Official RoMQA overall F1 ignores empty candidate-gold examples; cluster_min
retains them, and official evaluation excludes rows with empty complete_answer.
We ALSO publish all-row metrics and proper NLL without selective exclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from harness.multiselect_metrics import prediction_records, read_dataset
from harness.set_distillation import log_partitions


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()


def _read(path):
    out=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _ls(v):
    hi=max(v)
    return hi+math.log(math.fsum(math.exp(x-hi) for x in v))


def _stable_logz(scores):
    center=max(scores)
    return [v+m*center for m,v in enumerate(log_partitions([s-center for s in scores]))]


def _set_f1(pred,gold):
    if not pred and not gold:
        # Official RoMQA SetF1 returns 0 for two empty sets.
        return 0.0
    a=len(pred&gold)
    return 2*a/(len(pred)+len(gold)) if a else 0.0


def _bootstrap(data,*,n=1000,seed=43):
    """Resample RELATED QUESTION CLUSTERS, never individual candidates."""
    if not data:
        return None
    groups=defaultdict(list)
    for r in data:
        groups[r["cluster_id"]].append(r)
    if len(groups)<2 or n<1:
        return None
    names=sorted(groups)
    rng=random.Random(seed)
    samples=defaultdict(list)
    for _ in range(n):
        chosen=[groups[names[rng.randrange(len(names))]] for _ in names]
        cases=[item for group in chosen for item in group]
        samples["set_f1"].append(statistics.mean(c["set_f1"] for c in cases))
        samples["count_nll"].append(statistics.mean(c["count_nll"] for c in cases))
        samples["exact_match"].append(statistics.mean(c["exact_match"] for c in cases))
    def quantile(items,p):
        values=sorted(items)
        x=p*(len(values)-1)
        i=math.floor(x)
        j=math.ceil(x)
        return values[i]*(j-x)+values[j]*(x-i) if i!=j else values[i]
    return {key:[quantile(values,.025),quantile(values,.975)]
            for key,values in samples.items()}


def verify_source(rows,sidecar):
    if any(r.get("source")!="romqa_closed" or r.get("split")!="dev"
           or r.get("selection_mode")!="multi" or r.get("allow_empty") is not True
           for r in rows):
        raise ValueError("RoMQA Closed DEV only; no TEST, mixed sources, or known Gold count")
    if len(sidecar)!=len(rows):
        raise ValueError("missing complete-answer sidecar entries")
    evaluation={}
    for data in sidecar:
        name=data.get("id")
        if not isinstance(name,str) or name in evaluation:
            raise ValueError("bad official sidecar ID")
        evaluation[name]=data
    if set(evaluation)!={r["id"] for r in rows}:
        raise ValueError("official sidecar IDs differ from rows")
    for r in rows:
        a=evaluation[r["id"]]
        if a.get("cluster_id")!=r["group_id"]:
            raise ValueError("sidecar cluster mismatch")
        if a.get("candidate_keys")!=[c["key"] for c in r["options"]]:
            raise ValueError("sidecar candidate order mismatch")
        actual=[o["key"] for o,y in zip(r["options"],r["target"]) if y]
        if a.get("candidate_gold")!=actual:
            raise ValueError("sidecar candidate gold mismatch")
        complete=a.get("complete_answer")
        if complete is not None and (not isinstance(complete,list)
            or any(not isinstance(s,str) for s in complete)):
            raise ValueError("invalid official full-answer fields")
    return evaluation


def score(rows, predictions, *, full_answers, threshold=.5,
          decoder="auto",bootstrap=1000):
    if not rows:
        raise ValueError("empty RoMQA DEV examples")
    sidecar=verify_source(rows, full_answers)
    records=prediction_records(rows,predictions,threshold=threshold,decoder=decoder)
    if len(records)!=len(rows):
        raise AssertionError("unpaired predictions")
    per=[]
    for row,record in zip(rows,records):
        k=len(row["options"])
        gold={i for i,y in enumerate(row["target"]) if y}
        selected=set(record["predicted_indices"])
        scores=record.get("candidate_logits",record.get("logits"))
        if (not isinstance(scores,list) or len(scores)!=k or
            any(type(x) not in (float,int) or not math.isfinite(x) for x in scores)):
            raise ValueError(f"{row['id']}: finite RAW candidate logits required for proper NLL")
        z=_stable_logz(scores)
        count=record.get("count_log_probs")
        if count is not None:
            if (not isinstance(count,list) or len(count)!=k+1
                 or any(type(v) not in (float,int) or not math.isfinite(v) for v in count)):
                raise ValueError("missing/invalid structured full-precision count law")
            if abs(_ls(count))>1e-5:
                raise ValueError("structured count law not normalized")
        else:
            if record.get("decoder")=="set_map":
                raise ValueError("full-precision structured count law required, no clipped p")
            norm=_ls(z)
            count=[v-norm for v in z]
        m=len(gold)
        joint=-(count[m]+math.fsum(scores[i] for i in gold)-z[m])
        if joint < -1e-5 or not math.isfinite(joint):
            raise ValueError(f"{row['id']}: invalid proper joint set likelihood")
        full=sidecar[row["id"]]["complete_answer"]
        candidate=set(sidecar[row["id"]]["candidate_keys"])
        fullset=set(full) if full is not None else None
        true_count=m
        prediction_names={row["options"][i]["key"] for i in selected}
        one={
            "id":row["id"],"cluster_id":row["group_id"],
            "gold_count":m,"predicted_count":len(selected),
            "count_abs_error":abs(len(selected)-m),
            "exact_match":float(selected==gold),
            "set_f1":_set_f1(selected,gold),
            "count_nll":-count[m],
            "joint_set_nll":max(0.0,joint),
            "over_select":int(len(selected)>m),
            "under_select":int(len(selected)<m),
            "nonempty_candidate_gold":bool(gold),
            "complete_answer_available":fullset is not None,
            "complete_answer_nonempty":bool(fullset) if fullset is not None else False,
            "candidate_complete":(fullset <= candidate) if fullset is not None else None,
            "complete_exact_match":float(prediction_names==fullset) if fullset is not None else None,
            "complete_set_f1":_set_f1(prediction_names,fullset) if fullset is not None else None,
        }
        per.append(one)
    groups=defaultdict(list)
    for q in per:groups[q["cluster_id"]].append(q)
    def avg(name,cases):
        return statistics.mean(x[name] for x in cases) if cases else None
    def cluster_min(name, samples):
        units=defaultdict(list)
        for item in samples:units[item["cluster_id"]].append(item[name])
        return statistics.mean(min(group) for group in units.values()) if units else None
    nonempty=[p for p in per if p["nonempty_candidate_gold"]]
    official=[p for p in per if p["complete_answer_nonempty"]]
    # Unlike a custom all-case macro F1, the published official closed scorer
    # excludes rows without complete_answer from its input, and ignores empty
    # candidate gold rows for its average SetF1. It retains empty-gold rows
    # for per-cluster minima. Keep the separate selection denominators visible.
    official_nonempty=[p for p in official if p["nonempty_candidate_gold"]]
    checkpoints={p.get("checkpoint_id") for p in predictions if p.get("checkpoint_id")}
    if len(checkpoints)>1:
        raise ValueError("mixed checkpoint identities in one RoMQA prediction report")
    frozen_models={p.get("provenance",{}).get("model") for p in predictions
                   if isinstance(p.get("provenance"),dict)
                   and p["provenance"].get("frozen_candidate_scores")}
    if len(frozen_models)>1:
        raise ValueError("frozen feature base model changed within one report")
    report={
        "status":"ROMQA_DEV_OFFLINE_SCORED",
        "n":len(per),"n_clusters":len(groups),
        "n_nonempty_candidate_gold":len(nonempty),
        "n_official_complete_answer_nonempty":len(official),
        "official_denominator_difference_reported":True,
        "metrics_all_questions":{
            "set_f1":avg("set_f1",per),
            "exact_match":avg("exact_match",per),
            "count_nll":avg("count_nll",per),
            "joint_set_nll":avg("joint_set_nll",per),
            "count_mae":avg("count_abs_error",per),
            "over_selection":avg("over_select",per),
            "under_selection":avg("under_select",per),
        },
        "metrics_official_closed_candidate":{
            "set_f1_ignore_empty":avg("set_f1",official_nonempty),
            "exact_match":avg("exact_match",official),
            "cluster_min_f1":cluster_min("set_f1",official),
            "cluster_min_exact_match":cluster_min("exact_match",official),
            "complete_set_f1":avg("complete_set_f1",official),
            "complete_exact_match":avg("complete_exact_match",official),
            "candidate_complete_fraction":avg("candidate_complete",official),
        },
        "bootstrap_cluster_95ci_all_questions":_bootstrap(per,n=bootstrap),
        "threshold":threshold,"decode_mode":decoder,
        "proper_log_likelihood":"full candidate-set distribution; no Gold count supplied",
        "test_read":False,
        "checkpoint_id":next(iter(checkpoints)) if checkpoints else None,
        "frozen_bce_checkpoint":next(iter(frozen_models)) if frozen_models else None,
    }
    return report,per


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--sidecar",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threshold",type=float,default=.5)
    p.add_argument("--decoder",choices=["auto","protocol"],default="auto")
    p.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refuse to overwrite report directory")
    if args.bootstrap<0:p.error("bootstrap must be >=0")
    data=read_dataset(args.data,expected_split="dev")
    report,per=score(data,_read(args.predictions),full_answers=_read(args.sidecar),
                     threshold=args.threshold,decoder=args.decoder,bootstrap=args.bootstrap)
    args.output.mkdir(parents=True)
    report.update({
        "source_data_sha256":digest(args.data),
        "sidecar_sha256":digest(args.sidecar),
        "predictions_sha256":digest(args.predictions),
    })
    (args.output/"report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    with (args.output/"per_question.jsonl").open("x",encoding="utf-8") as f:
        for item in per:
            f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],
                      "all":report["metrics_all_questions"],
                      "official":report["metrics_official_closed_candidate"],
                      "source":str(args.data)},indent=2))


if __name__=="__main__":
    main()
