#!/usr/bin/env python3
"""P0: frozen DEV prediction comparison — zero Qwen forwards and no training.

Compared on the SAME gold rows: BCE threshold (0.5 and DEV sweep), set-MAP,
structured marginal threshold (also DEV sweep), count-mode Top-k and
NON-DEPLOYABLE gold-count Top-k rank oracle. Separately score count NLL,
proper gold-set NLL, count MAE, macro/micro F1 and paired cluster bootstrap.

Every structured prediction is validated by the production scorer before
any statistics; historical incorrectly decoded checkpoints must first be
redecoded using paper_static.redecode_set_map. NO Test access.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from harness.multiselect_metrics import read_dataset, read_jsonl, prediction_records
from harness.set_distillation import log_partitions, allowed_counts
from paper_static.eurlex_metrics import score as official_score


THRESHOLDS = tuple(i / 100 for i in range(5, 96, 5))


def _sum_confusion(questions, decisions, n=100):
    tp,fp,fn=[0]*n,[0]*n,[0]*n
    exact=over=under=0
    for row, selected in zip(questions,decisions):
        gold={i for i,y in enumerate(row["target"]) if y}
        selected=set(selected)
        exact+=selected==gold
        over+=len(selected)>len(gold)
        under+=len(selected)<len(gold)
        for j in selected & gold: tp[j]+=1
        for j in selected-gold: fp[j]+=1
        for j in gold-selected: fn[j]+=1
    TP,FP,FN=map(sum,(tp,fp,fn))
    def f1(a,b,c):
        return 2*a/(2*a+b+c) if 2*a+b+c else 0.0
    return {"micro_f1":f1(TP,FP,FN),
            "macro_f1":sum(f1(tp[j],fp[j],fn[j]) for j in range(n))/n,
            "exact_set_em":exact/len(questions),
            "over_selection_rate":over/len(questions),
            "under_selection_rate":under/len(questions),
            "mean_predicted_count":sum(len(s) for s in decisions)/len(questions),
            "mean_gold_count":sum(sum(r["target"]) for r in questions)/len(questions)}


def _logsum(values):
    m=max(values)
    return m+math.log(sum(math.exp(v-m) for v in values))


def _bce_log_count(scores,support):
    center=max(scores)
    z=log_partitions([s-center for s in scores])
    logz=[z[m]+m*center for m in range(len(scores)+1)]
    ln=_logsum([logz[m] for m in support])
    return [logz[m]-ln if m in support else None for m in range(len(scores)+1)],logz


def _selected_topk(scores,k):
    return sorted(sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k])


def _bootstrap_pair(rows, primary,baseline, *, samples=200, seed=43):
    """Cluster paired question bootstrap; threshold frozen on original DEV."""
    if samples<1:
        raise ValueError("bootstrap samples must be positive")
    if not (len(rows)==len(primary)==len(baseline)):
        raise ValueError("invalid paired cohort")
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[str(row.get("group_id") or row["id"])].append(i)
    clusters=list(groups.values())
    rng=random.Random(seed)
    vals=[]
    for _ in range(samples):
        indices=[i for _ in range(len(clusters))
                 for i in clusters[rng.randrange(len(clusters))]]
        chosen_rows=[rows[i] for i in indices]
        a=_sum_confusion(chosen_rows,[primary[i] for i in indices])["micro_f1"]
        b=_sum_confusion(chosen_rows,[baseline[i] for i in indices])["micro_f1"]
        vals.append(a-b)
    vals.sort()
    return [vals[max(0,int(.025*(samples-1)))],
            vals[min(samples-1,int(.975*(samples-1)))] ]


def analyze(rows, models, *, bootstrap=200, seed=20261102):
    """models: {arm: saved predictions list}, all covering same DEV IDs."""
    if not rows or any(r.get("split")!="dev" or r.get("source")!="lexglue_eurlex"
                       or len(r["target"])!=100 for r in rows):
        raise ValueError("P0 must run on 100-label EUR-LEX DEV only")
    if not models:
        raise ValueError("no prediction arms supplied")
    arms={}
    decisions={}
    per_item={}
    for arm, saved in models.items():
        if any(not isinstance(p.get("checkpoint_id"),str) or not p.get("checkpoint_id") for p in saved):
            raise ValueError(f"{arm}: missing checkpoint identity")
        if len({p.get("checkpoint_id") for p in saved})>1:
            raise ValueError(f"{arm}: mixed checkpoint identifiers")
        records=prediction_records(rows,saved,threshold=.5,decoder="auto")
        if any(r["decoder"] not in ("set_map","threshold_multi") for r in records):
            raise ValueError(f"{arm}: unexpected selection mode")
        # This call also checks every candidate identity, full MAP and
        # numerically coherent count law under public legal support.
        measures=official_score(rows,saved,threshold=.5,expected_k=100)
        is_structured=all(r["decoder"]=="set_map" for r in records)
        if is_structured != any(r["decoder"]=="set_map" for r in records):
            raise ValueError(f"{arm}: mixed structured and BCE outputs")
        raw_scores=[r.get("candidate_logits",r.get("logits")) for r in records]
        if any(not isinstance(s,list) or len(s)!=100 for s in raw_scores):
            raise ValueError(f"{arm}: missing raw logits; proper NLL is required")
        counts=[]
        count_modes=[]
        for row,r,s in zip(rows,records,raw_scores):
            support=allowed_counts(row)
            if is_structured:
                logcount=r.get("count_log_probs")
                if logcount is None:
                    raise ValueError(f"{arm}: structured count_log_probs must be stable")
            else:
                logcount,_=_bce_log_count(s,support)
            mode=max(support,key=lambda m:(logcount[m],-m))
            count_modes.append(mode)
            counts.append({
                "gold_count":sum(row["target"]),
                "count_nll":-logcount[sum(row["target"])],
                "count_mode":mode
            })
        dec={
            "set_map_or_fixed_threshold_05":[list(r["predicted_indices"]) for r in records],
            "count_mode_topk":[_selected_topk(s,m) for s,m in zip(raw_scores,count_modes)],
            "oracle_gold_count_topk_NON_DEPLOYABLE":[
                _selected_topk(s,sum(row["target"]))
                for row,s in zip(rows,raw_scores)
            ],
        }
        candidates=[]
        for t in THRESHOLDS:
            selected=[
                [i for i,p in enumerate(r["p"]) if p>=t]
                for r in records
            ]
            result=_sum_confusion(rows,selected)
            candidates.append({**result,"threshold":t})
        best=max(candidates,key=lambda x:(x["micro_f1"],-abs(x["threshold"]-.5)))
        dec["dev_tuned_marginal_threshold"]=[
            [i for i,p in enumerate(r["p"]) if p>=best["threshold"]]
            for r in records
        ]
        decisions[arm]=dec
        main={name:_sum_confusion(rows,choices) for name,choices in dec.items()}
        main["dev_tuned_marginal_threshold"]["threshold"]=best["threshold"]
        main["dev_tuned_marginal_threshold"]["dev_optimism_warning"]=(
            "Threshold selected and reported on SAME DEV; NOT held-out Test")
        # Treat official-score count/joint metrics as distinct proper
        # probability diagnostics, not as a F1-optimizing decoder result.
        arms[arm]={
            "is_structured":is_structured,
            "checkpoint_id":next(iter({p.get("checkpoint_id") for p in saved})),
            "proper_probability":{
                "count_nll":measures["count_nll"],
                "proper_joint_set_nll":measures["proper_joint_set_nll"],
                "count_brier":measures["count_brier"],
                "count_mode_accuracy":measures["count_mode_accuracy"],
                "count_mae_under_original_map":measures["count_mae"],
            },
            "decoders":main,
            "dev_threshold_grid":candidates,
        }
        per_item[arm]=counts

    # The same saved row IDs and gold are used for every pairwise difference.
    reference="bce_continue" if "bce_continue" in arms else None
    paired={}
    if reference:
        base=decisions[reference]["dev_tuned_marginal_threshold"]
        for arm in arms:
            if arm==reference:
                continue
            cand=decisions[arm]["dev_tuned_marginal_threshold"]
            paired[arm]={
                "delta_micro_f1_vs_bce_dev_tuned":
                arms[arm]["decoders"]["dev_tuned_marginal_threshold"]["micro_f1"]-
                arms[reference]["decoders"]["dev_tuned_marginal_threshold"]["micro_f1"],
                "paired_cluster_bootstrap_95_ci":
                _bootstrap_pair(rows,cand,base,samples=bootstrap,seed=seed),
                "delta_mean_count_nll_vs_bce":
                sum(x["count_nll"]-y["count_nll"]
                    for x,y in zip(per_item[arm],per_item[reference]))/len(rows),
            }
    return {
        "status":"DEV_OFFLINE_COMPLETE_NO_GPU",
        "n_questions":len(rows),"arms":arms,
        "paired_vs_bce_continue":paired,
        "oracle_not_deployable":True,
        "warning":"DEV threshold performance optimistic; freeze threshold and evaluate new official TEST once",
        "claim":"No automated superiority claim; inspect proper Count/Set NLL and candidate-oracle gap.",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--arm",action="append",required=True,
                   help="repeat NAME=/path/to/DEV/predictions.jsonl; one checkpoint per arm")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=200)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite prior P0 report")
    rows=read_dataset(args.gold,expected_split="dev")
    models={}
    inputs={}
    for item in args.arm:
        name,sep,path=item.partition("=")
        if not sep or not name or not path or name in models:
            p.error("--arm requires unique NAME=path")
        filename=Path(path)
        models[name]=read_jsonl(filename)
        inputs[name]={"file":str(filename),
                      "sha256":hashlib.sha256(filename.read_bytes()).hexdigest()}
    result=analyze(rows,models,bootstrap=args.bootstrap)
    result["gold_sha256"]=hashlib.sha256(args.gold.read_bytes()).hexdigest()
    result["source_predictions"]=inputs
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "n":result["n_questions"],
        "microF1_by_decoder":{
            arm:{name:round(info["micro_f1"],5)
                 for name,info in block["decoders"].items()}
            for arm,block in result["arms"].items()},
        "proper_probability":{
            arm:block["proper_probability"] for arm,block in result["arms"].items()},
    },indent=2,allow_nan=False))


if __name__=="__main__":
    main()
