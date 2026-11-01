#!/usr/bin/env python3
"""Score EUR-LEX 100-class Decision Models on matched, explicit gold/option IDs.

Reads harness.eval_multiselect.py's saved predictions.jsonl; no GPU.
Counts the official micro/macro F1, exact-set accuracy, count NLL/Brier,
proper joint-set NLL for BCE and structured readouts on the SAME public
legal-count support (here 0..K). Refuses partial/mismatched prediction IDs.
DEV-only by default. TEST requires --unlock-test and a frozen-selection JSON.
No ground-truth count is ever used as a decoding choice.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from harness.multiselect_metrics import prediction_records, read_dataset
from harness.set_distillation import log_partitions


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1 << 20),b""):
            h.update(part)
    return h.hexdigest()


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def _count_nll_rho_from_scores(scores, allowed):
    """Compute independent-BCE count law with stable full-shift log partitions."""
    if not scores or any(not math.isfinite(x) for x in scores):
        raise ValueError("finite raw candidate scores are mandatory")
    top = max(scores)
    logz_center = log_partitions([x - top for x in scores])
    logz = [logz_center[m] + m * top for m in range(len(scores)+1)]
    norm = _logsumexp([logz[m] for m in allowed])
    logs = [logz[m]-norm if m in allowed else None for m in range(len(scores)+1)]
    return logs, logz, norm


def _finite_probs(values, count, desc):
    if not isinstance(values,list) or len(values)!=count or any(
        type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1
        for p in values
    ):
        raise ValueError(desc + " must contain finite probability values")
    return values


def score(rows,predictions,threshold=0.5,expected_k=100):
    if not 0<threshold<1:
        raise ValueError("threshold must lie strictly in (0,1)")
    if not rows or any(len(row["options"])!=expected_k for row in rows):
        raise ValueError("all EUR-LEX questions must contain complete fixed 100 labels")
    catalog=[x["key"] for x in rows[0]["options"]]
    if any([x["key"] for x in r["options"]]!=catalog for r in rows):
        raise ValueError("option ordering changed across rows")
    if any(r["source"]!="lexglue_eurlex" for r in rows):
        raise ValueError("expected EUR-LEX source; do not pool unrelated benchmarks")
    # Existing validator enforces prediction ID coverage, input digest, option
    # order, public set support, and structured induced marginals/MAP consistency.
    records=prediction_records(rows,predictions,threshold,decoder="auto")
    by_id={r["id"]:r for r in rows}
    n=len(records)
    k=expected_k
    tp=[0]*k
    fp=[0]*k
    fn=[0]*k
    exact=0
    sum_doc_f1=0.0
    count_nll=0.0
    count_brier=0.0
    set_nll=0.0
    count_mae=0.0
    over=under=0
    count_modes=0
    n_structured=0
    for item in records:
        row=by_id[item["id"]]
        gold=row["target"]
        k=len(gold)
        true={i for i,y in enumerate(gold) if y}
        chosen=set(item["predicted_indices"])
        for j in range(k):
            pred=j in chosen
            actual=bool(gold[j])
            tp[j]+=int(pred and actual)
            fp[j]+=int(pred and not actual)
            fn[j]+=int(not pred and actual)
        a=len(chosen & true)
        sum_doc_f1+=(2*a/(len(chosen)+len(true))) if chosen or true else 1.0
        exact+=int(chosen==true)
        m=len(true)
        count_mae+=abs(len(chosen)-m)
        over+=len(chosen)>m
        under+=len(chosen)<m
        scores=item.get("candidate_logits", item.get("logits"))
        if not isinstance(scores,list) or len(scores)!=k:
            raise ValueError(f"{item['id']}: raw logits required for proper set NLL")
        support = list(range(k+1)) if row["allow_empty"] else list(range(1,k+1))
        if item.get("decoder")=="set_map":
            if "count_log_probs" not in item:
                raise ValueError(f"{item['id']}: stable count_log_probs required")
            log_count=item["count_log_probs"]
            if len(log_count)!=k+1:
                raise ValueError("invalid structured log-count dimensions")
            _, logz, _ = _count_nll_rho_from_scores(scores,support)
            ln=log_count[m]
            if type(ln) not in (float,int) or not math.isfinite(ln):
                raise ValueError(f"{item['id']}: nonfinite gold count probability")
            joint = -ln - sum(scores[i] for i in true) + logz[m]
            n_structured+=1
        else:
            log_count, _, norm = _count_nll_rho_from_scores(scores,support)
            ln=log_count[m]
            joint=norm-sum(scores[i] for i in true)
        if joint < -1e-5:
            raise ValueError(f"{item['id']}: negative normalized joint NLL")
        set_nll+=max(0.0,joint)
        count_nll+=-ln
        rho=[math.exp(v) if v is not None else 0.0 for v in log_count]
        count_modes+=int(max(range(k+1),key=lambda t:rho[t])==m)
        count_brier+=sum((p-int(j==m))**2 for j,p in enumerate(rho))
    TP=sum(tp)
    FP=sum(fp)
    FN=sum(fn)
    def f1(a,b,c):
        return (2*a/(2*a+b+c)) if (2*a+b+c) else 0.0
    return {
        "n_documents":n, "n_options":expected_k,
        "catalogue":catalog, "threshold":threshold,
        "structured_readout_count":n_structured,
        "official_micro_f1":f1(TP,FP,FN),
        "official_macro_f1":sum(f1(tp[j],fp[j],fn[j]) for j in range(expected_k))/expected_k,
        "exact_set_em":exact/n,
        "mean_per_document_f1":sum_doc_f1/n,
        "count_nll":count_nll/n,
        "count_brier":count_brier/n,
        "proper_joint_set_nll":set_nll/n,
        "count_mae":count_mae/n,
        "count_mode_accuracy":count_modes/n,
        "over_selection_rate":over/n,
        "under_selection_rate":under/n,
        "n_predicted_positive":TP+FP,
        "n_gold_positive":TP+FN,
        "protocol":"legal sets = all 2^K (allow_empty per row); BCE conditioned to same support",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["dev","test"],default="dev")
    p.add_argument("--unlock-test",action="store_true")
    p.add_argument("--frozen-selection",type=Path,
                   help="JSON written and hashed before first official TEST inference")
    p.add_argument("--threshold",type=float,default=0.5)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite evaluation results")
    if args.split=="test":
        if not args.unlock_test or not args.frozen_selection:
            p.error("TEST requires both --unlock-test and --frozen-selection")
        freeze=json.loads(args.frozen_selection.read_text())
        if not freeze.get("locked") or freeze.get("dataset")!="coastalcph/lex_glue":
            p.error("invalid frozen TEST selection provenance")
        if freeze.get("gold_sha256")!=file_sha256(args.gold):
            p.error("test gold hash does not match locked protocol")
        if freeze.get("checkpoint_id") is None:
            p.error("must freeze checkpoint identity before TEST")
    else:
        if args.unlock_test or args.frozen_selection:
            p.error("TEST selection flags are not permitted in DEV mode")
    rows=read_dataset(args.gold, expected_split=args.split)
    with args.predictions.open(encoding="utf-8") as f:
        predictions=[json.loads(s) for s in f if s.strip()]
    if args.split=="test":
        if any(x.get("checkpoint_id")!=freeze["checkpoint_id"] for x in predictions):
            p.error("test checkpoint_id differs from frozen selection")
        if args.threshold != freeze.get("threshold"):
            p.error("test threshold differs from frozen selection")
    result=score(rows,predictions,threshold=args.threshold)
    result.update({"status":"EVALUATED", "split":args.split,
                   "gold_sha256":file_sha256(args.gold),
                   "predictions_sha256":file_sha256(args.predictions),
                   "frozen_selection_sha256":file_sha256(args.frozen_selection)
                   if args.frozen_selection else None})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,
                                     allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","split","n_documents",
        "official_micro_f1","official_macro_f1","exact_set_em","count_nll",
        "proper_joint_set_nll")},indent=2))


if __name__=="__main__":
    main()
