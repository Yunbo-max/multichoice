#!/usr/bin/env python3
"""Freeze BCE threshold using EUR-LEX VALIDATION only, without opening Test.

Reads the same 100-option predictions as eurlex_metrics and validates each
question, option order and input digest before tuning. Output is a NEW JSON
selection file containing the selected threshold and DEV metrics. This tool
does not inspect or load Test. Thresholds are selected separately for
official micro-F1, macro-F1 and Exact-Set EM, with transparent tie-breaking.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from harness.multiselect_metrics import prediction_records, read_dataset


def tune(rows, predictions, thresholds=None):
    thresholds=thresholds or [i/100 for i in range(5,96,5)]
    if len(set(thresholds)) != len(thresholds) or any(not 0<t<1 for t in thresholds):
        raise ValueError("invalid, duplicate or out-of-range thresholds")
    records=prediction_records(rows,predictions,.5,decoder="protocol")
    if any(r["source"]!="lexglue_eurlex" or len(r["target"])!=100 for r in records):
        raise ValueError("this protocol expects full 100-label EUR-LEX")
    if any(r["decoder"]=="set_map" for r in records):
        raise ValueError("BCE threshold tuning must not score an anchored set decoder")
    results=[]
    for threshold in thresholds:
        tp=[0]*100
        fp=[0]*100
        fn=[0]*100
        em=0
        for row in records:
            selected=[p>=threshold for p in row["p"]]
            gold=[bool(x) for x in row["target"]]
            em+=selected==gold
            for j,(a,b) in enumerate(zip(selected,gold)):
                if a and b: tp[j]+=1
                if a and not b: fp[j]+=1
                if not a and b: fn[j]+=1
        def f1(tp,fp,fn):
            return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
        micro=f1(sum(tp),sum(fp),sum(fn))
        macro=sum(f1(tp[j],fp[j],fn[j]) for j in range(100))/100
        results.append({"threshold":threshold,"micro_f1":micro,
                        "macro_f1":macro,"exact_set_em":em/len(records)})
    winners={}
    for key in ("micro_f1","macro_f1","exact_set_em"):
        winners[key]=max(results,key=lambda x:(x[key],-x["threshold"]))
    return {"status":"DEV_THRESHOLD_SELECTION_ONLY","n":len(records),
            "grid":results,"best":winners,
            "not_a_test_result":True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing overwrite")
    rows=read_dataset(args.gold,expected_split="dev")
    preds=[json.loads(s) for s in args.predictions.read_text(encoding="utf-8").splitlines() if s]
    result=tune(rows,preds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"n":result["n"],"best":result["best"]},indent=2))


if __name__=="__main__":
    main()
