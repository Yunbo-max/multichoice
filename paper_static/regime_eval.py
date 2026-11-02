"""Evaluate trained decision readouts against exact controlled-regime teacher law.

Inputs are strictly split-disjoint student predictions and oracle evaluation
metadata. Never use oracle scores, edges, marginals, cardinality, or valid sets
during training/inference. Results are oracle expected NLL/KL over contexts,
not a classification score cherry-picked from repeated sampled labels.

Mode:
 bce: option logits only, a=J=0
 anchor: logits plus RAW count_residual_logits (NOT count_probs)
 pair: raw logits, raw residual, predicted/parsed disjoint pairs and J weights
      (must explicitly label edge_source; "oracle" is diagnostic only).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from .block_pair_dp import exact_block_readout, exact_subset_nll


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        data=[json.loads(line) for line in f if line.strip()]
    if not data:
        raise ValueError("empty file: "+str(path))
    return data


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_predictions(student, predictions, split):
    reference={}
    for row in student:
        if row.get("split")!=split or row["id"] in reference:
            raise ValueError("non-unique ID or split mismatch")
        reference[row["id"]]=row
    model={}
    for pred in predictions:
        ident=pred.get("id")
        if ident in model or ident not in reference:
            raise ValueError("duplicate or unknown prediction id")
        row=reference[ident]
        if pred.get("option_keys") != [o["key"] for o in row["options"]]:
            raise ValueError("option identity/order mismatch")
        model[ident]=pred
    if set(reference)!=set(model):
        raise ValueError("missing predictions")
    return reference,model


def _safe_array(array,k,label):
    if not isinstance(array,list) or len(array)!=k or any(type(v) not in (int,float)
       or not math.isfinite(v) for v in array):
        raise ValueError(f"{label} must be a finite {k}-element list")
    return array


def _model_parameters(row,pred,mode):
    k=len(row["options"])
    scores=_safe_array(pred.get("candidate_logits",pred.get("logits")),k,"candidate_logits")
    residual=([0.0]*(k+1) if mode=="bce" else
              _safe_array(pred.get("count_residual_logits"),k+1,"count_residual_logits"))
    pairs=[]
    weights=[]
    edge_source="none"
    if mode=="pair":
        pairs=pred.get("predicted_pairs")
        weights=pred.get("predicted_pair_weights")
        edge_source=pred.get("edge_source")
        if edge_source not in ("learned","public_rule_parser","oracle_diagnostic"):
            raise ValueError("pair predictions must declare learned/public/oracle edge source")
        if not isinstance(pairs,list):
            raise ValueError("missing predicted pairs")
        pairs=[tuple(pair) for pair in pairs]
        _safe_array(weights,len(pairs),"predicted pair weights")
    elif "predicted_pairs" in pred:
        raise ValueError("BCE/Anchor cannot silently use pair edges")
    return scores,residual,pairs,weights,edge_source


def evaluate(students,oracles,predictions,mode,split):
    if mode not in ("bce","anchor","pair"):
        raise ValueError("unknown mode")
    if split not in ("dev","test"):
        raise ValueError("audit only DEV or sealed synthetic TEST")
    by_row,by_pred=validate_predictions(students,predictions,split)
    contexts={}
    for row in students:
        contexts.setdefault(row["group_id"],[]).append(row)
    teacher={}
    for row in oracles:
        cid=row.get("context_id")
        if cid in teacher or not isinstance(cid,str):
            raise ValueError("oracle context missing or duplicated")
        teacher[cid]=row
    if set(contexts)!=set(teacher):
        raise ValueError("oracle and prediction contexts mismatch")
    records=[]
    for cid,rows in contexts.items():
        ref=rows[0]
        k=len(ref["options"])
        if any(r["state"]!=ref["state"] or r["question"]!=ref["question"] or
               r["options"]!=ref["options"] or
               r["selection_mode"]!=ref["selection_mode"] or
               r["allow_empty"]!=ref["allow_empty"] for r in rows):
            raise ValueError("repeated context must have identical public inputs")
        scores,residual,pairs,weights,edge_source=_model_parameters(ref,by_pred[ref["id"]],mode)
        for r in rows[1:]:
            other=_model_parameters(r,by_pred[r["id"]],mode)
            if any(abs(x-y)>2e-3 for x,y in zip(scores,other[0])):
                raise ValueError("batch/duplicate predictions disagree on same prompt")
            if any(abs(x-y)>2e-3 for x,y in zip(residual,other[1])):
                raise ValueError("count residual inconsistent across identical prompts")
            if other[2:]!=(pairs,weights,edge_source):
                raise ValueError("pair routing inconsistent across identical prompts")
        truth=teacher[cid]
        q=_safe_array(truth["true_marginals"],k,"oracle q")
        rho=_safe_array(truth["true_count_probs"],k+1,"oracle rho")
        mat=truth["true_pairwise_marginals"]
        p_truth=truth["true_set_probs"]
        if len(mat)!=k or any(len(a)!=k for a in mat):
            raise ValueError("invalid oracle pair moments")
        entropy=-sum(p*math.log(p) for p in p_truth.values() if p>0)
        out=exact_block_readout(scores,pairs,weights,residual)
        joint_ce=(out["log_partition"]-sum(a*b for a,b in zip(q,scores))-
                  sum(a*b for a,b in zip(rho,residual))-
                  sum(weight*mat[i][j] for (i,j),weight in zip(pairs,weights)))
        count_ce=-sum(p*lp for p,lp in zip(rho,out["count_log_probs"]) if p>0)
        count_entropy=-sum(p*math.log(p) for p in rho if p>0)
        joint_gap=joint_ce-entropy
        count_gap=count_ce-count_entropy
        if joint_gap < -1e-6 or count_gap < -1e-6:
            raise ValueError("negative KL beyond rounding tolerance: oracle/decoder mismatch")
        sampled_nll=sum(exact_subset_nll(scores,r["target"],pairs,weights,residual)
                        for r in rows)/len(rows)
        pair_metrics={}
        if mode=="pair" and truth["regime"]=="R":
            true_edges={tuple(sorted(e)) for e in truth["teacher_pairs"]}
            predicted_edges={tuple(sorted(e)) for e in pairs}
            tp=len(true_edges & predicted_edges)
            prec=tp/len(predicted_edges) if predicted_edges else 0.0
            rec=tp/len(true_edges) if true_edges else 0.0
            pair_metrics={"edge_precision":prec,"edge_recall":rec,
                          "edge_f1":(2*prec*rec/(prec+rec)) if prec+rec else 0.0}
        records.append({"context":cid,"regime":truth["regime"],
                        "k":k,"strength":truth["strength"],"draws":len(rows),
                        "edge_source":edge_source,"joint_ce":joint_ce,
                        "joint_kl":max(0.0,joint_gap),
                        "count_ce":count_ce,"count_kl":max(0.0,count_gap),
                        "sampled_set_nll":sampled_nll,**pair_metrics})
    groups={}
    for regime in ("I","C","R"):
        items=[r for r in records if r["regime"]==regime]
        if items:
            groups[regime]={"contexts":len(items),
                "mean_joint_ce":sum(r["joint_ce"] for r in items)/len(items),
                "mean_joint_kl":sum(r["joint_kl"] for r in items)/len(items),
                "mean_count_ce":sum(r["count_ce"] for r in items)/len(items),
                "mean_count_kl":sum(r["count_kl"] for r in items)/len(items),
                "mean_sampled_set_nll":sum(r["sampled_set_nll"] for r in items)/len(items)}
            if mode=="pair" and regime=="R":
                groups[regime]["mean_edge_f1"]=sum(r["edge_f1"] for r in items)/len(items)
                groups[regime]["mean_edge_precision"]=sum(r["edge_precision"] for r in items)/len(items)
                groups[regime]["mean_edge_recall"]=sum(r["edge_recall"] for r in items)/len(items)
    return {"status":"EVALUATED_NO_TRAINING_BY_THIS_SCRIPT",
            "split":split,"model_mode":mode,"n_contexts":len(records),
            "edge_source_counts":{v:sum(r["edge_source"]==v for r in records)
                                  for v in {r["edge_source"] for r in records}},
            "metrics_by_regime":groups,
            "not_a_learned_pair_result":mode=="pair" and
               any(r["edge_source"]=="oracle_diagnostic" for r in records)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--mode",required=True,choices=["bce","anchor","pair"])
    p.add_argument("--split",choices=["dev","test"],default="dev")
    p.add_argument("--confirm-test-once",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if args.split=="test" and not args.confirm_test_once:
        p.error("TEST is held-out: freeze method first and pass --confirm-test-once")
    if args.output.exists():
        p.error("refusing to overwrite evaluated results")
    st=args.data_root/"student"/(args.split+".jsonl")
    oracle=args.data_root/"oracle_DO_NOT_USE_AS_MODEL_INPUT"/(args.split+".jsonl")
    report=evaluate(read_jsonl(st),read_jsonl(oracle),
                    read_jsonl(args.predictions),args.mode,args.split)
    report["student_sha256"]=hash_file(st)
    report["oracle_sha256"]=hash_file(oracle)
    report["predictions_sha256"]=hash_file(args.predictions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
