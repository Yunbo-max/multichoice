#!/usr/bin/env python3
"""Reference exact small-K relational set model, not a 255-option GPU model.

Energy: sum_i s_i*y_i + a_|Y| + sum_{i<j} J_ij*y_i*y_j.
With J=0, this is Count Anchor. With J=a=0 and all counts permitted,
this is exactly independent Bernoulli BCE. Prohibit large K: enumerating
2**K is intentionally NOT represented as efficient for K=255.
"""
from __future__ import annotations
import argparse
import json
import math

MAX_EXACT_K = 12


def _validate(scores, residual, interactions, counts):
    k = len(scores)
    if not 1 <= k <= MAX_EXACT_K or any(not math.isfinite(s) for s in scores):
        raise ValueError(f"exact pairwise enumeration requires 1 <= K <= {MAX_EXACT_K} and finite logits")
    if residual is None:
        residual = [0.0] * (k+1)
    if len(residual) != k+1 or any(not math.isfinite(v) for v in residual):
        raise ValueError("a_m must provide K+1 finite residuals")
    if interactions is None:
        interactions = [[0.0]*k for _ in range(k)]
    if len(interactions)!=k or any(len(row)!=k for row in interactions):
        raise ValueError("J must be K x K")
    for i in range(k):
        if not math.isfinite(interactions[i][i]) or abs(interactions[i][i])>1e-12:
            raise ValueError("J diagonal must equal zero")
        for j in range(i+1,k):
            if not (math.isfinite(interactions[i][j]) and math.isfinite(interactions[j][i])
                    and abs(interactions[i][j]-interactions[j][i])<1e-9):
                raise ValueError("J must be finite and symmetric")
    if counts is None:
        counts = list(range(k+1))
    counts = sorted(set(counts))
    if not counts or any(type(m) is not int or not 0<=m<=k for m in counts):
        raise ValueError("empty or invalid public cardinality support")
    return residual, interactions, counts


def exact_distribution(scores, residual=None, interactions=None, counts=None):
    """Return a normalized dict {subset_bitmask: probability}, K<=12."""
    residual, interactions, allowed = _validate(scores, residual, interactions, counts)
    k=len(scores)
    energies={}
    for mask in range(1 << k):
        selected=[i for i in range(k) if mask >> i & 1]
        if len(selected) not in allowed:
            continue
        energies[mask]=(sum(scores[i] for i in selected)+residual[len(selected)]
                        +sum(interactions[i][j] for idx,i in enumerate(selected)
                             for j in selected[idx+1:]))
    maximum=max(energies.values())
    denom=sum(math.exp(x-maximum) for x in energies.values())
    return {mask:math.exp(energy-maximum)/denom for mask,energy in energies.items()}


def moments(probs,k):
    marg=[0.0]*k
    pair=[[0.0]*k for _ in range(k)]
    rho=[0.0]*(k+1)
    for mask,prob in probs.items():
        selected=[i for i in range(k) if mask>>i&1]
        rho[len(selected)]+=prob
        for i in selected:
            marg[i]+=prob
        for idx,i in enumerate(selected):
            for j in selected[idx+1:]:
                pair[i][j]+=prob
                pair[j][i]+=prob
    return {"marginals":marg,"count_probs":rho,"pairwise_marginals":pair}


def dataset_nll(probs, target):
    """Cross-entropy of an observed joint set distribution under model."""
    if abs(sum(target.values())-1)>1e-8 or any(v<0 for v in target.values()):
        raise ValueError("target must be a normalized distribution")
    if any(mask not in probs and mass>0 for mask,mass in target.items()):
        raise ValueError("gold set outside support")
    return -sum(mass*math.log(probs[mask]) for mask,mass in target.items() if mass>0)


def fit_pairs_fixed_scores(scores, target, counts, *, steps=160, lr=0.8):
    """Reference MLE over J only. No claim of GPU speed, generalization or optimality."""
    k=len(scores)
    if steps<1 or lr<=0:
        raise ValueError("invalid optimization schedule")
    pairs=[(i,j) for i in range(k) for j in range(i+1,k)]
    J=[[0.0]*k for _ in range(k)]
    gold_pair={ij:sum(w for mask,w in target.items() if mask>>ij[0]&1 and mask>>ij[1]&1)
               for ij in pairs}
    trajectory=[]
    for step in range(steps+1):
        probs=exact_distribution(scores,interactions=J,counts=counts)
        trajectory.append(dataset_nll(probs,target))
        if step==steps:
            break
        est=moments(probs,k)["pairwise_marginals"]
        for i,j in pairs:
            grad=est[i][j]-gold_pair[(i,j)]
            J[i][j]-=lr*grad
            J[j][i]=J[i][j]
    return J, trajectory


def demo():
    """Synthetic AB/CD multimodality: identical marginals and count, different pair law."""
    k=4
    gold={0b0011:0.5, 0b1100:0.5}  # {A,B} or {C,D}; other 2-sets invalid.
    scores=[0.0]*k
    bce=exact_distribution(scores)
    bce_protocol_matched=exact_distribution(scores,counts=[2])
    anchor=exact_distribution(scores,counts=[2])
    fitted,history=fit_pairs_fixed_scores(scores,gold,[2])
    pair=exact_distribution(scores,interactions=fitted,counts=[2])
    return {"status":"SYNTHETIC_ONLY_NOT_GPU_RESULT", "gold_sets":["AB","CD"],
            "k":k,"nll":{"bce":dataset_nll(bce,gold),
                       "bce_protocol_matched":dataset_nll(bce_protocol_matched,gold),
                       "anchor_fixed_count":dataset_nll(anchor,gold),
                       "pairwise_fit":dataset_nll(pair,gold)},
            "probability_on_invalid_2_sets":{
                "anchor_fixed_count":sum(anchor.get(m,0) for m in (0b0101,0b1001,0b0110,0b1010)),
                "pairwise_fit":sum(pair.get(m,0) for m in (0b0101,0b1001,0b0110,0b1010))},
            "pairwise_training_decreased_nll":history[-1]<history[0],
            "limits":"Exact joint computation only K<=12; multi-modal gold sets require joint labels."}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",help="optional new JSON output file")
    args=p.parse_args(argv)
    report=demo()
    result=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    if args.output:
        from pathlib import Path
        output=Path(args.output)
        if output.exists():
            p.error("refusing to overwrite")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(result,encoding="utf-8")
    print(result,end="")


if __name__=="__main__":
    main()
