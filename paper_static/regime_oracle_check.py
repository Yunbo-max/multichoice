"""Exact theoretical ceilings for three controlled regimes, not trained-model results.

Evaluate expected joint cross entropy from TRUE oracle moments without
repeatedly enumerating every gold set for each model. Gold oracle moments
must NEVER be features of a student model. Oracle edges here are a
capacity upper bound, not evidence a learned relation head can recover them.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from .pairwise_exact import exact_distribution, moments
from .block_pair_dp import exact_block_readout


def expected_ce(truth,scores,pairs=(),weights=(),residual=None):
    k=len(scores)
    if residual is None:
        residual=[0.]*(k+1)
    z=exact_block_readout(scores,pairs,weights,residual)
    result=z["log_partition"]
    result-=sum(a*b for a,b in zip(truth["true_marginals"],scores))
    result-=sum(a*b for a,b in zip(truth["true_count_probs"],residual))
    result-=sum(w*truth["true_pairwise_marginals"][i][j]
                for (i,j),w in zip(pairs,weights))
    return result


def stats(distribution,k):
    m=moments(distribution,k)
    m["entropy"]=-sum(p*math.log(p) for p in distribution.values() if p)
    m["true_set_probs"]={str(mask):p for mask,p in distribution.items()}
    return {"true_marginals":m["marginals"],
            "true_count_probs":m["count_probs"],
            "true_pairwise_marginals":m["pairwise_marginals"],
            "entropy":m["entropy"]}


def logit(v):
    if not 0<v<1:
        raise ValueError("oracle Bernoulli marginal must lie strictly inside (0,1)")
    return math.log(v)-math.log1p(-v)


def example_regimes():
    s=[.8,-.8,1.6,-.8,0.,.8]
    k=len(s)
    zeros=[0.]*(k+1)
    qi=exact_distribution(s)
    qci=stats(qi,k)
    si=[logit(q) for q in qci["true_marginals"]]
    base_ce=expected_ce(qci,si)
    a=[-(m-2.5)**2 for m in range(k+1)]
    qc=exact_distribution(s,a)
    qcc=stats(qc,k)
    sc=[logit(q) for q in qcc["true_marginals"]]
    count_bce=expected_ce(qcc,sc)
    count_oracle=expected_ce(qcc,s,residual=a)
    if abs(base_ce-qci["entropy"])>1e-10 or abs(count_oracle-qcc["entropy"])>1e-10:
        raise RuntimeError("nested oracle identification failed")
    k=8
    ps=[0.]*k
    pa=[-.9*(m-2.)**2 for m in range(k+1)]
    pairs=[(0,1),(2,3),(4,5),(6,7)]
    weight=[3.]*4
    matrix=[[0.]*k for _ in range(k)]
    for (i,j),v in zip(pairs,weight):
        matrix[i][j]=matrix[j][i]=v
    qr=exact_distribution(ps,pa,matrix)
    qrr=stats(qr,k)
    sr=[logit(q) for q in qrr["true_marginals"]]
    pair_bce=expected_ce(qrr,sr)
    # For this completely exchangeable paired law, its maximum-entropy count-only
    # projection is UNIFORM within each cardinality, so this is an optimal Anchor
    # joint law (without access to the specific matching).
    a_fit=[math.log(rho)-math.log(math.comb(k,m))
           for m,rho in enumerate(qrr["true_count_probs"])]
    pair_anchor=expected_ce(qrr,[0.]*k,residual=a_fit)
    pair_oracle=expected_ce(qrr,ps,pairs,weight,pa)
    if abs(pair_oracle-qrr["entropy"])>1e-10:
        raise RuntimeError("pairwise oracle reconstruction failed")
    # Rewire matching while maintaining q/rho exactly.
    rewired=[(0,2),(1,3),(4,6),(5,7)]
    j2=[[0.]*k for _ in range(k)]
    for i,j in rewired:
        j2[i][j]=j2[j][i]=3.
    q2=stats(exact_distribution(ps,pa,j2),k)
    if max(abs(x-y) for x,y in zip(q2["true_marginals"],qrr["true_marginals"]))>1e-10:
        raise RuntimeError("rewiring should preserve option marginals")
    if max(abs(x-y) for x,y in zip(q2["true_count_probs"],qrr["true_count_probs"]))>1e-10:
        raise RuntimeError("rewiring should preserve count law")
    return {
      "status":"SYNTHETIC_ORACLE_CAPACITY_ONLY_NOT_LLM_TRAINING",
      "scenarios":{
        "I_independent":{"k":6,"true_entropy":qci["entropy"],
                         "best_independent_bce_nll":base_ce,
                         "best_bce_excess_joint_kl":base_ce-qci["entropy"]},
        "C_cardinality":{"k":6,"true_entropy":qcc["entropy"],
                         "best_independent_bce_nll":count_bce,
                         "bce_excess_joint_kl":count_bce-qcc["entropy"],
                         "oracle_count_anchor_nll":count_oracle,
                         "oracle_count_anchor_excess_joint_kl":count_oracle-qcc["entropy"]},
        "R_pairwise":{"k":8,"true_entropy":qrr["entropy"],
                      "best_independent_bce_nll":pair_bce,
                      "bce_excess_joint_kl":pair_bce-qrr["entropy"],
                      "best_count_anchor_nll":pair_anchor,
                      "anchor_excess_joint_kl":pair_anchor-qrr["entropy"],
                      "oracle_pairwise_nll":pair_oracle,
                      "oracle_pairwise_excess_joint_kl":pair_oracle-qrr["entropy"],
                      "marginals_and_count_invariant_under_rewiring":True,
                      "cooccurrence_p_AB_original":qrr["true_pairwise_marginals"][0][1],
                      "cooccurrence_p_AB_rewired":q2["true_pairwise_marginals"][0][1]}}},
      "warning":"Oracle variables generate/evaluate the synthetic law; no learned Qwen model, real-world win, or 255-option training."}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args(argv)
    report=example_regimes()
    serial=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        if args.output.exists():
            p.error("refusing to overwrite")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(serial,encoding="utf-8")
    print(serial,end="")


if __name__=="__main__":
    main()
