"""Exact disjoint-pair + cardinality structured prediction up to K=255.

This is a RESTRICTED relational family: every candidate belongs to at most
one learned/known pair. Not a generic dense pairwise CRF. Exact count partition
and set MAP use polynomial block DP O(K^2), unlike dense pairwise enumeration.

Energy(Y) = sum_i s_i y_i + sum_(i,j) J_ij y_i y_j + a_|Y|.
For J=0 it is the count-anchor family; with J=a=0 and counts 0..K,
it is precisely independent BCE. Pair identities must come from a public
rule, a separately evaluated predictor, or learned content matching.
Never derive matching edges from gold labels at inference.
"""
from __future__ import annotations
import math


def _logadd(x,y):
    if x == -math.inf:
        return y
    if y == -math.inf:
        return x
    hi,lo=(x,y) if x >= y else (y,x)
    return hi + math.log1p(math.exp(lo-hi))


def _logsum(xs):
    result=-math.inf
    for x in xs:
        result=_logadd(result,x)
    return result


def _blocks(scores,pairs,weights):
    k=len(scores)
    if not 1 <= k <= 255 or any(type(x) not in (float,int) or not math.isfinite(x) for x in scores):
        raise ValueError("requires 1<=K<=255 finite candidate logits")
    if len(pairs)!=len(weights):
        raise ValueError("pair count != pair-weight count")
    partners={}
    for (i,j),jvalue in zip(pairs,weights):
        if (type(i) is not int or type(j) is not int or i==j or
            not (0<=i<k and 0<=j<k) or i in partners or j in partners or
            type(jvalue) not in (int,float) or not math.isfinite(jvalue)):
            raise ValueError("pairs must be disjoint valid endpoints with finite potentials")
        partners[i]=(j,jvalue)
        partners[j]=(i,jvalue)
    seen=set()
    blocks=[]
    for i in range(k):
        if i in seen:
            continue
        if i in partners:
            j,w=partners[i]
            seen.update((i,j))
            blocks.append(((0,0.0,0),
                           (1,float(scores[i]),1<<i),
                           (1,float(scores[j]),1<<j),
                           (2,float(scores[i]+scores[j]+w),(1<<i)|(1<<j))))
        else:
            seen.add(i)
            blocks.append(((0,0.0,0),(1,float(scores[i]),1<<i)))
    return blocks


def _protocol(k,residual,allowed):
    if residual is None:
        residual=[0.0]*(k+1)
    if len(residual)!=k+1 or any(type(v) not in (int,float)
                                 or not math.isfinite(v) for v in residual):
        raise ValueError("residual needs K+1 finite cardinality logits")
    if allowed is None:
        allowed=range(k+1)
    support=sorted(set(allowed))
    if not support or any(type(m) is not int or not 0<=m<=k for m in support):
        raise ValueError("invalid publicly declared count support")
    return list(residual),support


def exact_block_readout(scores,pairs=(),weights=(),residual=None,allowed=None):
    """Compute exact normalized count law and MAP over ALL legal sets, no 2^K enumeration.

    Returns selected indices and count log-probabilities. Complexity O(K^2)
    for a matching with bounded block size two. Stores one MAP bitmask per
    count; tie break uses smaller bitmask, then count.
    """
    k=len(scores)
    blocks=_blocks(scores,pairs,weights)
    residual,support=_protocol(k,residual,allowed)
    logz=[0.0]
    best={0:(0.0,0)}
    for block in blocks:
        next_z=[-math.inf]*(len(logz)+max(x[0] for x in block))
        next_best={}
        for m,old in enumerate(logz):
            for gain,energy,_mask in block:
                next_z[m+gain]=_logadd(next_z[m+gain],old+energy)
        for m,(old,mask) in best.items():
            for gain,energy,local_mask in block:
                t=m+gain
                cand=(old+energy,mask|local_mask)
                previous=next_best.get(t)
                if (previous is None or cand[0]>previous[0]+1e-12 or
                    (abs(cand[0]-previous[0])<=1e-12 and cand[1]<previous[1])):
                    next_best[t]=cand
        logz=next_z
        best=next_best
    ln=_logsum(logz[m]+residual[m] for m in support)
    log_count=[logz[m]+residual[m]-ln if m in support else None for m in range(k+1)]
    optimal_m=max(support,key=lambda m:(best[m][0]+residual[m],-m))
    _,mask=best[optimal_m]
    selected=[i for i in range(k) if mask>>i&1]
    return {"selected":selected,"map_count":optimal_m,
            "count_probs":[math.exp(lp) if lp is not None else 0.0 for lp in log_count],
            "count_log_probs":log_count,"log_partition":ln,"log_z_by_count":logz,
            "pair_blocks":len(pairs),"complexity":"O(K^2), disjoint pairs only"}


def exact_subset_nll(scores,gold,pairs=(),weights=(),residual=None,allowed=None):
    """Proper gold-set NLL; hard targets, NOT teacher-marginal BCE."""
    k=len(scores)
    if len(gold)!=k or any(type(y) is not int or y not in (0,1) for y in gold):
        raise ValueError("gold must be binary vector of length K")
    residual,support=_protocol(k,residual,allowed)
    selected={i for i,y in enumerate(gold) if y}
    if len(selected) not in support:
        raise ValueError("gold violates public count support")
    output=exact_block_readout(scores,pairs,weights,residual,support)
    energy=sum(scores[i] for i in selected)+residual[len(selected)]
    for (i,j),w in zip(pairs,weights):
        if i in selected and j in selected:
            energy+=w
    return output["log_partition"]-energy


def torch_block_log_partition(scores,pairs=(),weights=None,residual=None,allowed=None):
    """Differentiable O(K^2) PyTorch log Z, gradients through s, J, and a.

    Strictly requires disjoint pairs; selecting the matching is a separate
    model component and this routine does NOT learn an edge assignment.
    """
    import torch
    if scores.ndim!=1 or not scores.is_floating_point() or scores.numel()>255:
        raise ValueError("scores must be floating 1-D and length <=255")
    k=scores.numel()
    if weights is None:
        weights=scores.new_zeros(len(pairs))
    if isinstance(weights,(tuple,list)):
        weights=torch.stack([torch.as_tensor(w,dtype=scores.dtype,device=scores.device)
                             for w in weights]) if weights else scores.new_zeros(0)
    if weights.ndim!=1 or weights.numel()!=len(pairs):
        raise ValueError("incorrect number of pair-weight logits")
    # Validate disjoint indexing without reading differentiable tensor values.
    _blocks([0.0]*k,pairs,[0.0]*len(pairs))
    residual=scores.new_zeros(k+1) if residual is None else residual
    if residual.shape!=(k+1,):
        raise ValueError("residual must have K+1 logits")
    _,support=_protocol(k,[0.0]*(k+1),allowed)
    pair_lookup={i:(j,weights[n]) for n,(i,j) in enumerate(pairs)}
    pair_lookup.update({j:(i,weights[n]) for n,(i,j) in enumerate(pairs)})
    seen=set()
    old=scores.new_zeros(1)
    for i in range(k):
        if i in seen:
            continue
        if i in pair_lookup:
            j,w=pair_lookup[i]
            seen.update((i,j))
            local=(scores.new_zeros(()),torch.logaddexp(scores[i],scores[j]),scores[i]+scores[j]+w)
        else:
            seen.add(i)
            local=(scores.new_zeros(()),scores[i])
        degree=len(local)-1
        states=[]
        for shift,v in enumerate(local):
            # At least one finite term at every output count.
            states.append(torch.cat((
                old.new_full((shift,),-math.inf),old+v,
                old.new_full((degree-shift,),-math.inf))))
        next_old=states[0]
        for entry in states[1:]:
            next_old=torch.logaddexp(next_old,entry)
        old=next_old
    index=torch.tensor(support,dtype=torch.long,device=scores.device)
    return torch.logsumexp(old[index]+residual[index],dim=0)


def main():
    import json
    from .pairwise_exact import exact_distribution
    s=[0.1,-0.5,0.9,0.2]
    pairs=[(0,1),(2,3)]
    w=[1.3,-0.8]
    residual=[0.0,0.1,-0.3,-0.2,0.0]
    z=exact_block_readout(s,pairs,w,residual)
    q=exact_distribution(s,residual,[[0,w[0],0,0],[w[0],0,0,0],
                                       [0,0,0,w[1]],[0,0,w[1],0]])
    expected=-math.log(sum(math.exp(0) for _ in [])) if False else 0.0
    # Use elementary joint probability to certify the normalized DP.
    gold_set=[1,1,0,0]
    nll=exact_subset_nll(s,gold_set,pairs,w,residual)
    assert abs(math.exp(-nll)-q[0b0011])<1e-11
    print(json.dumps({"status":"SYNTHETIC_CPU_REFERENCE","nll_ab":nll,
                      "p_ab":math.exp(-nll),"map":z["selected"],
                      "log_partition":z["log_partition"]},indent=2))


if __name__=="__main__":
    main()
