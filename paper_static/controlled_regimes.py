"""Auditable synthetic families for independent, cardinality, and pairwise decisions.

DO NOT report synthetic oracle scores as real LLM results. Split by *context*
before drawing repeated answer sets; option text never exposes oracle logits.
Data rows use the existing multichoice unified format with allow_empty=True.
Truth/reference distributions are written separately and are NEVER student input.

For 255 option scalability use block_pair_dp.py (this generator deliberately
limits exact oracle enumeration to K<=10). The three underlying joint laws:
 I:  p(Y|x) proportional to exp(sum s_i y_i)
 C:  p(Y|x) proportional to exp(sum s_i y_i + a_|Y|)
 R:  p(Y|x) proportional to exp(sum s_i y_i + a_|Y| + sum J_ij y_i y_j)
    with a rewired perfect matching; all candidates exchangeable, so q_i
    and rho_m are invariant under rewiring, unlike pair co-occurrences.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from .pairwise_exact import exact_distribution, moments

ALPHABET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def matchings(nodes):
    """Enumerate perfect matchings; small K only."""
    nodes=tuple(nodes)
    if not nodes:
        yield ()
        return
    head=nodes[0]
    for pos in range(1,len(nodes)):
        other=nodes[pos]
        for rest in matchings(nodes[1:pos]+nodes[pos+1:]):
            yield ((head,other),)+rest


def pairing_pools(k):
    if k%2 or not 6<=k<=10:
        raise ValueError("generator uses even K in [4,10]; 255 scaling is a separate DP check")
    pools={"train":[],"dev":[],"test":[]}
    for matching in matchings(range(k)):
        signature=json.dumps(matching,separators=(",",":"))
        h=int.from_bytes(hashlib.sha256(signature.encode()).digest()[:8],"big")
        # ~70/15/15 pair-structure disjoint, not merely question-ID disjoint.
        part=("train" if h%20<14 else "dev" if h%20<17 else "test")
        pools[part].append(matching)
    if not all(pools.values()):
        raise ValueError("not enough pairings for disjoint train/dev/test structures")
    return pools


def _options(rng,k,tags=None, utilities=None):
    order=list(range(k))
    rng.shuffle(order)
    if tags is None:
        tags=["none"]*k
    if utilities is None:
        utilities=[0]*k
    # Expose observable features, not teacher logits; shuffle irrelevant IDs only.
    return [{"key":ALPHABET[i],
             "text":f"Candidate {order[i]:02d}, utility feature {utilities[i]}, "
                    f"compatibility code {tags[i]}"} for i in range(k)]


def _choice(rng,distribution):
    r=rng.random()
    cumulative=0.0
    for mask,p in sorted(distribution.items()):
        cumulative+=p
        if r<=cumulative:
            return mask
    return max(distribution)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out, *, seed=20261030, ntrain=240, ndev=60, ntest=60, repeats=4, k=8):
    if k not in (6,8,10) or any(type(v) is not int or v<=0
       for v in (ntrain,ndev,ntest,repeats)):
        raise ValueError("invalid K or context count")
    out=Path(out)
    if out.exists():
        raise ValueError("output directory already exists; refuse overwrite")
    rng=random.Random(seed)
    pools=pairing_pools(k)
    allocations={"train":ntrain,"dev":ndev,"test":ntest}
    paths={split:[] for split in allocations}
    oracle={split:[] for split in allocations}
    seen_contexts=set()
    used_graphs={split:set() for split in allocations}
    for split,n in allocations.items():
        for regime in ("I","C","R"):
            for t in range(n):
                cid=f"{split}-{regime}-{t:05d}"
                assert cid not in seen_contexts
                seen_contexts.add(cid)
                utilities=[rng.choice([-2,-1,0,1,2]) for _ in range(k)]
                scores=[0.8*u for u in utilities]
                residual=[0.0]*(k+1)
                weights=[]
                pairings=[]
                tags=["none"]*k
                strength=0.0
                if regime=="C":
                    # Count-only dependence is identifiable without oracle counts in prompts.
                    load=rng.choice(["low","medium","high"])
                    center={"low":1.8,"medium":k*.48,"high":k*.75}[load]
                    strength_label=rng.choice(["mild","medium","strong"])
                    strength={"mild":.5,"medium":1.0,"strong":1.8}[strength_label]
                    residual=[-strength*(m-center)**2 for m in range(k+1)]
                    rule=(f"Shared workload regime {load}; collective pressure {strength_label}. "
                          "Option utility features are assessed jointly.")
                elif regime=="R":
                    # Make q_i and count rho invariant to matching *re-wiring*.
                    scores=[0.0]*k
                    utilities=[0]*k
                    pairings=list(rng.choice(pools[split]))
                    used_graphs[split].add(tuple(pairings))
                    strength_label=rng.choice(["mild","medium","strong"])
                    strength={"mild":1.5,"medium":3.0,"strong":4.5}[strength_label]
                    weights=[strength]*len(pairings)
                    residual=[-.9*(m-2.0)**2 for m in range(k+1)]
                    for pair_number,(i,j) in enumerate(pairings):
                        code=f"g{pair_number:02d}"
                        tags[i]=tags[j]=code
                    rule=(f"Items sharing a compatibility code tend to be accepted together; "
                          f"coupling strength {strength_label}.")
                    load="paired"
                else:
                    load="independent"
                    rule="Each candidate is accepted or rejected independently."
                options=_options(rng,k,tags,utilities)
                matrix=[[0.0]*k for _ in range(k)]
                for (i,j),weight in zip(pairings,weights):
                    matrix[i][j]=matrix[j][i]=weight
                truth=exact_distribution(scores,residual,matrix)
                stats=moments(truth,k)
                public={"state":f"Selection task. {rule} Workload={load}.",
                        "question":"Select all appropriate candidates; zero selections are allowed.",
                        "options":options,"selection_mode":"multi","allow_empty":True}
                oracle[split].append({
                    "context_id":cid,"regime":regime,"strength":strength,
                    "teacher_scores":scores,"teacher_residual":residual,
                    "teacher_pairs":pairings,"teacher_pair_weights":weights,
                    "true_marginals":stats["marginals"],
                    "true_count_probs":stats["count_probs"],
                    "true_pairwise_marginals":stats["pairwise_marginals"],
                    "true_set_probs":{str(mask):prob for mask,prob in truth.items()}})
                for repeat in range(repeats):
                    sample=_choice(rng,truth)
                    item={**public,"id":f"{cid}/draw-{repeat:02d}",
                          "group_id":cid,"source":f"controlled_{regime}",
                          "lang":"en","split":split,
                          "target":[int(bool(sample>>i&1)) for i in range(k)]}
                    paths[split].append(item)
    # Split is context-disjoint AND pair-matching topology-disjoint in R.
    if any(used_graphs[a]&used_graphs[b]
           for a,b in (("train","dev"),("train","test"),("dev","test"))):
        raise AssertionError("pairing topology leakage")
    out.mkdir(parents=True)
    data_dir=out/"student"
    oracle_dir=out/"oracle_DO_NOT_USE_AS_MODEL_INPUT"
    data_dir.mkdir()
    oracle_dir.mkdir()
    written={}
    for split in allocations:
        data_file=data_dir/f"{split}.jsonl"
        truth_file=oracle_dir/f"{split}.jsonl"
        data_file.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in paths[split]),encoding="utf-8")
        truth_file.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in oracle[split]),encoding="utf-8")
        written[split]={"contexts":allocations[split]*3,
                        "rows":len(paths[split]),
                        "student_sha256":_sha(data_file),
                        "oracle_sha256":_sha(truth_file),
                        "pair_structures":len(used_graphs[split])}
    manifest={"status":"SYNTHETIC_ORACLE_DATA_ONLY_NOT_TRAINED_MODEL",
              "seed":seed,"k":k,"repeats":repeats,"regimes":["I","C","R"],
              "split_by":"context; pair matching structure disjoint",
              "forbidden_leakage":"never include oracle files or its scores/gradients in model prompt",
              "files":written,"generation_schema":"controlled_joint_set_v1"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--ntrain",type=int,default=240)
    p.add_argument("--ndev",type=int,default=60)
    p.add_argument("--ntest",type=int,default=60)
    p.add_argument("--repeats",type=int,default=4)
    p.add_argument("--k",type=int,default=8)
    a=p.parse_args(argv)
    result=generate(a.output_dir,seed=a.seed,ntrain=a.ntrain,ndev=a.ndev,
                    ntest=a.ntest,repeats=a.repeats,k=a.k)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
