"""A transparent environment-search baseline, NOT learned latent imagination.

Uses the exact Sokoban simulator for training/debug only; do not call it an
ARC-AGI-3-valid hidden-state searcher. No learned value/world model is used unless
the caller supplies them explicitly.
"""
from dataclasses import dataclass
from math import log
from treejev.sokoban import ACTION_ORDER

@dataclass
class BeamResult:
    won: bool
    actions: list
    expanded: int
    depth: int
    width: int

def beam_search(board, state, policy, *, width=4, horizon=80,
                max_expanded=20000, estimated_remaining=None):
    if width<1 or horizon<0 or max_expanded<1:
        raise ValueError("invalid search budget")
    # One node owns one immutable environment state and one independent path.
    frontier=[(0.0,state,())]; expanded=0
    if board.won(state):
        return BeamResult(True,[],0,0,width)
    for depth in range(horizon):
        candidates={}
        for score,current,path in frontier:
            if expanded>=max_expanded: break
            expanded+=1
            raw=policy(current)
            legal=board.legal_actions(current)
            if set(raw)!=set(legal):
                raise ValueError("policy scores must cover precisely legal actions")
            for action in ACTION_ORDER:
                if action not in legal: continue
                child=board.next_state(current,action)
                newpath=path+(action,)
                if board.won(child):
                    return BeamResult(True,list(newpath),expanded,len(newpath),width)
                if child in candidates and candidates[child][0]>=score+raw[action]:
                    continue
                heuristic=(estimated_remaining(child) if estimated_remaining else 0.0)
                candidates[child]=(score+raw[action],newpath,heuristic)
        if not candidates or expanded>=max_expanded:
            break
        ranked=sorted(candidates.items(),key=lambda kv:(
            -(kv[1][0]-kv[1][2]),len(kv[1][1]),kv[1][1]))
        frontier=[(value[0],key,value[1]) for key,value in ranked[:width]]
    return BeamResult(False,[],expanded,horizon,width)
