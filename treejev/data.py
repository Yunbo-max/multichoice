"""Adapters for exact Sokoban oracle and Valen image-action data.

Never use Valen meta symbolic states or hidden certificates as model input.
Official eval_sokoban is NEVER used for train/dev; its 100 game test set and 500
single-step evaluations overlap each other and must not be reported as independent.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import random
from treejev.sokoban import ACTION_ORDER, canonical_layout_group
from treejev.oracle import shortest_path_dag

OPTION_DESCRIPTIONS = {"up":"Move Up", "right":"Move Right",
                       "down":"Move Down", "left":"Move Left"}

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fd:
        for i, line in enumerate(fd,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f"{path}:{i}: {e}") from e

def save_jsonl(path, rows):
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(f"refusing to replace {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w",encoding="utf-8") as fd:
        for record in rows:
            fd.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")

def split_groups(rows, dev_fraction=0.1, seed=43):
    """Split whole canonical boards, never individual states/images."""
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0,1)")
    by_group = defaultdict(list)
    for row in rows:
        if not row.get("group_id"):
            raise ValueError("all examples require nonempty group_id")
        by_group[row["group_id"]].append(row)
    keys = sorted(by_group)
    random.Random(seed).shuffle(keys)
    n_dev = min(len(keys)-1, max(1, round(len(keys)*dev_fraction))) if dev_fraction and len(keys)>1 else 0
    dev_ids = set(keys[:n_dev])
    train = [r for r in rows if r["group_id"] not in dev_ids]
    dev = [r for r in rows if r["group_id"] in dev_ids]
    if set(r["group_id"] for r in train) & set(r["group_id"] for r in dev):
        raise AssertionError("train/dev board leak")
    return train, dev

def ensure_action_probs(probabilities):
    if not isinstance(probabilities,dict):
        raise ValueError("next_action.probabilities must be a direction->probability dict")
    lowered = {str(k).lower():float(v) for k,v in probabilities.items()}
    if set(lowered) != set(ACTION_ORDER):
        raise ValueError("expected probability for every move: up/right/down/left")
    if any(not (0 <= x <= 1) for x in lowered.values()) or abs(sum(lowered.values())-1)>1e-5:
        raise ValueError("invalid optimal-action probability distribution")
    positive=[a for a in ACTION_ORDER if lowered[a]>0]
    if not positive:
        raise ValueError("no optimal move")
    if len({round(lowered[a],8) for a in positive})!=1:
        raise ValueError("positive tied optimal actions should share equal mass")
    return {a:lowered[a] for a in ACTION_ORDER}

def valen_records(train_jsonl, assets_root, *, require_images=True):
    path=Path(train_jsonl)
    if "eval_sokoban" in path.parts:
        raise ValueError("official eval_sokoban may not be used as training data")
    root=Path(assets_root).resolve()
    records=[]; seen=set()
    for idx,record in enumerate(read_jsonl(path)):
        group=str(record.get("group_id",""))
        ident=str(record.get("id") or f"valen-{idx}")
        if not group or ident in seen:
            raise ValueError("missing group_id or duplicate sample ID")
        seen.add(ident)
        probs=ensure_action_probs(record["targets"]["next_action"]["probabilities"])
        assets=record.get("assets")
        if not isinstance(assets,list) or not assets:
            raise ValueError("Valen image asset missing")
        rel=assets[0]["path"]
        resolved=(root/rel).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("unsafe asset path escapes asset root")
        if require_images and not resolved.is_file():
            raise FileNotFoundError(resolved)
        records.append({
            "id":ident,"group_id":group,"source":"valen_sokoban",
            "image_path":str(resolved),"target_probs":probs,
            "optimal_actions":[a for a in ACTION_ORDER if probs[a]>0],
            "input_modality":"image",
            # No symbolic states/solutions, failure certificates, or test labels.
        })
    return records

def as_multiselect_row(board, state, optimal, level_id, depth, remaining, split):
    """Gold multi-hot target: what moves could BEGIN a shortest solution?

    This is NOT a single executable multi-action. At deployment choose exactly
    one action among positive predictions.
    """
    positives=set(optimal)
    if not positives or any(a not in ACTION_ORDER for a in positives):
        raise ValueError("must have at least one legal optimal action")
    ascii_state=board.render_ascii(state)
    key=sha256((str(level_id)+ascii_state).encode()).hexdigest()[:20]
    target=[int(a in positives) for a in ACTION_ORDER]
    return {
        "id":f"tree-{level_id}-{key}","group_id":canonical_layout_group(board),
        "source":"sokoban_oracle","lang":"en","split":split,
        "state":ascii_state,"question":"Select ALL actions that begin a shortest valid solution to the Sokoban goal.",
        "options":[{"key":chr(ord("a")+j),"desc":OPTION_DESCRIPTIONS[a]}
                   for j,a in enumerate(ACTION_ORDER)],
        "target":target,"n_options":len(ACTION_ORDER),
        "gold_letters":"".join(chr(ord("a")+j) for j,y in enumerate(target) if y),
        "selection_mode":"multi","allow_empty":False,
        "optimal_actions":[a for a in ACTION_ORDER if a in positives],
        "distance_to_goal":remaining,"oracle_depth":depth,
        # The last two fields are AUDIT ONLY; make_prompt uses state/question/options.
    }

def generate_oracle_level(level_id, board, initial, *, max_nodes=20000,
                          max_depth=80, trajectories=16):
    dag=shortest_path_dag(board,initial,max_nodes=max_nodes,max_depth=max_depth)
    rows=[]
    for state in dag.good_states():
        if board.won(state): continue
        rows.append(as_multiselect_row(board,state,dag.optimal[state],
                  level_id,dag.depths[state],dag.distance-dag.depths[state],"train"))
    paths=[{"level_id":str(level_id),"group_id":canonical_layout_group(board),
            "actions":p,"steps":len(p),"verified_success":True}
           for p in dag.trajectories(limit=trajectories)]
    return rows,paths,{"level_id":str(level_id),"distance":dag.distance,
                        "solved_paths_capped":dag.count_paths(),
                        "optimal_states":len(rows),"sampled_paths":len(paths),
                        "complete_shortest_dag":True}
