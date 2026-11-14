#!/usr/bin/env python3
"""Convert official RoMQA CLOSED Train/DEV into existing Qwen multi-select rows.

This is a converter, NOT a new or synthetic QA benchmark. It never opens blind
TEST or fills candidates from complete_answer. Evidence comes only from the
official candidate DPR snippets, with a deterministic, gold-independent budget.

Official data: facebookresearch/romqa data/closed/top_20.{train,dev}.json.bz2.
Use the original archive or official reconstructed files; access is NOT assumed.
The archive must actually be obtained and verified before any real experiment.

Example:
  python -m paper_static.romqa_data --archive /path/romqa_data.zip \
    --tokenizer /path/Qwen3.5-4B --output data/romqa_closed_v1 \
    --max-input-tokens 16384 --evidence-chars 256

No GPU, no training, no test access. Fails loudly on missing data, duplicates,
candidate K deviations or insufficient context; never silently excludes rows.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from paper_static.romqa_data_screen import load_zip, load_dir, screen
from paper_static.eurlex_data import readout_labels, render_prompt, digest_file


DATASET = "facebookresearch/romqa"
SOURCE = "romqa_closed"
SOURCE_SPLITS = ("train", "dev")
SCHEMA = "romqa_closed_decision_v1"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(path, values):
    with path.open("x", encoding="utf-8") as f:
        for item in values:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def _entity_text(candidate, ident):
    if not isinstance(candidate, dict):
        raise ValueError(f"{ident}: candidate must be an object")
    name = candidate.get("text")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{ident}: candidate canonical entity text missing")
    if name != name.strip() or "\n" in name or "\r" in name:
        raise ValueError(f"{ident}: malformed candidate canonical entity text")
    if type(candidate.get("is_answer")) is not bool:
        raise ValueError(f"{ident}: Train/DEV requires true binary is_answer")
    return name


def _dpr_first(candidate, ident, threshold=.65):
    snippets = candidate.get("dpr", [])
    similarities = candidate.get("dpr_score", [])
    if snippets is None:
        snippets = []
    if similarities is None:
        similarities = []
    if not isinstance(snippets, list) or not isinstance(similarities, list):
        raise ValueError(f"{ident}: DPR evidence/score must be aligned lists")
    if len(similarities) != len(snippets):
        raise ValueError(f"{ident}: DPR snippets/similarity length mismatch")
    if any(not isinstance(s, str) for s in snippets):
        raise ValueError(f"{ident}: invalid DPR evidence string")
    if any(type(s) not in (int, float) or not math.isfinite(s) for s in similarities):
        raise ValueError(f"{ident}: invalid DPR similarity")
    for passage, similarity in zip(snippets, similarities):
        if similarity >= threshold and passage.strip():
            return " ".join(passage.split())
    return next((" ".join(p.split()) for p in snippets if p.strip()), "")


def _as_record(source, split, *, evidence_mode="dpr-first", expected_k=100):
    """Assemble raw, gold-independent content and separately held-out labels."""
    if not isinstance(source, dict):
        raise ValueError(f"{split}: expected source object")
    ident = str(source.get("id", ""))
    cluster = str(source.get("cluster_id", ""))
    question = source.get("question")
    if not ident or ident == "None" or not cluster or cluster == "None":
        raise ValueError(f"{split}: missing id/cluster_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{ident}: empty question")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
        raise ValueError(f"{ident}: need full 2..255 candidate entity list")
    if expected_k and len(candidates) != expected_k:
        raise ValueError(f"{ident}: K={len(candidates)} differs from locked K={expected_k}")
    if evidence_mode not in ("dpr-first", "no-evidence"):
        raise ValueError("unsupported evidence mode")
    names, targets, evidence = [], [], []
    for c in candidates:
        names.append(_entity_text(c, ident))
        targets.append(int(c["is_answer"]))
        evidence.append(_dpr_first(c, ident) if evidence_mode == "dpr-first" else "")
    if len(names) != len(set(names)):
        raise ValueError(f"{ident}: duplicate canonical candidates; cannot map Gold")
    full = source.get("complete_answer")
    if full is not None and (not isinstance(full, list) or any(
            not isinstance(a, dict) or not isinstance(a.get("text"), str)
            for a in full)):
        raise ValueError(f"{ident}: malformed official complete_answer")
    complete_names = [a["text"] for a in full] if full is not None else None
    if complete_names is not None and len(set(complete_names)) != len(complete_names):
        raise ValueError(f"{ident}: duplicate complete_answer names")
    row = {
        "id": ident,
        "group_id": cluster,
        "source": SOURCE,
        "lang": "en",
        "split": split,
        "state": "Closed-world entity selection. Use only the question, entity names and listed retrieved evidence.",
        "question": question.strip(),
        "selection_mode": "multi",
        "allow_empty": True,
        "n_options": len(names),
        "target": targets,
        "options": [{"key": name, "desc": ""} for name in names],
    }
    sidecar = {
        "id": ident, "cluster_id": cluster,
        "candidate_keys": names,
        "complete_answer": complete_names,
        "candidate_gold": [n for n, t in zip(names, targets) if t],
    }
    return row, evidence, sidecar


def _with_evidence(row, snippets, char_limit):
    options = []
    for old, passage in zip(row["options"], snippets):
        excerpt = passage[:char_limit] if char_limit > 0 else ""
        desc = ("Entity: " + old["key"]
                + ("; Retrieved evidence: " + excerpt if excerpt else ""))
        options.append({"key": old["key"], "desc": desc})
    return {**row, "options": options}


def fit_prompt(row, snippets, tokenizer, labels, *, max_input_tokens=16384,
               evidence_chars=256):
    """Fair per-candidate DPR clipping; full candidate identities NEVER dropped."""
    if len(snippets) != len(row["options"]):
        raise ValueError("snippet/candidate mismatch")
    if max_input_tokens < 512 or evidence_chars < 0:
        raise ValueError("invalid prompt budget")
    def length(cap):
        candidate = _with_evidence(row, snippets, cap)
        text = render_prompt(tokenizer, candidate, labels)
        return candidate, len(tokenizer.encode(text, add_special_tokens=False))
    baseline, baseline_len = length(0)
    if baseline_len > max_input_tokens:
        raise ValueError(f"{row['id']}: full candidate names exceed budget "
                         f"without evidence ({baseline_len}>{max_input_tokens})")
    if evidence_chars == 0 or not any(snippets):
        return baseline, {"prompt_tokens": baseline_len, "evidence_chars_used": 0,
                          "evidence_available": sum(bool(s) for s in snippets),
                          "evidence_retained": 0}
    full, full_len = length(evidence_chars)
    if full_len <= max_input_tokens:
        return full, {"prompt_tokens": full_len, "evidence_chars_used": evidence_chars,
                      "evidence_available": sum(bool(s) for s in snippets),
                      "evidence_retained": sum(bool(s[:evidence_chars]) for s in snippets)}
    low, high = 0, evidence_chars
    best, actual, cap = baseline, baseline_len, 0
    while low <= high:
        middle = (low + high) // 2
        candidate, size = length(middle)
        if size <= max_input_tokens:
            best, actual, cap = candidate, size, middle
            low = middle + 1
        else:
            high = middle - 1
    if actual > max_input_tokens:
        raise AssertionError("deterministic evidence truncation budget failed")
    return best, {"prompt_tokens": actual, "evidence_chars_used": cap,
                  "evidence_available": sum(bool(s) for s in snippets),
                  "evidence_retained": sum(bool(s[:cap]) for s in snippets)}


def convert(splits, tokenizer, labels, *, expected_k=100, evidence_mode="dpr-first",
            max_input_tokens=16384, evidence_chars=256):
    if set(splits) != set(SOURCE_SPLITS):
        raise ValueError("Only official TRAIN and DEV are allowed (never TEST)")
    by_split = {}
    sidecars = {}
    audits = {}
    ids, cluster_owner = set(), {}
    for split in SOURCE_SPLITS:
        records = splits[split]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{split}: official data not loaded")
        rows, gold, counters, lengths, count_hist = [], [], Counter(), [], Counter()
        for source in records:
            row, snippets, sidecar = _as_record(
                source, split, evidence_mode=evidence_mode, expected_k=expected_k)
            ident, cluster = row["id"], row["group_id"]
            if ident in ids:
                raise ValueError(f"duplicate official sample id: {ident}")
            if cluster in cluster_owner and cluster_owner[cluster] != split:
                raise ValueError(f"cluster leakage train/dev: {cluster}")
            cluster_owner[cluster] = split
            ids.add(ident)
            item, sizes = fit_prompt(
                row, snippets, tokenizer, labels, max_input_tokens=max_input_tokens,
                evidence_chars=evidence_chars)
            rows.append(item)
            gold.append(sidecar)
            count_hist[sum(item["target"])] += 1
            lengths.append(sizes["prompt_tokens"])
            counters["candidate_total"] += len(item["options"])
            counters["dpr_available"] += sizes["evidence_available"]
            counters["dpr_visible"] += sizes["evidence_retained"]
            counters["prompt_evidence_clipped"] += sizes["evidence_chars_used"] < evidence_chars
            counters["full_answer_not_in_candidate"] += (len(set(sidecar["complete_answer"] or [])
                                                            -set(sidecar["candidate_keys"]))
                                                         if sidecar["complete_answer"] is not None else 0)
            counters["full_answer_unknown"] += sidecar["complete_answer"] is None
        by_split[split] = rows
        sidecars[split] = gold
        audits[split] = {
            "n":len(rows),"n_clusters":len({r["group_id"] for r in rows}),
            "candidate_K_histogram":dict(sorted(Counter(r["n_options"] for r in rows).items())),
            "gold_count_histogram":dict(sorted(count_hist.items())),
            "empty_fraction":count_hist[0]/len(rows),
            "multi_answer_fraction":sum(n for k,n in count_hist.items() if k >= 2)/len(rows),
            "min_prompt_tokens":min(lengths),
            "max_prompt_tokens":max(lengths),
            "mean_prompt_tokens":sum(lengths)/len(lengths),
            **dict(counters),
        }
    return by_split, sidecars, audits


def _cluster_sample(rows, n_clusters, seed):
    if not isinstance(n_clusters, int) or n_clusters < 0:
        raise ValueError("pilot cluster count must be nonnegative")
    if n_clusters == 0:
        return []
    groups = sorted({r["group_id"] for r in rows})
    if n_clusters > len(groups):
        raise ValueError("pilot requests more clusters than exist")
    chosen = set(random.Random(seed).sample(groups, n_clusters))
    return [r for r in rows if r["group_id"] in chosen]


def prepare(output, splits, tokenizer, labels, *, expected_k=100,
            evidence_mode="dpr-first", max_input_tokens=16384,
            evidence_chars=256, pilot_train_clusters=0, pilot_dev_clusters=0,
            provenance=None):
    output = Path(output)
    if output.exists():
        raise ValueError("refuse to overwrite an existing derived dataset")
    # Validate the real official schema and global ID/cluster integrity first.
    screening = screen(splits)
    if screening["splits"]["train"]["duplicate_candidate_text_question_count"] or \
       screening["splits"]["dev"]["duplicate_candidate_text_question_count"]:
        raise ValueError("duplicate original canonical entity names")
    rows, sidecars, audits = convert(
        splits, tokenizer, labels, expected_k=expected_k,
        evidence_mode=evidence_mode, max_input_tokens=max_input_tokens,
        evidence_chars=evidence_chars)
    pilots = {
        "train":_cluster_sample(rows["train"],pilot_train_clusters,43),
        "dev":_cluster_sample(rows["dev"],pilot_dev_clusters,44),
    }
    # Refuse silently treating a zero-sized pilot as training data.
    output.mkdir(parents=True)
    try:
        manifest = {
            "schema":SCHEMA,"dataset":DATASET,"source_setting":"closed",
            "source_filenames":["top_20.train.json.bz2","top_20.dev.json.bz2"],
            "test_opened":False,"source":provenance or {},
            "evidence_mode":evidence_mode,"evidence_char_cap":evidence_chars,
            "max_input_tokens":max_input_tokens,"expected_k":expected_k,
            "tokenizer":getattr(tokenizer,"name_or_path",None),
            "tokenizer_chat_template_sha256":_sha(str(tokenizer.chat_template)),
            "readout_labels_sha256":_sha(json.dumps(labels,ensure_ascii=False)),
            "split_audit":audits,"pilot":{},
            "files":{},
            "screen":screening,
            "candidate_injection":False,
            "test_model_selection_prohibited":True,
        }
        (output/"evaluation_gold_only").mkdir()
        for split in SOURCE_SPLITS:
            file = output/f"{split}.jsonl"
            side = output/"evaluation_gold_only"/f"{split}.jsonl"
            _jsonl(file,rows[split])
            _jsonl(side,sidecars[split])
            manifest["files"][split] = {
                "name":file.name,"n":len(rows[split]),"sha256":digest_file(file),
                "eval_sidecar":str(side.relative_to(output)),
                "eval_sha256":digest_file(side),
            }
        if any(pilots.values()):
            (output/"pilot").mkdir()
            for split, selected in pilots.items():
                if selected:
                    path = output/"pilot"/f"{split}.jsonl"
                    _jsonl(path,selected)
                    manifest["pilot"][split] = {
                        "path":str(path.relative_to(output)),
                        "sha256":digest_file(path),
                        "n":len(selected),
                        "clusters":len({r["group_id"] for r in selected}),
                    }
        (output/"manifest.json").write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
    except Exception:
        # Fail closed: retain partial files for investigation, without manifest.
        raise
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive",type=Path)
    src.add_argument("--data-dir",type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--expected-k",type=int,default=100)
    p.add_argument("--max-input-tokens",type=int,default=16384)
    p.add_argument("--evidence-mode",choices=["dpr-first","no-evidence"],default="dpr-first")
    p.add_argument("--evidence-chars",type=int,default=256)
    p.add_argument("--pilot-train-clusters",type=int,default=0)
    p.add_argument("--pilot-dev-clusters",type=int,default=0)
    args=p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite output")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=Path(args.tokenizer).is_dir())
    if tok.chat_template is None:
        p.error("model tokenizer must provide a chat template")
    labels,_=readout_labels(tok)
    if args.archive is not None:
        if not args.archive.is_file():
            p.error("RoMQA third-party ZIP not present: data NOT verified/downloaded")
        splits=load_zip(args.archive)
        origin={"kind":"third_party_zip","path":str(args.archive),
                "sha256":digest_file(args.archive)}
    else:
        splits=load_dir(args.data_dir)
        origin={"kind":"official_reconstruction_dir",
                "train_sha256":digest_file(args.data_dir/"top_20.train.json.bz2"),
                "dev_sha256":digest_file(args.data_dir/"top_20.dev.json.bz2")}
    result=prepare(args.output,splits,tok,labels,
                   expected_k=args.expected_k,evidence_mode=args.evidence_mode,
                   max_input_tokens=args.max_input_tokens,evidence_chars=args.evidence_chars,
                   pilot_train_clusters=args.pilot_train_clusters,
                   pilot_dev_clusters=args.pilot_dev_clusters,provenance=origin)
    print(json.dumps({"status":"TRAIN_DEV_DERIVED_NOT_TRAINED",
                      "output":str(args.output),
                      "train":result["split_audit"]["train"],
                      "dev":result["split_audit"]["dev"],
                      "test_opened":False},indent=2))


if __name__=="__main__":
    main()
