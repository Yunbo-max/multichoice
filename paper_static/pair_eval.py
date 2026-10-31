#!/usr/bin/env python
"""Evaluate a learned-pair (objective=pair) checkpoint on controlled DEV.

Loads the pair head, forwards the DEV rows, and reports, per regime:
- joint NLL of the model's dense-J set law against the oracle distribution
- count KL against the oracle count law
- for R regime: edge precision/recall/F1 of the greedy top-disjoint-pairs
  inferred from J, versus the oracle teacher_pairs.

Requires GPU. Never reads TEST here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import torch  # noqa: E402


def _greedy_pairs(J, k, max_pairs):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((abs(J[i][j]), i, j))
    edges.sort(reverse=True)
    used = set()
    chosen = []
    for _, i, j in edges:
        if i not in used and j not in used:
            chosen.append((i, j))
            used.add(i)
            used.add(j)
            if len(chosen) >= max_pairs:
                break
    return chosen


def _edge_f1(true_pairs, pred_pairs):
    true = {tuple(sorted(p)) for p in true_pairs}
    pred = {tuple(sorted(p)) for p in pred_pairs}
    tp = len(true & pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)

    import train_multiselect as trainer
    from set_distillation import log_partitions
    from block_pair_dp import exact_block_readout, exact_subset_nll
    from multiselect_metrics import read_dataset
    from eval_multiselect import make_batches, validate_right_padding, checkpoint_count_anchor
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("pair eval requires CUDA")

    student = args.data_root / "student" / f"{args.split}.jsonl"
    oracle = args.data_root / "oracle_DO_NOT_USE_AS_MODEL_INPUT" / f"{args.split}.jsonl"
    rows = read_dataset(student, args.split)
    if args.limit:
        rows = rows[: args.limit]
    oracles = {r["context_id"]: r for r in (json.loads(l) for l in oracle.read_text().splitlines())}

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = trainer.option_labels(tokenizer)
    label_ids = [tokenizer.encode(lbl, add_special_tokens=False)[0] for lbl in labels]
    prompts = [trainer.make_prompt(tokenizer, row, labels) for row in rows]
    lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    batches = make_batches([r["id"] for r in rows], lengths, args.batch_size, 8192, 8192)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else Qwen3_5ForCausalLM
    base = cls.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                               local_files_only=True)
    base.requires_grad_(False)
    base.config.use_cache = False
    wrapper = trainer.Readout(base, label_ids, full_rank=True, objective="pair", pair_pairs=28)
    trainer.load_count_head(wrapper, args.model)
    count_anchor = checkpoint_count_anchor(args.model)
    wrapper = wrapper.to("cuda").eval()

    scores_by = {}
    head_by = {}
    pair_by = {}
    with torch.inference_mode():
        for batch in batches:
            enc = tokenizer(
                [prompts[index] for index in batch],
                padding=True, return_tensors="pt", add_special_tokens=False,
            )
            validate_right_padding(enc["attention_mask"].tolist(), [lengths[i] for i in batch])
            logits, count_logits, pair_logits = wrapper(
                input_ids=enc["input_ids"].to("cuda"),
                attention_mask=enc["attention_mask"].to("cuda"),
            )
            for pos, index in enumerate(batch):
                row = rows[index]
                k = len(row["options"])
                scores_by[row["id"]] = logits[pos, :k].float().cpu().tolist()
                head_by[row["id"]] = count_logits[pos, : k + 1].float().cpu().tolist()
                pair_by[row["id"]] = pair_logits[pos, : k * (k - 1) // 2].float().cpu().tolist()

    # ---- per-context scoring against oracle ----
    contexts = {}
    for row in rows:
        contexts.setdefault(row["group_id"], []).append(row)

    records = []
    for cid, group in contexts.items():
        o = oracles[cid]
        regime = o["regime"]
        k = len(group[0]["options"])
        true_pairs = [tuple(sorted(e)) for e in o.get("teacher_pairs", [])]
        joint_nlls = []
        count_kls = []
        edge_f1s = []
        for row in group:
            scores = scores_by[row["id"]]
            head = head_by[row["id"]]
            plog = pair_by[row["id"]]
            if count_anchor:
                center = max(scores)
                logz = log_partitions([s - center for s in scores])
                head = [head[m] + logz[m] + m * center for m in range(k + 1)]
            # greedy disjoint matching from pair weights (mirrors training)
            ii = [a for a in range(k) for b in range(a + 1, k)]
            jj = [b for a in range(k) for b in range(a + 1, k)]
            order = sorted(range(len(plog)), key=lambda n: abs(plog[n]), reverse=True)
            used = set()
            sel_pairs = []
            sel_weights = []
            for n in order:
                a, b = ii[n], jj[n]
                if a not in used and b not in used:
                    sel_pairs.append((a, b))
                    sel_weights.append(plog[n])
                    used.add(a)
                    used.add(b)
            out = exact_block_readout(scores, sel_pairs, sel_weights, head, list(range(k + 1)))
            # joint NLL of gold under the block law
            gold = row["target"]
            nll = exact_subset_nll(scores, gold, sel_pairs, sel_weights, head, list(range(k + 1)))
            joint_nlls.append(nll)
            # count KL
            rho_model = out["count_probs"]
            rho_true = o["true_count_probs"]
            kl = sum(p * math.log(p / max(q, 1e-12)) for p, q in zip(rho_true, rho_model) if p > 0)
            count_kls.append(kl)
            if regime == "R" and true_pairs:
                edge_f1s.append(_edge_f1(true_pairs, sel_pairs))
        records.append({
            "context": cid, "regime": regime,
            "joint_nll": sum(joint_nlls) / len(joint_nlls),
            "count_kl": sum(count_kls) / len(count_kls),
            "edge_f1": (sum(x[2] for x in edge_f1s) / len(edge_f1s)) if edge_f1s else None,
        })

    out = {"status": "PAIR_EVAL_DENSE_J_DEV", "model": str(args.model),
           "count_anchor": count_anchor, "n_contexts": len(records),
           "by_regime": {}}
    for regime in ("I", "C", "R"):
        items = [r for r in records if r["regime"] == regime]
        if items:
            out["by_regime"][regime] = {
                "contexts": len(items),
                "mean_joint_nll": sum(r["joint_nll"] for r in items) / len(items),
                "mean_count_kl": sum(r["count_kl"] for r in items) / len(items),
                "mean_edge_f1": (sum(r["edge_f1"] for r in items if r["edge_f1"] is not None)
                                 / max(1, sum(1 for r in items if r["edge_f1"] is not None)))
                if regime == "R" else None,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
