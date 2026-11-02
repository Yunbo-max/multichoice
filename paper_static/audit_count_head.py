#!/usr/bin/env python
"""Audit the learned count-head correction a_m(x) and its decode impact (GPU).

Loads a structured (set) checkpoint, forwards DEV rows, and answers two
questions with the *actual* count-head outputs rather than its parameter norms:

1. Is a_m(x) = count_head(hidden) really close to zero? It reports the magnitude
   of a_m and of log Z_m(s) = log sum_{|S|=m} exp(sum_{i in S} s_i), and their
   ratio, so a small weight/bias cannot hide a large hidden activation.

2. What changes if a_m is forced to zero? It decodes the same candidate logits
   through structured_prediction twice (a_m vs a_m=0) and reports the fraction
   of changed predicted sets plus question_em, count_nll, count_brier, count_mae.

Run (GPU required)::

    V/bin/python paper_static/audit_count_head.py \
        --model runs/paper_static/qwen3.5-4b-bce-s44/seed_43/set_ce_anchor/final \
        --input data/multiselect_grouped_multi/dev.jsonl \
        --limit 200 --output runs/audit/count_head/seed_43_set_ce_anchor
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


def _stats(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {"n": 0, "max": None, "min": None, "mean": None, "std": None}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "n": len(values),
        "max": max(values),
        "min": min(values),
        "mean": mean,
        "std": math.sqrt(var),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="rows to forward (0 = all)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    args = parser.parse_args(argv)

    import train_multiselect as trainer
    import set_distillation as sd
    import multiselect_metrics as metrics
    from eval_multiselect import (
        checkpoint_count_anchor,
        make_batches,
        validate_right_padding,
    )
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu")

    rows = metrics.read_dataset(args.input, "dev")
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    labels = trainer.option_labels(tokenizer)
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    prompts = [trainer.make_prompt(tokenizer, row, labels) for row in rows]
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
        for prompt in prompts
    ]
    batches = make_batches(
        [row["id"] for row in rows], lengths, args.batch_size, args.max_input_tokens, args.max_batch_tokens
    )

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else Qwen3_5ForCausalLM
    begin = time.monotonic()
    base = cls.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base.requires_grad_(False)
    base.config.use_cache = False
    wrapper = trainer.Readout(base, label_ids, full_rank=True, objective="set")
    trainer.load_count_head(wrapper, args.model)
    count_anchor = checkpoint_count_anchor(args.model)
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)
    load_seconds = time.monotonic() - begin
    print(f"count_anchor={count_anchor} rows={len(rows)} load={load_seconds:.1f}s", file=sys.stderr)

    # Raw count-head outputs a_m and the candidate logits s_i, keyed by row id
    # (make_batches sorts rows by length, so results must be re-aligned by id).
    scores_by_id = {}
    a_m_by_id = {}
    begin = time.monotonic()
    with torch.inference_mode():
        for batch in batches:
            encoded = tokenizer(
                [prompts[index] for index in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            validate_right_padding(encoded["attention_mask"].tolist(), [lengths[i] for i in batch])
            candidate_logits, count_logits = wrapper(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            for position, index in enumerate(batch):
                row = rows[index]
                k = len(row["options"])
                scores_by_id[row["id"]] = candidate_logits[position, :k].float().cpu().tolist()
                a_m_by_id[row["id"]] = count_logits[position, : k + 1].float().cpu().tolist()
    inference_seconds = time.monotonic() - begin
    print(f"forward {inference_seconds:.1f}s", file=sys.stderr)
    if len(scores_by_id) != len(rows):
        raise ValueError(f"forward covered {len(scores_by_id)} of {len(rows)} rows")

    # ---- Problem 1: magnitudes of a_m(x) vs log Z_m(s) -------------------
    abs_a = []
    abs_logz = []
    raw_a = []
    raw_logz = []
    ratio_elems = []
    for row in rows:
        scores = scores_by_id[row["id"]]
        a_m = a_m_by_id[row["id"]]
        center = max(scores)
        logz = sd.log_partitions([s - center for s in scores])
        for m in range(len(a_m)):
            z = logz[m] + m * center  # = log Z_m(s) (absolute, centered DP + restore)
            a = a_m[m]
            raw_a.append(a)
            raw_logz.append(z)
            abs_a.append(abs(a))
            abs_logz.append(abs(z))
            if abs(z) > 1e-9:
                ratio_elems.append(abs(a) / abs(z))
    magnitude = {
        "a_m_abs": _stats(abs_a),
        "a_m_raw": _stats(raw_a),
        "logZ_abs": _stats(abs_logz),
        "logZ_raw": _stats(raw_logz),
        "ratio_abs_a_over_abs_logZ": _stats(ratio_elems),
    }

    # ---- Problem 2: decode with a_m vs a_m=0 -----------------------------
    pred_with = []
    pred_zero = []
    for idx, row in enumerate(rows):
        scores = scores_by_id[row["id"]]
        a_m = a_m_by_id[row["id"]]
        k = len(row["options"])
        protocol = {
            "selection_mode": row.get("selection_mode", "multi"),
            "allow_empty": row.get("allow_empty", False),
            "options": row["options"],
            "n_options": k,
        }
        common = {
            "id": row["id"],
            "input_tokens": lengths[idx],
            "option_keys": [o["key"] for o in row["options"]],
            "input_digest": sd.input_digest(row),
            "checkpoint_id": str(args.model.resolve()),
            "provenance": {
                "checkpoint_identity_kind": "local_path",
                "model_weight_hash_verified": False,
                "model_path": str(args.model.resolve()),
                "decision_prompt_version": "sii_answer_slot_protocol_v1",
                "mode": "direct_no_thinking",
            },
        }
        item = sd.structured_prediction(scores, a_m, protocol, count_anchor=count_anchor, zero_count_head=False)
        item["candidate_logits"] = scores
        item.update(common)
        pred_with.append(item)
        item0 = sd.structured_prediction(scores, a_m, protocol, count_anchor=count_anchor, zero_count_head=True)
        item0["candidate_logits"] = scores
        item0.update(common)
        pred_zero.append(item0)

    changed = 0
    for pw, pz in zip(pred_with, pred_zero):
        if pw["predicted_indices"] != pz["predicted_indices"]:
            changed += 1

    report_with = metrics.compute_metrics(rows, pred_with, threshold=0.5, decoder="auto")
    report_zero = metrics.compute_metrics(rows, pred_zero, threshold=0.5, decoder="auto")
    keys = ("question_em", "count_nll", "count_brier", "count_mae")
    comparison = {
        "n_rows": len(rows),
        "changed_predicted_sets": changed,
        "changed_ratio": changed / len(rows) if rows else 0.0,
        "with_a_m": {k: report_with.get(k) for k in keys},
        "zero_a_m": {k: report_zero.get(k) for k in keys},
        "delta_zero_minus_with": {
            k: (report_zero.get(k) - report_with.get(k)
                if isinstance(report_zero.get(k), (int, float)) and isinstance(report_with.get(k), (int, float))
                else None)
            for k in keys
        },
    }

    result = {
        "model": str(args.model.resolve()),
        "input": str(args.input),
        "limit": args.limit or len(rows),
        "count_anchor": count_anchor,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "magnitude": magnitude,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
