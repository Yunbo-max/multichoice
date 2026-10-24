#!/usr/bin/env python3
"""SetJev v2 count-prior diagnostic: candidate logit scale and anchored count prior.

For one checkpoint, compute the native answer-slot logits on a dev sample and
report the anchored count prior ``softmax(log Z_m(raw))`` at zero residual:

  - logit scale: mean / max / std of candidate logits
  - P(|S|=K): mass on "select all" (the degenerate cold-start failure mode)
  - count CE against gold: -log rho[gold_count]
  - predicted-count histogram vs gold-count histogram

A low P(|S|=K) and a small count CE mean the candidate logits already imply a
reasonable count distribution (warm start). P(|S|=K)->1 means the cold-start
anchor will oscillate (docs/setjev_v2_gate3_smoke.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HARNESS = str(Path(__file__).resolve().parent)
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)


def _local(name):
    import importlib.util

    module_name = "_count_prior_" + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _count_prior(scores, row):
    sd = _local("set_distillation")
    k = len(scores)
    anchored = sd.count_anchor_logits(scores, [0.0] * (k + 1))  # log Z_m(raw)
    support = sd.allowed_counts(row)
    log_total = _logsumexp([anchored[m] for m in support])
    rho = {
        m: math.exp(anchored[m] - log_total) if m in support else 0.0
        for m in range(k + 1)
    }
    gold = sum(row["target"])
    return {
        "p_select_all": rho[k],
        "count_ce": -math.log(rho[gold]),
        "predicted_count": max(support, key=lambda m: rho[m]),
        "gold_count": gold,
        "logit_mean": sum(scores) / k,
        "logit_max": max(scores),
        "logit_std": math.sqrt(sum((s - sum(scores) / k) ** 2 for s in scores) / k),
    }


def _predict_logits(args, rows):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

    trainer = _local("train_multiselect")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = trainer.option_labels(tokenizer)
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    prompts = [trainer.make_prompt(tokenizer, row, labels) for row in rows]

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cls = (
        Qwen3_5ForConditionalGeneration
        if config.model_type == "qwen3_5"
        else Qwen3_5ForCausalLM
    )
    device = torch.device(args.device)
    base, loading = cls.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("incomplete model export")
    base.requires_grad_(False)
    base.config.use_cache = False
    base = base.to(device).eval()

    import readout_audit as ra

    return ra.native_predictions(
        base,
        tokenizer,
        labels,
        label_ids,
        rows,
        device,
        args.batch_size,
        args.max_input_tokens,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--limit-mode", choices=("head", "stride"), default="head")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        metrics = _local("multiselect_metrics")
        rows = metrics.read_dataset(args.input, "any")
        if args.limit_mode == "head" and 0 < args.limit < len(rows):
            rows = rows[: args.limit]
        elif args.limit_mode == "stride" and 0 < args.limit < len(rows):
            rows = [rows[(i * len(rows)) // args.limit] for i in range(args.limit)]
        predictions = _predict_logits(args, rows)
        stats = []
        for row, pred in zip(rows, predictions):
            k = len(row["options"])
            stats.append(_count_prior(pred["logits"][:k], row))
        n = len(stats)
        report = {
            "model": str(args.model),
            "n_questions": n,
            "logit_mean": sum(s["logit_mean"] for s in stats) / n,
            "logit_max_mean": sum(s["logit_max"] for s in stats) / n,
            "logit_std_mean": sum(s["logit_std"] for s in stats) / n,
            "p_select_all_mean": sum(s["p_select_all"] for s in stats) / n,
            "p_select_all_max": max(s["p_select_all"] for s in stats),
            "count_ce_mean": sum(s["count_ce"] for s in stats) / n,
            "predicted_count_histogram": {
                str(c): sum(1 for s in stats if s["predicted_count"] == c)
                for c in sorted({s["predicted_count"] for s in stats})
            },
            "gold_count_histogram": {
                str(c): sum(1 for s in stats if s["gold_count"] == c)
                for c in sorted({s["gold_count"] for s in stats})
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"count-prior diagnostic failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
