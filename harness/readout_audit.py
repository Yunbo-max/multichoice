#!/usr/bin/env python3
"""Gate 1 readout audit: native option-logit vs candidate Yes/No readout.

Implements section 3 of ``docs/setjev_v2_plan.md`` (Teacher--Readout Mismatch).
One fixed checkpoint produces two teacher probability vectors per question, both
WITHOUT reasoning:

  native  sigmoid of the answer-slot K label logits (the trained direct decision)
  yesno   softmax([Yes, No]) per candidate, empty reasoning (the readout interface)

Both vectors are scored against gold with identical metrics (membership BCE /
Brier, count MAE, exact-set EM, precision/recall/F1, oracle-count EM), so a
readout mismatch is isolated from any reasoning effect. The two no-reasoning
readouts always cover every question, so truncation/coverage only become
relevant for the reasoning modes (handled by ``sii_style_teacher.py``).

Modes
-----
Dependency-free plan (no tokenizer or weights)::

  python -I harness/readout_audit.py --input data/multiselect/dev.jsonl --validate-only

GPU audit (both readouts on one checkpoint)::

  python -I harness/readout_audit.py --input data/multiselect/dev.jsonl \
      --model runs/qz/grouped_bce_4gpu_s44/final --output runs/readout_audit \
      --limit 300 --limit-mode stratified --sample-seed 43

``--limit N --limit-mode stratified`` selects N questions across source,
gold count and candidate count, using a fixed seed and storing selected-ID
provenance. Gold is used only to construct the offline *audit* cohort, never
as an inference-time feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = str(Path(__file__).resolve().parent)
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

SCHEMA = "readout_audit_v1"
KEY_METRICS = (
    "question_em",
    "f1",
    "precision",
    "recall",
    "bce",
    "membership_brier",
    "count_mae",
    "count_accuracy",
)


def _local_module(name):
    import importlib.util

    module_name = "_readout_audit_" + name
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


def audit_report(rows, native_predictions, yesno_predictions, threshold=0.5):
    """Dependency-free side-by-side scoring of two readout probability vectors.

    ``native_predictions`` and ``yesno_predictions`` are lists of prediction
    objects (``id``, ``p`` and optionally ``logits``/``candidate_logits``) aligned
    one-to-one with ``rows``. Returns the two full metric reports plus a paired
    comparison of the headline metrics.
    """
    metrics = _local_module("multiselect_metrics")
    native = metrics.compute_metrics(
        rows, native_predictions, threshold, decoder="protocol"
    )
    yesno = metrics.compute_metrics(
        rows, yesno_predictions, threshold, decoder="protocol"
    )

    def oracle_em(report):
        return report["oracle_gold_count"]["question_em"]

    comparison = {
        name: {
            "native": native.get(name),
            "yesno": yesno.get(name),
            "delta_yesno_minus_native": (
                None
                if name not in native or name not in yesno
                else yesno[name] - native[name]
                if isinstance(native[name], (int, float))
                and isinstance(yesno[name], (int, float))
                else None
            ),
        }
        for name in KEY_METRICS
    }
    comparison["oracle_gold_count_em"] = {
        "native": oracle_em(native),
        "yesno": oracle_em(yesno),
        "delta_yesno_minus_native": oracle_em(yesno) - oracle_em(native),
    }
    return {
        "schema": SCHEMA,
        "threshold": threshold,
        "n_questions": native["n_questions"],
        "native": native,
        "yesno": yesno,
        "comparison": comparison,
    }


def _subset(rows, limit, limit_mode, seed=43):
    """Return exactly N rows where feasible; stratified sampling is auditable.

    The offline diagnostic can stratify by gold count. Gold never reaches the
    model or prediction rule; it determines only which dev questions are audited.
    """
    if limit_mode not in ("head", "stride", "stratified"):
        raise ValueError("limit_mode must be head, stride or stratified")
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    if limit_mode == "head":
        return rows[:limit]
    if limit_mode == "stride":
        return [rows[(i * len(rows)) // limit] for i in range(limit)]

    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row["source"], min(sum(row["target"]), 5),
               min(len(row["options"]), 8))
        grouped[key].append(i)
    if limit < len(grouped):
        raise ValueError(
            f"stratified sample {limit} smaller than {len(grouped)} nonempty strata"
        )
    rng = random.Random(seed)
    for key in sorted(grouped):
        rng.shuffle(grouped[key])
    quotas = {key: 1 for key in grouped}
    target = {key: limit * len(indices) / len(rows)
              for key, indices in grouped.items()}
    for _ in range(limit - len(grouped)):
        possible = [key for key in grouped if quotas[key] < len(grouped[key])]
        key = max(possible, key=lambda k: (
            target[k] - quotas[k], len(grouped[k]) - quotas[k], k
        ))
        quotas[key] += 1
    selected = {i for key in grouped for i in grouped[key][:quotas[key]]}
    if len(selected) != limit:
        raise AssertionError("stratified selection must contain exactly N rows")
    return [row for i, row in enumerate(rows) if i in selected]


def _forward_logits(model, tokenizer, prompts, output_ids, device, max_input_tokens):
    import torch
    import torch.nn.functional as F

    backbone = getattr(model.model, "language_model", model.model)
    head = model.get_output_embeddings()
    encoded = tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    lengths = encoded["attention_mask"].sum(-1)
    longest = int(lengths.max())
    if longest > max_input_tokens:
        raise ValueError(
            f"input length {longest} exceeds --max-input-tokens {max_input_tokens}"
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = backbone(
            **encoded, use_cache=False, return_dict=True
        ).last_hidden_state
        end = (lengths - 1).to(hidden.device)
        slot = hidden[torch.arange(len(prompts), device=hidden.device), end]
        slot = slot.to(head.weight.device)
        ids = torch.tensor(output_ids, dtype=torch.long, device=head.weight.device)
        weight = head.weight.index_select(0, ids).detach()
        bias = (
            head.bias.index_select(0, ids).detach()
            if getattr(head, "bias", None) is not None
            else None
        )
        logits = F.linear(slot, weight, bias).float()
    return logits.cpu()


def native_predictions(
    model, tokenizer, labels, label_ids, rows, device, batch_size, max_input_tokens
):
    import torch

    trainer = _local_module("train_multiselect")
    prompts = [trainer.make_prompt(tokenizer, row, labels) for row in rows]
    predictions = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        logits = _forward_logits(
            model,
            tokenizer,
            prompts[start : start + batch_size],
            label_ids,
            device,
            max_input_tokens,
        )
        for position, row in enumerate(chunk):
            k = len(row["options"])
            scores = logits[position, :k].float()
            predictions.append(
                {
                    "id": row["id"],
                    "p": torch.sigmoid(scores).tolist(),
                    "logits": scores.tolist(),
                    "option_keys": [option["key"] for option in row["options"]],
                }
            )
    return predictions


def yesno_predictions(
    model, tokenizer, labels, yes_id, no_id, rows, device, batch_size, max_input_tokens
):
    import torch

    style = _local_module("sii_style_teacher")
    jobs = []  # (row_index, candidate_index, prompt)
    for row_index, row in enumerate(rows):
        for candidate_index in range(len(row["options"])):
            prompt = style.make_membership_prompt(
                tokenizer, row, labels, "", candidate_index
            )
            jobs.append((row_index, candidate_index, prompt))
    logits = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        logits.append(
            _forward_logits(
                model,
                tokenizer,
                [job[2] for job in chunk],
                [yes_id, no_id],
                device,
                max_input_tokens,
            )
        )
    import torch as _torch

    logits = _torch.cat(logits, dim=0)
    probs = _torch.softmax(logits, dim=-1)[:, 0].tolist()
    per_row = {row["id"]: [0.0] * len(row["options"]) for row in rows}
    for (row_index, candidate_index, _), q in zip(jobs, probs):
        per_row[rows[row_index]["id"]][candidate_index] = q
    return [
        {
            "id": row["id"],
            "p": per_row[row["id"]],
            "option_keys": [option["key"] for option in row["options"]],
        }
        for row in rows
    ]


def _resolve_yesno(tokenizer):
    style = _local_module("sii_style_teacher")
    info = style.binary_token_info(tokenizer, yes_token="Yes", no_token="No")
    return info["yes_token_id"], info["no_token_id"]


def _run_model(args, rows):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

    trainer = _local_module("train_multiselect")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    labels = trainer.option_labels(tokenizer)
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    yes_id, no_id = _resolve_yesno(tokenizer)

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
        raise RuntimeError(
            "incomplete model export: "
            + str({k: loading.get(k) for k in ("missing_keys", "mismatched_keys")})
        )
    base.requires_grad_(False)
    base.config.use_cache = False
    base = base.to(device).eval()

    native = native_predictions(
        base,
        tokenizer,
        labels,
        label_ids,
        rows,
        device,
        args.batch_size,
        args.max_input_tokens,
    )
    yesno = yesno_predictions(
        base,
        tokenizer,
        labels,
        yes_id,
        no_id,
        rows,
        device,
        args.batch_size,
        args.max_input_tokens,
    )
    return audit_report(rows, native, yesno, args.threshold)


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="gold unified JSONL (e.g. dev)"
    )
    parser.add_argument(
        "--model", type=Path, help="local standalone exported HF checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, help="output directory for audit.json (model mode)"
    )
    parser.add_argument(
        "--validate-only", "--dry-run", dest="validate_only", action="store_true"
    )
    parser.add_argument(
        "--expected-split", choices=("test", "dev", "train", "any"), default="any"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument(
        "--limit-mode", choices=("head", "stride", "stratified"), default="head"
    )
    parser.add_argument("--sample-seed", type=int, default=43)
    args = parser.parse_args(argv)
    try:
        metrics = _local_module("multiselect_metrics")
        metrics.validate_threshold(args.threshold)
        if min(args.batch_size, args.max_input_tokens) < 1:
            raise ValueError("batch size and token budgets must be positive")
        if args.model is None and not args.validate_only:
            raise ValueError("provide --model or --validate-only")
        if args.validate_only and args.output is None:
            args.output = None
        if (
            args.output is not None
            and args.output.exists()
            and any(args.output.iterdir())
        ):
            raise ValueError("--output must be a new or empty directory")
        rows = metrics.read_dataset(
            args.input, None if args.expected_split == "any" else args.expected_split
        )
        full_rows = len(rows)
        rows = _subset(rows, args.limit, args.limit_mode, seed=args.sample_seed)
        if not rows:
            raise ValueError("no rows to audit after applying --limit")
        stratification = {}
        for row in rows:
            name = (f"{row['source']}|gold{min(sum(row['target']), 5)}"
                    f"|k{min(len(row['options']), 8)}")
            stratification[name] = stratification.get(name, 0) + 1
        selected_hash = hashlib.sha256(
            json.dumps(sorted(row["id"] for row in rows),
                       ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        config = {
            "input": str(args.input),
            "expected_split": args.expected_split,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "model": str(args.model) if args.model else None,
            "threshold": args.threshold,
            "limit": args.limit,
            "limit_mode": args.limit_mode,
            "sample_seed": args.sample_seed,
            "n_rows_before_limit": full_rows,
            "n_rows_after_limit": len(rows),
            "selected_ids_sha256": selected_hash,
            "audit_strata": dict(sorted(stratification.items())),
        }
        if args.validate_only:
            config.update(
                {"mode": "validate-only", "ok": True, "weights_loaded": False}
            )
            if args.model is not None:
                if not args.model.is_dir():
                    raise ValueError(
                        f"model/tokenizer directory does not exist: {args.model}"
                    )
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                _save_json(args.output / "audit_plan.json", config)
            print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        report = _run_model(args, rows)
        report.update(config)
        report["mode"] = "checkpoint-audit"
        args.output.mkdir(parents=True, exist_ok=True)
        _save_json(args.output / "audit.json", report)
        print(
            json.dumps(
                report["comparison"], ensure_ascii=False, indent=2, allow_nan=False
            )
        )
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"readout audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
