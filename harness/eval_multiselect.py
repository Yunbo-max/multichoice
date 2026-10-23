#!/usr/bin/env python
"""Evaluate an exported multi-select HF checkpoint at a frozen threshold.

The BCE path imports the trainer's exact make_prompt, option_labels and Readout,
then applies independent sigmoid to the first K label logits. No candidate
softmax and no threshold fitting occur. Explicit single-choice rows use argmax;
legacy rows and explicit multi rows use the supplied threshold (default 0.5).

Structured checkpoints are detected by decision_readout.json and require their
decision_count_head.pt sidecar. Their joint-set MAP decoder and coherent
marginals are used; a missing count head cannot silently fall back to BCE.

Dependency-free dataset validation (no tokenizer or weights)::

  python -I harness/eval_multiselect.py --input data/multiselect/test.jsonl \
      --validate-only

Score saved per-question probabilities without torch/transformers::

  python -I harness/eval_multiselect.py --input data/multiselect/test.jsonl \
      --score-predictions runs/eval/predictions.jsonl --threshold 0.5 \
      --output runs/rescored_test

Evaluate the standalone model selected on dev::

  python -I harness/eval_multiselect.py --input data/multiselect/test.jsonl \
      --model runs/ms_gold/final --threshold 0.5 --output runs/test_eval

``--validate-only --check-prompts --model PATH`` additionally checks the exact
tokenized prompts and batch plan without loading weights (model dependencies
and tokenizer files are needed). Output directories must be new or empty.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def _local_module(name):
    """Load an audited sibling by path, including under python -I."""
    module_name = "_multichoice_eval_" + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


metrics = _local_module("multiselect_metrics")


def checkpoint_objective(path):
    """Detect structured exports and fail closed on partial/malformed sidecars."""
    path = Path(path)
    metadata_path = path / "decision_readout.json"
    head_path = path / "decision_count_head.pt"
    if not metadata_path.exists():
        if head_path.exists():
            raise ValueError(f"{path}: decision_count_head.pt exists without decision_readout.json")
        return "bce"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "decision_set_readout_v1" or metadata.get("objective") != "set":
        raise ValueError(f"{metadata_path}: unsupported structured readout metadata")
    if metadata.get("max_candidates") != 255:
        raise ValueError(f"{metadata_path}: max_candidates must match the trainer's 255 labels")
    if not head_path.is_file():
        raise ValueError(f"{path}: structured checkpoint is missing decision_count_head.pt")
    return "set"


def check_export(path):
    """Require a local standalone HF export and every referenced weight shard."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"model must be a local exported HF directory: {path}")
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise ValueError(f"{path}: missing {name}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping:
            raise ValueError(f"{index}: empty weight_map")
        shards = sorted(set(mapping.values()))
        missing = [name for name in shards if not (path / name).is_file()]
        if missing:
            raise ValueError(f"{path}: missing weight shards: {missing}")
    else:
        # The trainer uses safe_serialization; adapter-only directories do not qualify.
        shards = [item.name for item in path.glob("model*.safetensors") if item.is_file()]
        if not shards:
            raise ValueError(f"{path}: no standalone model safetensors weights found")
    return {"objective": checkpoint_objective(path), "weight_shards": len(shards)}


def make_batches(ids, lengths, batch_size=8, max_input_tokens=8192, max_batch_tokens=8192):
    """Length-sorted batches bounded by both row count and padded token count."""
    if len(ids) != len(lengths):
        raise ValueError("IDs and lengths must have the same length")
    if min(batch_size, max_input_tokens, max_batch_tokens) < 1:
        raise ValueError("batch size and token budgets must be positive")
    for ident, length in zip(ids, lengths):
        if length < 1:
            raise ValueError(f"{ident}: prompt is empty")
        if length > max_input_tokens:
            raise ValueError(f"{ident}: prompt length {length} exceeds --max-input-tokens "
                             f"{max_input_tokens}; inputs are never truncated")
        if length > max_batch_tokens:
            raise ValueError(f"{ident}: prompt length {length} exceeds padded-token budget {max_batch_tokens}")
    batches, current, longest = [], [], 0
    for index in sorted(range(len(ids)), key=lambda i: (lengths[i], ids[i])):
        new_longest = max(longest, lengths[index])
        if current and (len(current) >= batch_size or (len(current) + 1) * new_longest > max_batch_tokens):
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, lengths[index])
    if current:
        batches.append(current)
    return batches


def validate_right_padding(masks, expected_lengths):
    """The trainer's last-slot gather requires contiguous right-padded masks."""
    if len(masks) != len(expected_lengths):
        raise ValueError("attention mask batch size does not match planned rows")
    for mask, length in zip(masks, expected_lengths):
        if any(value not in (0, 1) for value in mask) or sum(mask) != length:
            raise ValueError("encoded prompt length differs from its untruncated plan")
        if length < 1 or list(mask) != [1] * length + [0] * (len(mask) - length):
            raise ValueError("Readout requires contiguous right padding")


def _prepare_prompts(args, rows):
    # Deferred imports keep validation and score-existing modes dependency free.
    from transformers import AutoTokenizer
    trainer = _local_module("train_multiselect")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor eos token; cannot batch safely")
        tokenizer.pad_token = tokenizer.eos_token
    labels = trainer.option_labels(tokenizer)
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    if len(label_ids) != 255 or len(set(label_ids)) != 255:
        raise ValueError("the 255 single-token legal label IDs must be unique")
    prompts = [trainer.make_prompt(tokenizer, row, labels) for row in rows]
    lengths = [len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
               for prompt in prompts]
    batches = make_batches([row["id"] for row in rows], lengths, args.batch_size,
                           args.max_input_tokens, args.max_batch_tokens)
    return trainer, tokenizer, labels, label_ids, prompts, lengths, batches


def _prompt_report(labels, lengths, batches):
    return {"legal_label_count": len(labels), "padding_side": "right", "truncation": False,
            "prompt_lengths_checked": True, "input_tokens_max": max(lengths),
            "input_tokens_total": sum(lengths), "batches": len(batches),
            "padded_tokens_total": sum(len(batch) * max(lengths[i] for i in batch) for batch in batches)}


def infer(args, rows):
    import torch
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    export = check_export(args.model)
    objective = export["objective"]
    if objective == "set" and args.decoder == "protocol":
        raise ValueError("a structured checkpoint requires its set MAP decoder; use --decoder auto")
    if objective == "bce" and args.decoder == "provided":
        raise ValueError("--decoder provided is for saved predictions or structured checkpoints")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu or use --validate-only")
    trainer, tokenizer, labels, label_ids, prompts, lengths, batches = _prepare_prompts(args, rows)
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else Qwen3_5ForCausalLM
    begin = time.monotonic()
    base, loading = cls.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("incomplete model export: " + str({name: loading.get(name)
                           for name in ("missing_keys", "mismatched_keys")}))
    base.requires_grad_(False)
    base.config.use_cache = False
    if objective == "set":
        wrapper = trainer.Readout(base, label_ids, full_rank=True, objective="set")
        trainer.load_count_head(wrapper, args.model)
        structured = _local_module("set_distillation").structured_prediction
    else:
        wrapper = trainer.Readout(base, label_ids, full_rank=True)
        structured = None
    wrapper = wrapper.to(device).eval()
    wrapper.requires_grad_(False)
    load_seconds = time.monotonic() - begin
    predictions = []
    begin = time.monotonic()
    with torch.inference_mode():
        for batch in batches:
            encoded = tokenizer([prompts[index] for index in batch], padding=True,
                                truncation=False, return_tensors="pt", add_special_tokens=False)
            validate_right_padding(encoded["attention_mask"].tolist(), [lengths[i] for i in batch])
            outputs = wrapper(input_ids=encoded["input_ids"].to(device),
                              attention_mask=encoded["attention_mask"].to(device))
            if objective == "set":
                candidate_logits, count_logits = outputs
            else:
                candidate_logits, count_logits = outputs, None
            for position, index in enumerate(batch):
                row = rows[index]
                k = len(row["options"])
                scores = candidate_logits[position, :k].float()
                if objective == "set":
                    # Only public task protocol and K reach the decoder, never gold targets/count.
                    protocol = {"selection_mode": row.get("selection_mode", "multi"),
                                "allow_empty": row.get("allow_empty", False),
                                "options": row["options"], "n_options": k}
                    item = structured(scores.cpu().tolist(), count_logits[position].float().cpu().tolist(),
                                      protocol)
                    if item.get("decoder") != "set_map" or "predicted_indices" not in item:
                        raise ValueError("structured decoder did not return a set MAP prediction")
                    item["candidate_logits"] = scores.cpu().tolist()
                else:
                    item = {"p": torch.sigmoid(scores).cpu().tolist(), "logits": scores.cpu().tolist()}
                item.update({"id": row["id"], "input_tokens": lengths[index],
                             "option_keys": [option["key"] for option in row["options"]],
                             "input_digest": _local_module("set_distillation").input_digest(row),
                             "checkpoint_id": args.checkpoint_id or str(args.model.resolve()),
                             "provenance": {
                                 "checkpoint_identity_kind": "explicit" if args.checkpoint_id else "local_path",
                                 "model_weight_hash_verified": False,
                                 "model_path": str(args.model.resolve()),
                                 "decision_prompt_version": "sii_answer_slot_protocol_v1",
                                 "mode": "direct_no_thinking",
                             }})
                predictions.append(item)
    report = {**export, **_prompt_report(labels, lengths, batches),
              "load_seconds": load_seconds, "inference_seconds": time.monotonic() - begin,
              "device": args.device, "dtype": args.dtype,
              "threshold_applies_to_set_map": False}
    return predictions, report


def _new_output(path):
    if path is None:
        raise ValueError("--output is required for inference or scoring")
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("--output must be a new or empty directory")
    return path


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="gold unified test JSONL")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="local standalone exported HF checkpoint")
    source.add_argument("--score-predictions", "--predictions", dest="score_predictions", type=Path,
                        help="score saved JSONL probabilities; no model dependencies")
    parser.add_argument("--output", type=Path, help="new directory for predictions.jsonl and metrics.json")
    parser.add_argument("--checkpoint-id",
                        help="explicit immutable identity for native inference; supplied identity is not a weight hash check")
    parser.add_argument("--expected-split", choices=("test", "dev", "train", "any"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5, help="frozen beforehand; never fitted on input")
    parser.add_argument("--threshold-provenance", default="fixed beforehand (CLI value or default 0.5)")
    parser.add_argument("--decoder", choices=("auto", "protocol", "provided"), default="auto")
    parser.add_argument("--validate-only", "--dry-run", dest="validate_only", action="store_true")
    parser.add_argument("--check-prompts", action="store_true", help="validate-only: load tokenizer, never weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        if args.checkpoint_id is not None and (not args.model or not args.checkpoint_id.strip()):
            raise ValueError("--checkpoint-id requires --model and a nonempty identity; saved predictions retain their provenance")
        metrics.validate_threshold(args.threshold)
        if min(args.batch_size, args.max_input_tokens, args.max_batch_tokens) < 1:
            raise ValueError("batch size and token budgets must be positive")
        if args.check_prompts and (not args.validate_only or args.model is None):
            raise ValueError("--check-prompts requires --validate-only and --model")
        rows = metrics.read_dataset(args.input, None if args.expected_split == "any" else args.expected_split)
        config = {"input": str(args.input), "expected_split": args.expected_split,
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "model": str(args.model) if args.model else None,
                  "checkpoint_id": args.checkpoint_id or (str(args.model.resolve()) if args.model else None),
                  "checkpoint_identity_kind": "explicit" if args.checkpoint_id else "local_path" if args.model else None,
                  "model_weight_hash_verified": False,
                  "score_predictions": str(args.score_predictions) if args.score_predictions else None,
                  "threshold": args.threshold, "threshold_provenance": args.threshold_provenance,
                  "threshold_fitted_on_input": False, "decoder": args.decoder,
                  "dataset": metrics.dataset_stats(rows), "prompt_lengths_checked": False,
                  "max_input_tokens": args.max_input_tokens, "max_batch_tokens": args.max_batch_tokens,
                  "batch_size": args.batch_size,
                  "oracle_policy": "gold count used only by separately labeled post-hoc ranking diagnostic"}
        if args.validate_only:
            config.update({"mode": "validate-only", "ok": True, "weights_loaded": False})
            if args.model:
                if not args.model.is_dir():
                    raise ValueError(f"model/tokenizer directory does not exist: {args.model}")
                config["objective"] = checkpoint_objective(args.model)
            if args.check_prompts:
                _, _, labels, _, _, lengths, batches = _prepare_prompts(args, rows)
                config.update(_prompt_report(labels, lengths, batches))
            if args.score_predictions:
                metrics.prediction_records(rows, metrics.read_jsonl(args.score_predictions), args.threshold,
                                           decoder=args.decoder)
                config["predictions_validated"] = True
            if args.output:
                output = _new_output(args.output)
                output.mkdir(parents=True, exist_ok=True)
                _save_json(output / "evaluation_plan.json", config)
            print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        output = _new_output(args.output)
        if args.score_predictions:
            predictions = metrics.read_jsonl(args.score_predictions)
            config["mode"] = "score-existing-predictions"
        elif args.model:
            predictions, runtime = infer(args, rows)
            config.update(runtime)
            config["mode"] = "checkpoint-evaluation"
        else:
            raise ValueError("provide --model, --score-predictions, or --validate-only")
        records = metrics.prediction_records(rows, predictions, args.threshold, decoder=args.decoder)
        report = metrics.compute_metrics(rows, predictions, args.threshold, decoder=args.decoder)
        report.update({"expected_split": args.expected_split, "threshold_provenance": args.threshold_provenance,
                       "mode": config["mode"]})
        output.mkdir(parents=True, exist_ok=True)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        _save_json(output / "metrics.json", report)
        _save_json(output / "evaluation_config.json", config)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
