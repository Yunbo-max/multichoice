#!/usr/bin/env python3
"""Offline Qwen feature-cache + recurrent set-head training/evaluation.

No LLM/API calls, CoT generation or checkpoint modification. Start with the
existing gold-BCE Qwen3.5-4B final export and cache *native answer-slot*
features; train only the small shared-step head on those frozen features.

Commands:
  cache:    extract Qwen answer slot / native option logits from a JSONL split
  train:    supervised exact-set training on paired cached splits
  evaluate: evaluate depth sweep with one saved head, no gold used in decoding
  smoke:    CPU-only autograd and variable-K sanity check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from recurrent_set_decision import (
    RecurrentSetConfig,
    RecurrentSetDecision,
    config_as_dict,
    loss_over_steps,
    set_map,
)
from set_distillation import allowed_counts

SCHEMA = "recurrent-set-feature-cache-v1"
CHECKPOINT_SCHEMA = "recurrent-set-head-v1"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def safe_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def check_cache(cache: dict) -> None:
    if cache.get("schema") != SCHEMA:
        raise ValueError("feature cache schema mismatch")
    n = len(cache["ids"])
    if n == 0 or len(set(cache["ids"])) != n:
        raise ValueError("empty cache or duplicate question IDs")
    context, logits, mask, gold = (
        cache["context"],
        cache["native_logits"],
        cache["mask"],
        cache["gold"],
    )
    if context.ndim != 2 or any(v.shape[0] != n for v in (logits, mask, gold)):
        raise ValueError("cache batch dimensions mismatch")
    if logits.shape != mask.shape or logits.shape != gold.shape:
        raise ValueError("candidate tensor shapes mismatch")
    if len(cache["allowed"]) != n or not mask.bool().any(dim=1).all():
        raise ValueError("invalid allowed cardinalities or masks")
    if (
        not torch.isfinite(context).all()
        or not torch.isfinite(logits[mask.bool()]).all()
    ):
        raise ValueError("nonfinite features")
    if not bool((mask.long().cummin(dim=1).values == mask.long()).all()):
        raise ValueError("candidate validity must be a prefix")
    for i, support in enumerate(cache["allowed"]):
        k = int(mask[i].sum().item())
        if (
            len(support) != len(set(support))
            or not support
            or min(support) < 0
            or max(support) > k
        ):
            raise ValueError("invalid cardinality support")
        if int(gold[i, :k].sum().item()) not in support:
            raise ValueError("gold answer violates permitted count")
        if bool(gold[i, k:].any()):
            raise ValueError("padded gold labels must be 0")


@torch.no_grad()
def cache_command(args) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    import train_multiselect as base

    source = args.dataset.resolve()
    rows = base.read_jsonl(source)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("empty dataset")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = base.option_labels(tokenizer)
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
    cls = (
        Qwen3_5ForConditionalGeneration
        if config.model_type == "qwen3_5"
        else Qwen3_5ForCausalLM
    )
    model, loading = cls.from_pretrained(
        str(args.model),
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError("incomplete Qwen export: missing/mismatched tensors")
    model = model.to(device).eval()
    model.requires_grad_(False)
    backbone = getattr(model.model, "language_model", model.model)
    output_weight = (
        model.get_output_embeddings()
        .weight.index_select(
            0, torch.as_tensor(label_ids, device=device, dtype=torch.long)
        )
        .float()
    )

    max_k = max(len(row["options"]) for row in rows)
    if max_k > args.max_options:
        raise ValueError("dataset includes more options than --max-options")
    features, score_rows, mask_rows, gold_rows, support_rows, identifiers = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        prompts = []
        for row in chunk:
            if len(row["target"]) != len(row["options"]):
                raise ValueError("dataset gold labels do not match candidates")
            if row.get("selection_mode") != "multi":
                raise ValueError("this first pilot supports multi-select rows only")
            support = allowed_counts(row)
            if int(sum(row["target"])) not in support:
                raise ValueError("gold violates declared selection protocol")
            prompts.append(base.make_prompt(tokenizer, row, labels))
        tokenized = tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        if tokenized["input_ids"].shape[1] > args.max_input_tokens:
            raise ValueError(
                "prompt exceeds max token limit; refusing silent truncation"
            )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        out = backbone(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
        last_indices = tokenized["attention_mask"].sum(-1) - 1
        slot = out.last_hidden_state[
            torch.arange(len(chunk), device=device), last_indices
        ]
        logits = torch.nn.functional.linear(slot.float(), output_weight)
        features.append(slot.float().cpu())
        for j, row in enumerate(chunk):
            k = len(row["options"])
            score_rows.append(
                torch.nn.functional.pad(logits[j, :k].detach().cpu(), (0, max_k - k))
            )
            mask_rows.append(torch.tensor([True] * k + [False] * (max_k - k)))
            gold_rows.append(
                torch.tensor(row["target"] + [0] * (max_k - k), dtype=torch.float32)
            )
            support_rows.append(allowed_counts(row))
            identifiers.append(row["id"])
        print(f"cached {min(len(rows), start + len(chunk))}/{len(rows)}", flush=True)

    cache = {
        "schema": SCHEMA,
        "ids": identifiers,
        "context": torch.cat(features, dim=0),
        "native_logits": torch.stack(score_rows),
        "mask": torch.stack(mask_rows),
        "gold": torch.stack(gold_rows),
        "allowed": support_rows,
        "data_sha256": digest(source),
        "dataset_path": str(source),
        "model_identity": args.model_identity or str(args.model.resolve()),
        "model_weights_hash_verified": False,
        "model_path": str(args.model.resolve()),
    }
    check_cache(cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("feature cache already exists; pass --overwrite")
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.save(cache, tmp)
    os.replace(tmp, args.output)
    write_json(
        args.output.with_suffix(".meta.json"),
        {
            "schema": SCHEMA,
            "examples": len(rows),
            "context_dim": cache["context"].shape[-1],
            "max_options": max_k,
            "data_sha256": cache["data_sha256"],
            "model_identity": cache["model_identity"],
            "model_weights_hash_verified": False,
        },
    )


def read_caches(train_path: Path, dev_path: Path) -> tuple[dict, dict]:
    train, dev = safe_load(train_path), safe_load(dev_path)
    check_cache(train)
    check_cache(dev)
    if train["context"].shape[1] != dev["context"].shape[1]:
        raise ValueError("train/dev backbone hidden widths differ")
    if train["model_identity"] != dev["model_identity"]:
        raise ValueError("train/dev were extracted from different model checkpoints")
    overlap = set(train["ids"]) & set(dev["ids"])
    if overlap:
        raise ValueError(f"train/dev question ID overlap: {list(overlap)[:3]}")
    return train, dev


def take(cache: dict, indices, device: torch.device, k: int | None = None):
    if k is None:
        k = cache["native_logits"].shape[1]
    idx = torch.as_tensor(indices, dtype=torch.long)
    return (
        cache["context"][idx].to(device=device, dtype=torch.float32),
        cache["native_logits"][idx, :k].to(device=device, dtype=torch.float32),
        cache["mask"][idx, :k].to(device=device),
        cache["gold"][idx, :k].to(device=device),
        [cache["allowed"][int(i)] for i in indices],
    )


@torch.no_grad()
def evaluate_model(
    model: RecurrentSetDecision,
    cache: dict,
    device: torch.device,
    steps: int,
    batch_size: int = 32,
) -> dict:
    model.eval()
    correct, total, true_positive, false_positive, false_negative = 0, 0, 0, 0, 0
    count_abs, count_correct, nll_total = 0, 0, 0.0
    for start in range(0, len(cache["ids"]), batch_size):
        inds = list(range(start, min(len(cache["ids"]), start + batch_size)))
        context, native, mask, gold, allowed = take(cache, inds, device)
        history = model(context, native, mask, steps=steps)
        final = history[-1]
        for j, support in enumerate(allowed):
            k = int(mask[j].sum().item())
            decoded = set_map(
                final["scores"][j, :k], final["correction"][j, : k + 1], support
            )
            pred = torch.zeros(k, device=device, dtype=torch.bool)
            pred[decoded["indices"]] = True
            ref = gold[j, :k].bool()
            correct += int(torch.equal(pred, ref))
            true_positive += int((pred & ref).sum())
            false_positive += int((pred & ~ref).sum())
            false_negative += int((~pred & ref).sum())
            count_abs += abs(int(pred.sum()) - int(ref.sum()))
            count_correct += int(int(pred.sum()) == int(ref.sum()))
            # Unnormalized exact negative log probability for comparison across depths.
            nll_total += float(
                -(
                    final["scores"][j, :k][ref].sum()
                    + final["correction"][j, int(ref.sum())]
                    - log_partition_for_eval(
                        final["scores"][j, :k], final["correction"][j, : k + 1], support
                    )
                )
            )
            total += 1
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "n": total,
        "question_em": correct / total,
        "micro_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "precision": precision,
        "recall": recall,
        "count_accuracy": count_correct / total,
        "count_mae": count_abs / total,
        "set_nll": nll_total / total,
        "steps": steps,
    }


def log_partition_for_eval(scores, corrections, support):
    from recurrent_set_decision import set_log_partition

    return set_log_partition(scores, corrections, support)


@torch.no_grad()
def initial_native_map(native, mask, allowed):
    """Per-question native 0-step MAP sets; uses no gold labels."""
    if len(allowed) != native.shape[0]:
        raise ValueError("allowed count / batch mismatch")
    prior = torch.zeros_like(native)
    for i, counts in enumerate(allowed):
        k = int(mask[i].sum().item())
        initial = set_map(native[i, :k], native.new_zeros(k + 1), counts)
        prior[i, initial["indices"]] = 1
    return prior


def load_model_checkpoint(path: Path, device: torch.device):
    saved = safe_load(path)
    if saved.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unknown recurrent checkpoint schema")
    model = RecurrentSetDecision(RecurrentSetConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, saved


def train_command(args) -> None:
    if args.steps < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError(
            "head training requires positive steps, epochs, and batch size"
        )
    if not math.isfinite(args.lr) or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")
    if not math.isfinite(args.revision_weight) or args.revision_weight < 0:
        raise ValueError("revision_weight must be finite and nonnegative")
    if not math.isfinite(args.unchanged_weight) or not 0 <= args.unchanged_weight <= 1:
        raise ValueError("unchanged_weight must be finite in [0,1]")
    train_cache, dev_cache = read_caches(args.train_cache, args.dev_cache)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    max_options = max(
        train_cache["native_logits"].shape[1], dev_cache["native_logits"].shape[1]
    )
    cfg = RecurrentSetConfig(
        context_dim=train_cache["context"].shape[-1],
        latent_dim=args.latent_dim,
        max_options=max_options,
        feedback=args.feedback,
    )
    model = RecurrentSetDecision(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "config.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "config": config_as_dict(model),
            "seed": args.seed,
            "train_cache": str(args.train_cache),
            "dev_cache": str(args.dev_cache),
            "model_identity": train_cache["model_identity"],
            "train_data_sha256": train_cache["data_sha256"],
            "dev_data_sha256": dev_cache["data_sha256"],
            "steps": args.steps,
            "step_supervision": args.step_supervision,
            "monotonic_weight": args.monotonic_weight,
            "revision_weight": args.revision_weight,
            "unchanged_weight": args.unchanged_weight,
            "revision_source": "frozen initial native set-MAP (no gold labels)",
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "note": "Only recurrent head trained; original Qwen weights frozen in feature cache.",
        },
    )
    best = -1.0
    last_metrics = None
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_cache["ids"])))
        random.shuffle(order)
        model.train()
        epoch_loss, count = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            inds = order[start : start + args.batch_size]
            context, native, mask, gold, allowed = take(train_cache, inds, device)
            optimizer.zero_grad(set_to_none=True)
            history = model(context, native, mask, steps=args.steps)
            previous = (
                initial_native_map(native, mask, allowed)
                if args.revision_weight else None
            )
            loss = loss_over_steps(
                history,
                gold,
                mask,
                allowed,
                step_supervision=args.step_supervision,
                monotonic_weight=args.monotonic_weight,
                revision_previous=previous,
                revision_weight=args.revision_weight,
                unchanged_weight=args.unchanged_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_loss += float(loss) * len(inds)
            count += len(inds)
        last_metrics = evaluate_model(
            model, dev_cache, device, args.steps, args.batch_size
        )
        print(
            json.dumps(
                {"epoch": epoch, "train_loss": epoch_loss / count, **last_metrics}
            )
        )
        if last_metrics["question_em"] > best:
            best = last_metrics["question_em"]
            ckpt = {
                "schema": CHECKPOINT_SCHEMA,
                "config": config_as_dict(model),
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "model_identity": train_cache["model_identity"],
                "train_data_sha256": train_cache["data_sha256"],
                "dev_data_sha256": dev_cache["data_sha256"],
                "trained_steps": args.steps,
                "best_epoch": epoch,
                "selected_dev_metrics": last_metrics,
            }
            torch.save(ckpt, args.output / "best_head.pt")
    if last_metrics is None:
        raise RuntimeError("no training epoch completed")
    saved_model, ckpt = load_model_checkpoint(args.output / "best_head.pt", device)
    metrics = {
        str(n): evaluate_model(saved_model, dev_cache, device, n, args.batch_size)
        for n in sorted(set([0, 1, 2, 4, 8, args.steps]))
    }
    write_json(
        args.output / "depth_sweep.json",
        {
            "trained_steps": args.steps,
            "selected_epoch": ckpt["best_epoch"],
            "selected_question_em": best,
            "depths": metrics,
        },
    )


def evaluate_command(args) -> None:
    cache = safe_load(args.cache)
    check_cache(cache)
    device = torch.device(args.device)
    model, saved = load_model_checkpoint(args.checkpoint, device)
    if saved["model_identity"] != cache["model_identity"]:
        raise ValueError("cached features from another Qwen checkpoint")
    if model.config.context_dim != cache["context"].shape[1]:
        raise ValueError("feature dimension mismatch")
    if model.config.max_options < cache["native_logits"].shape[1]:
        raise ValueError("checkpoint cannot represent the candidate count")
    depths = [int(v) for v in args.depths.split(",")]
    report = {
        str(n): evaluate_model(model, cache, device, n, args.batch_size) for n in depths
    }
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)


def smoke_command(_) -> None:
    torch.manual_seed(12)
    model = RecurrentSetDecision(
        RecurrentSetConfig(context_dim=24, latent_dim=32, max_options=5)
    )
    context = torch.randn(3, 24)
    logits = torch.randn(3, 5)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool
    )
    gold = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    allowed = [list(range(1, 4)), list(range(1, 6)), [1, 2]]
    hist = model(context, logits, mask, steps=4, demonstrations=torch.randn(3, 2, 24))
    loss = loss_over_steps(hist, gold, mask, allowed, step_supervision="last")
    loss.backward()
    grads = {
        name: float(param.grad.abs().sum()) if param.grad is not None else 0.0
        for name, param in model.named_parameters()
    }
    assert (
        math.isfinite(float(loss))
        and grads["score_delta.weight"] > 0
        and grads["count_delta.weight"] > 0
    )
    print(
        json.dumps(
            {
                "smoke": "PASS",
                "loss": float(loss),
                "steps": len(hist) - 1,
                "grad_score": grads["score_delta.weight"],
                "grad_count": grads["count_delta.weight"],
            }
        )
    )


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("cache")
    extract.add_argument("--dataset", type=Path, required=True)
    extract.add_argument("--model", type=Path, required=True)
    extract.add_argument(
        "--model-identity", help="explicit immutable checkpoint identity"
    )
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument("--max-options", type=int, default=32)
    extract.add_argument("--max-input-tokens", type=int, default=8192)
    extract.add_argument("--batch-size", type=int, default=2)
    extract.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    extract.add_argument("--device", default="cuda")
    train = sub.add_parser("train")
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--dev-cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--latent-dim", type=int, default=128)
    train.add_argument(
        "--feedback", choices=("set", "latent_only", "static", "mlp"), default="set"
    )
    train.add_argument("--steps", type=int, default=4)
    train.add_argument("--step-supervision", choices=("all", "last"), default="all")
    train.add_argument("--monotonic-weight", type=float, default=0.0)
    train.add_argument("--revision-weight", type=float, default=0.0,
                       help="weight of added/removed-option BCE relative to frozen native set MAP")
    train.add_argument("--unchanged-weight", type=float, default=0.0,
                       help="relative auxiliary BCE weight on unchanged options, 0..1")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=43)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--depths", default="0,1,2,4,8")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output", type=Path)
    sub.add_parser("smoke")
    return top


if __name__ == "__main__":
    args = parser().parse_args()
    {
        "cache": cache_command,
        "train": train_command,
        "evaluate": evaluate_command,
        "smoke": smoke_command,
    }[args.command](args)
