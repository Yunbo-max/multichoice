#!/usr/bin/env python
"""Unified multi-select decision model training (SII readout recipe).

The model reads the SII answer slot: the chat-templated State/Question/Options
prompt followed by ``The correct answer is (``.  At the last prompt position it
projects the output-head logits of the 255 legal single-token labels
(A..Z, AA..) and keeps the first K = n_options candidates.  Candidates are
independent Bernoulli variables scored with sigmoid in the default BCE baseline.
``--objective set`` additionally learns an answer-count head and uses a coherent
cardinality-conditioned distribution over subsets; its probabilities are not
the raw sigmoid scores. See docs/multiteacher_set_distillation.md.

Objective (per question, K candidates, logits s, target t in [0, 1]^K)::

    L = (1/K) sum_i [ softplus(s_i) - t_i s_i ]
        + (lambda_g / K) (sigmoid(s) - t)^T G (sigmoid(s) - t)

``--lambda-g 0 --graph none`` is pure BCEWithLogits.  ``--graph identity`` uses
G = I_K.  ``--graph cosine`` builds, for each K, the normalised graph Laplacian
of the non-negative cosine-similarity graph over the K label-token output
embeddings (a positive-semidefinite smoothness penalty).

Training is full-rank FSDP over the 176 language linear modules in
``innerjev/configs/4B/modules.json`` with the official SII hyperparameters.
Embeddings and the output head stay frozen.  The final export is a standalone
BF16 Hugging Face model directory (safe_serialization, <=5 GB shards) that
``harness/eval_multiselect.py`` can load. Structured checkpoints also require
their exported count-head sidecars; the old single-choice evaluator is unsuitable.

Examples
--------
CPU (no weights loaded)::

    .venv-sii/bin/python -I harness/train_multiselect.py --self-test
    .venv-sii/bin/python -I harness/train_multiselect.py --validate-only \\
        --train data/multiselect/train.jsonl --dev data/multiselect/dev.jsonl \\
        --model models/Qwen3.5-4B --modules innerjev/configs/4B/modules.json

8xH100 (QZ job / torchrun)::

    .venv-sii/bin/torchrun --standalone --nnodes 1 --nproc_per_node 8 \\
        harness/train_multiselect.py \\
        --train data/multiselect/train.jsonl --dev data/multiselect/dev.jsonl \\
        --model models/Qwen3.5-4B --modules innerjev/configs/4B/modules.json \\
        --target gold --output runs/qz/ms_gold/v1 \\
        --lr 2e-6 --global-batch 64 --epochs 1 \\
        --max-input-tokens 8192 --max-batch-tokens 8192 \\
        --eval-every-frac 0.1 --lambda-g 0 --graph none
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import string
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

# Also support the repository's documented ``python -I harness/...`` invocations.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from set_distillation import (
    READOUT_SCHEMA, TARGET_SCHEMA, allowed_counts, attach_set_targets, input_digest,
    read_set_target_file, set_loss_terms, structured_prediction,
)

ANSWER_CUE = "The correct answer is ("
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-4B"
DEFAULT_MODULES = ROOT / "innerjev" / "configs" / "4B" / "modules.json"
EXPECTED_TRAINABLE = 3313500160  # official SII 4B full-rank count


# ---------------------------------------------------------------------------
# I/O and prompt rendering (byte-for-byte equivalent to the SII harness)
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_jsonl(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def text_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def option_labels(tokenizer):
    candidates = list(string.ascii_uppercase)
    candidates += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    labels = [label for label in candidates
              if len(tokenizer.encode(label, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise ValueError(f"tokenizer provides only {len(labels)} single-token labels")
    return labels


def make_body(row, labels):
    choices = "\n".join(
        f"({labels[i]}) {o['key']}: {o['desc']}" for i, o in enumerate(row["options"]))
    body = (f'State:\n{text_value(row["state"])}\nQuestion: {text_value(row["question"])}\n'
            f"Options:\n{choices}").rstrip()
    # Legacy rows retain their original prompt; new datasets declare the task.
    if "selection_mode" in row:
        rule = ("Select exactly one correct option." if row["selection_mode"] == "single"
                else "Select all correct options; " +
                ("an empty answer is allowed." if row.get("allow_empty", False)
                 else "at least one option is correct."))
        body += "\nDecision rule: " + rule
    return body


def make_prompt(tokenizer, row, labels):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": make_body(row, labels)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prefix + ANSWER_CUE


# ---------------------------------------------------------------------------
# data validation
# ---------------------------------------------------------------------------
def validate_row(row, expected_split, label):
    errors = []
    ident = row.get("id")
    if not isinstance(ident, str) or not ident:
        errors.append("id must be a nonempty string")
        ident = "<missing>"
    prefix = f"{label}:{ident}"
    if expected_split and row.get("split") not in (None, expected_split):
        errors.append(f"{prefix}: split {row.get('split')!r} != {expected_split!r}")
    for key in ("source", "lang"):
        if not isinstance(row.get(key), str) or not row.get(key):
            errors.append(f"{prefix}: {key} must be a nonempty string")
    if "state" not in row:
        errors.append(f"{prefix}: state is required")
    if not isinstance(row.get("question"), str):
        errors.append(f"{prefix}: question must be a string")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        errors.append(f"{prefix}: options must contain 2..255 entries")
        return errors, 0
    keys = [o.get("key") if isinstance(o, dict) else None for o in options]
    if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
        errors.append(f"{prefix}: option keys must be unique nonempty strings")
    if any(not isinstance(o, dict) or not isinstance(o.get("desc"), str) for o in options):
        errors.append(f"{prefix}: option descriptions must be strings")
    target = row.get("target")
    if not isinstance(target, list) or len(target) != len(options):
        errors.append(f"{prefix}: target must be a list of length {len(options)}")
        return errors, len(options)
    if any(not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1) for v in target):
        errors.append(f"{prefix}: gold target must be a multi-hot int list")
    else:
        try:
            if sum(target) not in allowed_counts(row):
                errors.append(f"{prefix}: gold count violates the declared selection protocol")
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
    if row.get("n_options") != len(options):
        errors.append(f"{prefix}: n_options {row.get('n_options')} != {len(options)}")
    gold_letters = row.get("gold_letters")
    if isinstance(gold_letters, str) and gold_letters:
        expected = "".join(o["key"] for o, value in zip(options, target) if value)
        if sorted(expected) != sorted(gold_letters):
            errors.append(f"{prefix}: gold_letters {gold_letters!r} != target {expected!r}")
    return errors, len(options)


def read_dataset(path: Path, expected_split: str):
    rows = read_jsonl(path)
    errors, seen = [], set()
    option_counts = collections.Counter()
    for row in rows:
        ident = row.get("id")
        if isinstance(ident, str):
            if ident in seen:
                errors.append(f"{path}: duplicate id {ident!r}")
            seen.add(ident)
        row_errors, k = validate_row(row, expected_split, path.name)
        errors.extend(row_errors)
        if k:
            option_counts[k] += 1
    if not rows:
        errors.append(f"{path}: dataset is empty")
    if errors:
        raise ValueError("dataset validation failed:\n  " + "\n  ".join(errors[:40])
                         + (f"\n  ... {len(errors) - 40} more" if len(errors) > 40 else ""))
    return rows


def _sorted_counts(values):
    return {str(key): value for key, value in
            sorted(collections.Counter(values).items(), key=lambda item: str(item[0]))}


def dataset_stats(rows):
    return {
        "n": len(rows),
        "by_source": _sorted_counts(row.get("source") for row in rows),
        "by_lang": _sorted_counts(row.get("lang") for row in rows),
        "option_counts": {str(key): value for key, value in sorted(
            collections.Counter(len(row["options"]) for row in rows).items())},
        "positive_counts": {str(key): value for key, value in sorted(
            collections.Counter(int(sum(row["target"])) for row in rows).items())},
        "target_sum_min": min(sum(row["target"]) for row in rows),
        "target_sum_max": max(sum(row["target"]) for row in rows),
    }


# ---------------------------------------------------------------------------
# soft targets
# ---------------------------------------------------------------------------
def read_target_file(path: Path, allow_legacy=False):
    """Return (mapping id -> list[float], errors)."""
    mapping, errors, seen = {}, [], set()
    if not Path(path).exists():
        return mapping, [f"target file does not exist: {path}"]
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{path}:{line_number}: {error}")
                continue
            ident = row.get("id")
            values = row.get("q")
            if not allow_legacy and row.get("q_semantics") != "candidate_membership":
                errors.append(f"{path}:{line_number}: not a candidate-membership target; "
                              "diagnostic/legacy logits require --allow-legacy-targets")
                continue
            if not isinstance(ident, str) or not ident:
                errors.append(f"{path}:{line_number}: id must be a nonempty string")
                continue
            if ident in seen:
                errors.append(f"{path}:{line_number}: duplicate id {ident!r}")
                continue
            seen.add(ident)
            if not isinstance(values, list) or not values:
                errors.append(f"{path}:{line_number}: {ident}: q must be a nonempty list")
                continue
            clean = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"{path}:{line_number}: {ident}: q values must be numbers")
                    clean = None
                    break
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    errors.append(f"{path}:{line_number}: {ident}: q value {value} not in [0,1]")
                    clean = None
                    break
                clean.append(value)
            if clean is not None:
                mapping[ident] = clean
    return mapping, errors


def attach_train_targets(rows, mapping, mode):
    """Attach row['train_target']; return (errors, stats)."""
    errors, missing = [], []
    if mapping is None:
        for row in rows:
            row["train_target"] = [float(value) for value in row["target"]]
        return errors, {"source": "gold", "from_file": 0, "missing": 0, "extra_ids": 0}
    for row in rows:
        values = mapping.get(row["id"])
        k = len(row["options"])
        if values is None:
            missing.append(row["id"])
            if mode == "zero":
                row["train_target"] = [0.0] * k
            elif mode == "gold":
                row["train_target"] = [float(value) for value in row["target"]]
            continue
        if len(values) != k:
            errors.append(f"{row['id']}: soft target length {len(values)} != K={k}")
            continue
        row["train_target"] = list(values)
    if missing and mode == "error":
        errors.append(f"{len(missing)} row(s) missing from the target file, e.g. {missing[:5]}")
    stats = {
        "source": "file",
        "from_file": len(rows) - len(missing),
        "missing": len(missing),
        "missing_mode": mode,
        "extra_ids": len(set(mapping) - {row["id"] for row in rows}),
    }
    return errors, stats


def prepare_training_targets(rows, args):
    """One checked target path shared by validation and actual training."""
    mode = args.target or ("file" if args.target_file else "gold")
    if mode == "gold":
        return attach_set_targets(rows)
    if not args.target_file:
        return ["--target file requires --target-file"], {}
    records = read_jsonl(args.target_file)
    if records and records[0].get("schema") == TARGET_SCHEMA:
        mapping, errors = read_set_target_file(args.target_file)
        if errors:
            return errors, {}
        return attach_set_targets(rows, mapping, args.gold_weight, args.target_missing)
    if args.objective != "bce":
        return ["--objective set requires coherent q/count targets built with "
                "build_multiteacher_targets.py"], {}
    mapping, errors = read_target_file(args.target_file, args.allow_legacy_targets)
    # Verify content and order before discarding metadata in this legacy BCE path.
    by_id = {row["id"]: row for row in rows}
    for record in records:
        row = by_id.get(record.get("id"))
        if row is None or args.allow_legacy_targets:
            continue
        if record.get("option_keys") != [o["key"] for o in row["options"]]:
            errors.append(f"{row['id']}: teacher option identity/order mismatch")
        if record.get("input_digest") != input_digest(row):
            errors.append(f"{row['id']}: teacher input content/protocol mismatch")
    if errors:
        return errors, {}
    errors, stats = attach_train_targets(rows, mapping, args.target_missing)
    if args.gold_weight:
        for row in rows:
            if "train_target" in row:
                row["train_target"] = [(1 - args.gold_weight) * q + args.gold_weight * y
                                       for q, y in zip(row["train_target"], row["target"])]
    stats.update(gold_weight=args.gold_weight, allow_legacy_targets=args.allow_legacy_targets)
    return errors, stats


# ---------------------------------------------------------------------------
# length-bucketed plan, microbatch selection, LR schedule
# ---------------------------------------------------------------------------
def batch_plan(rows, lengths, seed, epochs=1, global_batch=64):
    """Deterministic length buckets, global batches, shuffled batch order."""
    result = []
    for epoch in range(epochs):
        def ordering(index):
            key = hashlib.sha256(
                json.dumps([seed, epoch, rows[index]["id"]], sort_keys=True).encode()).hexdigest()
            return lengths[index], key
        order = sorted(range(len(rows)), key=ordering)
        batches = [order[i:i + global_batch] for i in range(0, len(order), global_batch)]
        random.Random(seed + epoch).shuffle(batches)
        result.extend(batches)
    return result


def pick_micro(max_length, world, budget, global_batch=64, candidates=(8, 4, 2, 1)):
    for micro in candidates:
        if global_batch % (world * micro) == 0 and micro * max_length <= budget:
            return micro
    raise ValueError(f"no microbatch in {candidates} fits world={world}, "
                     f"max_length={max_length}, budget={budget}")


def split_batch(indices, lengths, world, budget, global_batch=64):
    """Return per-rank microbatch partitions and the chosen micro size."""
    micro = min(8, global_batch // world)
    maximum = max(lengths[index] for index in indices)
    while micro > 1 and micro * maximum > budget:
        micro //= 2
    if maximum > budget or global_batch % (world * micro):
        raise ValueError(f"incompatible world={world}, micro={micro}, max_length={maximum}, "
                         f"budget={budget}, global_batch={global_batch}")
    padded = indices + [indices[-1]] * (global_batch - len(indices))
    partitions = [
        [[(position, padded[position]) for position in
          range(offset + rank * micro, offset + (rank + 1) * micro)]
         for offset in range(0, global_batch, world * micro)]
        for rank in range(world)
    ]
    return partitions, micro


def lr_factor(step, total):
    index = step - 1
    warmup = max(1, math.ceil(0.03 * total))
    if index < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (index - warmup) / max(1, total - warmup)))


# ---------------------------------------------------------------------------
# graph regulariser
# ---------------------------------------------------------------------------
def cosine_kernel(label_vectors):
    """Non-negative cosine similarity between label-token output embeddings."""
    x = F.normalize(label_vectors.float(), dim=-1)
    return (x @ x.T).clamp_min(0.0)


def normalised_laplacian(similarity):
    matrix = similarity.clone()
    matrix.fill_diagonal_(0.0)
    degree = matrix.sum(-1)
    inv = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    return torch.eye(matrix.shape[0], dtype=matrix.dtype) - inv[:, None] * matrix * inv[None, :]


def build_graph_cache(kind, ks, cosine=None, device=None):
    """Return {K: G_K}; G_K is a KxK PSD matrix or None for the none graph."""
    if kind == "none" or not ks:
        return {}
    if kind == "identity":
        return {k: torch.eye(k, dtype=torch.float32, device=device) for k in ks}
    if kind == "cosine":
        if cosine is None:
            raise ValueError("cosine graph requires label embedding vectors")
        cache = {}
        for k in ks:
            matrix = normalised_laplacian(cosine[:k, :k])
            cache[k] = matrix.to(device) if device is not None else matrix
        return cache
    raise ValueError(f"unknown graph kind {kind!r}")


def loss_terms(logits, targets, lambda_g, graph_cache):
    """Per-row total loss, BCE term and graph term.

    logits: [B, 255]; targets: list of B lists (length K_i, values in [0,1]).
    The BCE term is exactly (1/K) sum_i softplus(s_i) - t_i s_i.
    """
    losses, bces, graphs = [], [], []
    for index, values in enumerate(targets):
        k = len(values)
        scores = logits[index, :k].float()
        target = torch.as_tensor(values, dtype=torch.float32, device=scores.device)
        bce = F.binary_cross_entropy_with_logits(scores, target, reduction="mean")
        graph = scores.new_zeros(())
        if lambda_g != 0.0:
            matrix = graph_cache.get(k)
            if matrix is None:
                raise ValueError(f"no graph matrix for K={k}")
            residual = torch.sigmoid(scores) - target
            graph = (lambda_g / k) * (residual @ matrix @ residual)
        losses.append(bce + graph)
        bces.append(bce)
        graphs.append(graph)
    return torch.stack(losses), torch.stack(bces), torch.stack(graphs)


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------
def _counts(target, probs, threshold):
    tp = fp = fn = tn = 0
    for value, probability in zip(target, probs):
        predicted = probability >= threshold
        if value:
            if predicted:
                tp += 1
            else:
                fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _k_bucket(k):
    if k <= 4:
        return "2-4"
    if k == 5:
        return "5"
    return "6+"


def _validated_predictions(rows, predictions):
    by_id = {}
    for item in predictions:
        by_id[item["id"]] = item
    result = []
    for row in rows:
        item = by_id.get(row["id"])
        if item is None:
            raise ValueError(f"missing prediction for {row['id']}")
        k = len(row["options"])
        probs = item["p"]
        if len(probs) != k:
            raise ValueError(f"{row['id']}: {len(probs)} probabilities != K={k}")
        result.append((row, probs, float(item["bce"])))
    return result


def _legacy_compute_metrics(rows, predictions, threshold=0.5):
    """Micro candidate P/R/F1, question EM, group-wise F1 and mean BCE."""
    records = _validated_predictions(rows, predictions)
    groups = {
        "source": collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0]),
        "lang": collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0]),
        "k": collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0]),
        "k_bucket": collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0]),
    }
    total = [0, 0, 0, 0, 0, 0]  # tp, fp, fn, tn, exact questions, questions
    bce_sum = 0.0
    n_candidates = 0
    for row, probs, bce in records:
        target = row["target"]
        tp, fp, fn, tn = _counts(target, probs, threshold)
        exact = int(all((probability >= threshold) == bool(value)
                        for value, probability in zip(target, probs)))
        for bucket in (total,
                       groups["source"][str(row.get("source"))],
                       groups["lang"][str(row.get("lang"))],
                       groups["k"][str(len(row["options"]))],
                       groups["k_bucket"][_k_bucket(len(row["options"]))]):
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn
            bucket[3] += tn
            bucket[4] += exact
            bucket[5] += 1
        bce_sum += bce
        n_candidates += len(target)

    def report(bucket):
        precision, recall, f1 = _prf(bucket[0], bucket[1], bucket[2])
        n_questions = bucket[5]
        return {
            "n_questions": n_questions,
            "n_candidates": bucket[0] + bucket[1] + bucket[2] + bucket[3],
            "tp": bucket[0], "fp": bucket[1], "fn": bucket[2], "tn": bucket[3],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "question_em": bucket[4] / n_questions if n_questions else 0.0,
        }

    result = report(total)
    result.update({
        "threshold": threshold,
        "n_questions": len(rows),
        "n_candidates": n_candidates,
        "bce": bce_sum / len(rows) if rows else 0.0,
        "question_em_count": total[4],
        "groups": {name: {key: report(value) for key, value in sorted(values.items())}
                   for name, values in groups.items()},
    })
    return result


def compute_metrics(rows, predictions, threshold=0.5):
    from multiselect_metrics import compute_metrics as shared_metrics
    return shared_metrics(rows, predictions, threshold=threshold, decoder="auto")


def tune_threshold(rows, predictions, low=0.05, high=0.95, step=0.01):
    # Caller uses dev only. Explicit single-choice tasks always use argmax.
    grid = []
    steps = int(round((high - low) / step)) + 1
    for index in range(steps):
        threshold = round(low + index * step, 2)
        record = compute_metrics(rows, predictions, threshold)
        grid.append({key: record[key] for key in
                     ("threshold", "precision", "recall", "f1", "question_em")})
    best_em = max(grid, key=lambda item: (item["question_em"], item["f1"], item["precision"]))
    best_f1 = max(grid, key=lambda item: (item["f1"], item["question_em"]))
    return {
        "range": [low, high, step],
        "grid": grid,
        "best_by_question_em": best_em,
        "best_by_f1": best_f1,
    }


# ---------------------------------------------------------------------------
# FSDP readout module
# ---------------------------------------------------------------------------
class Readout(torch.nn.Module):
    """Answer slot -> 255 legal-label logits (frozen output head)."""

    def __init__(self, base, label_ids, full_rank, objective="bce"):
        super().__init__()
        if objective not in ("bce", "set"):
            raise ValueError(f"unknown readout objective {objective!r}")
        self.base = base
        self.full_rank = full_rank
        self.objective = objective
        self.register_buffer("label_ids", torch.tensor(label_ids, dtype=torch.long),
                             persistent=False)
        if not full_rank:
            weight = base.get_base_model().get_output_embeddings().weight[label_ids]
            self.register_buffer("legal_weight", weight.detach().clone(), persistent=False)
        head = (base if full_rank else base.get_base_model()).get_output_embeddings()
        if objective == "set":
            self.count_head = torch.nn.Linear(head.weight.shape[1], len(label_ids) + 1,
                                              device=head.weight.device, dtype=head.weight.dtype)
            # A uniform allowed-count prior, independent of candidate logit shifts.
            torch.nn.init.zeros_(self.count_head.weight)
            torch.nn.init.zeros_(self.count_head.bias)
        else:
            self.count_head = None

    def forward(self, input_ids, attention_mask):
        base = self.base if self.full_rank else self.base.get_base_model()
        backbone = getattr(base.model, "language_model", base.model)
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=True).last_hidden_state
        slot = hidden[torch.arange(len(hidden), device=hidden.device),
                      attention_mask.sum(-1) - 1]
        if self.full_rank:
            weight = base.get_output_embeddings().weight.index_select(
                0, self.label_ids.to(dtype=torch.long))
        else:
            weight = self.legal_weight
        logits = F.linear(slot, weight).float()
        if self.count_head is None:
            return logits
        counts = self.count_head(slot.to(dtype=self.count_head.weight.dtype)).float()
        return logits, counts


def load_count_head(wrapper, directory, required=True):
    directory = Path(directory)
    meta_path = directory / "decision_readout.json"
    state_path = directory / "decision_count_head.pt"
    if not meta_path.exists() and not state_path.exists() and not required:
        return False
    if wrapper.count_head is None:
        raise ValueError("set checkpoint requires Readout(..., objective='set')")
    if not meta_path.exists() or not state_path.exists():
        raise ValueError(f"incomplete set readout checkpoint in {directory}")
    meta = json.loads(meta_path.read_text())
    if (meta.get("schema") != READOUT_SCHEMA or meta.get("objective") != "set"
            or meta.get("max_candidates") != len(wrapper.label_ids)
            or meta.get("hidden_size") != wrapper.count_head.in_features
            or meta.get("label_ids") != wrapper.label_ids.cpu().tolist()):
        raise ValueError("count-head schema, tokenizer label ids or dimensions mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite count-head checkpoint")
    wrapper.count_head.load_state_dict(state, strict=True)
    return True


def save_count_head(wrapper, directory, state=None):
    if wrapper.count_head is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = wrapper.count_head.state_dict()
    torch.save({name: value.detach().cpu().float() for name, value in state.items()},
               directory / "decision_count_head.pt")
    save_json(directory / "decision_readout.json", {
        "schema": READOUT_SCHEMA, "objective": "set", "decoder": "set_map",
        "max_candidates": len(wrapper.label_ids),
        "hidden_size": wrapper.count_head.in_features,
        "label_ids": wrapper.label_ids.cpu().tolist(),
    })


# ---------------------------------------------------------------------------
# validation-only (no model weights)
# ---------------------------------------------------------------------------
def check_modules_meta(model_path, modules_path, expect_trainable):
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    targets = json.loads(Path(modules_path).read_text())
    config = AutoConfig.from_pretrained(model_path)
    cls = (Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5"
           else Qwen3_5ForCausalLM)
    with torch.device("meta"):
        model = cls._from_config(config)
    named_modules = dict(model.named_modules())
    named_params = dict(model.named_parameters())
    missing = [name for name in targets if name not in named_modules]
    nonlinear = [name for name in targets
                 if name in named_modules and not isinstance(named_modules[name], torch.nn.Linear)]
    no_weight = [name for name in targets if name + ".weight" not in named_params]
    trainable = 0
    for name in targets:
        if name + ".weight" in named_params:
            trainable += named_params[name + ".weight"].numel()
        if name + ".bias" in named_params:
            trainable += named_params[name + ".bias"].numel()
    head = model.get_output_embeddings()
    embeddings = model.get_input_embeddings()
    result = {
        "ok": not (missing or nonlinear or no_weight),
        "model_type": config.model_type,
        "architecture": type(model).__name__,
        "modules": len(targets),
        "missing": missing[:10],
        "nonlinear": nonlinear[:10],
        "no_weight": no_weight[:10],
        "trainable_parameters": trainable,
        "expected_trainable_parameters": expect_trainable,
        "trainable_matches": (expect_trainable in (0, None)
                              or trainable == expect_trainable),
        "output_head_shape": list(head.weight.shape),
        "input_embedding_shape": list(embeddings.weight.shape),
        "ties_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
    }
    del model
    return result


def prompt_length_stats(lengths, max_input_tokens):
    ordered = sorted(lengths)

    def percentile(fraction):
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "n": len(lengths),
        "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": max(lengths) if lengths else 0,
        "total": sum(lengths),
        "over_max_input": sum(1 for value in lengths if value > max_input_tokens),
    }


def plan_micro_histogram(lengths, world, budget, global_batch=64):
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches = [order[i:i + global_batch] for i in range(0, len(order), global_batch)]
    counts = collections.Counter()
    problems = []
    for batch in batches:
        maximum = max(lengths[index] for index in batch)
        try:
            counts[pick_micro(maximum, world, budget, global_batch)] += 1
        except ValueError as error:
            problems.append(str(error))
    return {"batches": len(batches),
            "micro_counts": {str(key): value for key, value in sorted(counts.items())},
            "problems": problems[:10],
            "problem_count": len(problems)}


def validate_only(args):
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    errors, warnings_list = [], []
    mode = args.target or ("file" if args.target_file else "gold")
    if mode == "file" and not args.target_file:
        errors.append("--target file requires --target-file")
    if not Path(args.model).exists():
        errors.append(f"model path does not exist: {args.model}")
    if not Path(args.modules).exists():
        errors.append(f"modules file does not exist: {args.modules}")

    train_rows = dev_rows = None
    try:
        train_rows = read_dataset(args.train, "train")
        if args.limit:
            train_rows = train_rows[:args.limit]
    except Exception as error:
        errors.append(f"train dataset: {error}")
    try:
        dev_rows = read_dataset(args.dev, "dev")
        if args.limit_dev:
            dev_rows = dev_rows[:args.limit_dev]
    except Exception as error:
        errors.append(f"dev dataset: {error}")

    report = {
        "mode": "validate-only",
        "ok": False,
        "model": str(args.model),
        "modules": str(args.modules),
        "target_mode": mode,
        "max_input_tokens": args.max_input_tokens,
        "max_batch_tokens": args.max_batch_tokens,
        "world_size": args.world_size,
        "global_batch": args.global_batch,
        "errors": errors,
        "warnings": warnings_list,
    }
    if errors:
        report["ok"] = False
        _emit_validation(report, args.validate_output)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    labels = option_labels(tokenizer)
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    report["labels"] = {
        "count": len(labels),
        "first": labels[:8],
        "last": labels[-3:],
        "token_ids_first": label_ids[:8],
    }
    if len(set(label_ids)) != 255:
        errors.append("label token ids are not unique")

    for name, rows, split in (("train", train_rows, "train"), ("dev", dev_rows, "dev")):
        if rows is None:
            continue
        stats = dataset_stats(rows)
        prompts = [make_prompt(tokenizer, row, labels) for row in rows]
        lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                   for prompt in prompts]
        prompt_stats = prompt_length_stats(lengths, args.max_input_tokens)
        if prompt_stats["over_max_input"]:
            errors.append(f"{split}: {prompt_stats['over_max_input']} prompt(s) exceed "
                          f"--max-input-tokens {args.max_input_tokens}")
        if max((len(row["options"]) for row in rows), default=0) > len(labels):
            errors.append(f"{split}: K exceeds the {len(labels)} legal labels")
        stats["prompts"] = prompt_stats
        stats["micro_plan"] = plan_micro_histogram(
            lengths, args.world_size, args.max_batch_tokens, args.global_batch)
        if stats["micro_plan"]["problem_count"]:
            errors.append(f"{split}: {stats['micro_plan']['problem_count']} batch(es) have "
                          f"no valid microbatch")
        report[name] = stats
        if split == "train":
            report["train_plan"] = {
                "planned_steps": math.ceil(len(rows) / args.global_batch),
                "warmup_steps": max(1, math.ceil(0.03 * math.ceil(len(rows) / args.global_batch))),
                "eval_every": max(1, math.ceil(
                    math.ceil(len(rows) / args.global_batch) * args.eval_every_frac)),
            }
            if args.max_steps > 0:
                report["train_plan"]["stop_step"] = min(
                    report["train_plan"]["planned_steps"], args.max_steps)

    # Same target/provenance validation as the training path; no weights loaded.
    target_report = {"mode": mode, "objective": args.objective}
    try:
        target_errors, stats = prepare_training_targets(train_rows, args)
        errors.extend(target_errors)
        target_report.update(stats)
    except (ValueError, OSError) as exc:
        errors.append(f"targets: {exc}")
    report["targets"] = target_report

    # module / trainable-parameter check on meta tensors (no weights)
    modules_check = None
    try:
        modules_check = check_modules_meta(args.model, args.modules, args.expect_trainable)
        if not modules_check["ok"]:
            errors.append("modules.json does not match the model architecture")
        if not modules_check["trainable_matches"]:
            errors.append(f"trainable parameter count {modules_check['trainable_parameters']} "
                          f"!= expected {args.expect_trainable}")
    except Exception as error:
        errors.append(f"module check failed: {error}")
    report["modules_check"] = modules_check

    report["ok"] = not errors
    report["errors"] = errors
    _emit_validation(report, args.validate_output)
    return 0 if report["ok"] else 1


def _emit_validation(report, output):
    if output:
        save_json(output, report)
        print(f"validation report -> {output}", file=sys.stderr)
    printable = dict(report)
    if "modules_check" in printable and printable["modules_check"]:
        printable["modules_check"] = {
            key: value for key, value in printable["modules_check"].items()
            if key in ("ok", "modules", "trainable_parameters", "trainable_matches",
                       "output_head_shape")
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CPU self test
# ---------------------------------------------------------------------------
def self_test():
    torch.manual_seed(0)
    checks = {}

    logits = torch.randn(4, 7)
    targets = [[1.0, 0.0, 0.7, 0.0, 1.0], [0.0, 1.0], [1.0, 1.0, 0.0], [0.3, 0.6]]
    losses, bces, graphs = loss_terms(logits, targets, 0.0, {})
    manual = torch.stack([
        (F.softplus(logits[i, :len(t)]) - torch.tensor(t) * logits[i, :len(t)]).mean()
        for i, t in enumerate(targets)])
    checks["bce_matches_formula"] = bool(torch.allclose(bces, manual, atol=1e-6))
    checks["g_zero_when_lambda_zero"] = bool(torch.count_nonzero(graphs) == 0)

    cache = build_graph_cache("identity", [2, 3, 4, 5])
    lam = 0.3
    losses_g, _, graphs_g = loss_terms(logits, targets, lam, cache)
    expected = []
    for i, t in enumerate(targets):
        scores = logits[i, :len(t)]
        residual = torch.sigmoid(scores) - torch.tensor(t)
        expected.append(lam / len(t) * (residual * residual).sum())
    checks["identity_graph_matches"] = bool(
        torch.allclose(graphs_g[:3], torch.stack(expected[:3]), atol=1e-6))

    kernel = cosine_kernel(torch.randn(9, 5))
    laplacian = normalised_laplacian(kernel)
    checks["cosine_laplacian_psd"] = bool(
        torch.linalg.eigvalsh(laplacian).min().item() > -1e-4)
    checks["cosine_kernel_nonnegative"] = bool(kernel.min().item() >= 0)

    probe = torch.randn(3, 7, requires_grad=True)
    probe_targets = [[0.5, 1.0, 0.0], [0.2, 0.4, 0.6, 0.8], [0.1, 0.9, 0.3, 0.7, 0.5, 0.5]]
    probe_cache = build_graph_cache("cosine", [3, 4, 6], cosine_kernel(torch.randn(20, 6)))
    probe_loss, _, _ = loss_terms(probe, probe_targets, 0.2, probe_cache)
    probe_loss.sum().backward()
    checks["graph_gradients_finite"] = bool(torch.isfinite(probe.grad).all())

    parts, micro = split_batch(list(range(7)), [700] * 7, 8, 8192)
    coverage = [position for rank_parts in parts for part in rank_parts
                for position, _ in part]
    checks["split_batch_cover"] = (micro == 8 and sorted(coverage) == list(range(64)))

    rows = [
        {"id": "a", "source": "s1", "lang": "zh", "options": [{}] * 3, "target": [1, 0, 1]},
        {"id": "b", "source": "s2", "lang": "en", "options": [{}] * 2, "target": [0, 1]},
        {"id": "c", "source": "s1", "lang": "en", "options": [{}] * 5,
         "target": [1, 1, 0, 0, 1]},
    ]
    predictions = [
        {"id": "a", "p": [0.9, 0.4, 0.8], "bce": 0.1},
        {"id": "b", "p": [0.2, 0.7], "bce": 0.2},
        {"id": "c", "p": [0.8, 0.6, 0.1, 0.2, 0.9], "bce": 0.3},
    ]
    metrics = compute_metrics(rows, predictions, 0.5)
    checks["question_em"] = metrics["question_em"] == 1.0
    checks["candidate_counts"] = (metrics["tp"] == 6 and metrics["fp"] == 0
                                  and metrics["fn"] == 0)
    checks["group_keys"] = set(metrics["groups"]) == {
        "source", "lang", "k", "k_bucket", "selection_mode"}

    tuned = tune_threshold(rows, predictions)
    checks["threshold_grid_size"] = len(tuned["grid"]) == 91
    checks["threshold_best_em"] = tuned["best_by_question_em"]["question_em"] == 1.0

    lengths = [10, 20, 30, 40, 50, 60, 70]
    plan = batch_plan([{"id": str(i)} for i in range(7)], lengths, seed=43)
    checks["batch_plan"] = [len(batch) for batch in plan] == [7]
    checks["micro_choice"] = (pick_micro(700, 8, 8192) == 8
                              and pick_micro(2500, 8, 8192) == 2
                              and pick_micro(5000, 8, 8192) == 1)
    try:
        pick_micro(9000, 8, 8192)
        checks["micro_rejects_oversize"] = False
    except ValueError:
        checks["micro_rejects_oversize"] = True

    checks["lr_warmup"] = abs(lr_factor(1, 100) - 1 / 3) < 1e-12
    checks["lr_end"] = lr_factor(100, 100) < 1e-3

    result = {"ok": all(checks.values()), "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------
def resolve_resume(path: Path, override_step: int):
    path = Path(path)
    if path.is_dir() and (path / "config.json").exists():
        checkpoint = path
    elif path.is_dir():
        candidates = []
        for child in path.iterdir():
            if child.is_dir() and (child / "config.json").exists():
                try:
                    step = int(child.name.rsplit("-", 1)[-1])
                except ValueError:
                    continue
                candidates.append((step, child))
        if not candidates:
            raise ValueError(f"no HF checkpoint found under {path}")
        checkpoint = max(candidates)[1]
    else:
        raise ValueError(f"--resume expects a checkpoint or run directory: {path}")
    step = override_step
    if step is None or step < 0:
        try:
            step = int(checkpoint.name.rsplit("-", 1)[-1])
        except ValueError:
            raise ValueError(f"cannot infer step from {checkpoint}; pass --resume-step")
    return checkpoint, step


def verify_export(path):
    path = Path(path)
    result = {"path": str(path), "ok": True, "errors": [], "files": {}}
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).exists():
            result["errors"].append(f"missing {name}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(mapping.values()))
        missing = [name for name in shards if not (path / name).exists()]
        result["files"]["shards"] = len(shards)
        result["files"]["missing_shards"] = missing
        if missing:
            result["errors"].append(f"missing {len(missing)} weight shard(s)")
    else:
        shards = sorted(str(item.name) for item in path.glob("*.safetensors"))
        result["files"]["shards"] = len(shards)
        if not shards:
            result["errors"].append("no safetensors weights found")
    result["files"]["bytes"] = sum(
        item.stat().st_size for item in path.iterdir() if item.is_file())
    result["ok"] = not result["errors"]
    return result


def materialize_lora(base_path, adapter_dir, destination):
    """Merge a saved PEFT adapter into a standalone BF16 HF model (rank 0)."""
    from peft import PeftModel
    from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    config = AutoConfig.from_pretrained(base_path)
    cls = (Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5"
           else Qwen3_5ForCausalLM)
    model = cls.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cpu",
                                attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    return merged


# ---------------------------------------------------------------------------
# main training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, help="train JSONL")
    parser.add_argument("--dev", type=Path, help="dev JSONL")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--modules", type=Path, default=DEFAULT_MODULES)
    parser.add_argument("--output", type=Path, help="run directory")
    parser.add_argument("--target", choices=["gold", "file"], default=None,
                        help="gold multi-hot target or soft targets from --target-file")
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--target-missing", choices=["error", "gold"], default="error")
    parser.add_argument("--objective", choices=["bce", "set"], default="bce")
    parser.add_argument("--gold-weight", type=float, default=0.0,
                        help="fixed gold fraction in a file target; gold-only ignores this flag")
    parser.add_argument("--count-target", choices=["teacher", "gold"], default="teacher",
                        help="set: coherent set CE; gold: marginal-BCE plus gold count-CE ablation")
    parser.add_argument("--allow-legacy-targets", action="store_true",
                        help="explicit BCE-only diagnostic override for old untyped target files")
    parser.add_argument("--lambda-g", type=float, default=0.0)
    parser.add_argument("--graph", choices=["none", "identity", "cosine"], default="none")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--eval-every-frac", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="limit train rows (smoke run)")
    parser.add_argument("--limit-dev", type=int, default=0, help="limit dev rows")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, help="checkpoint or run directory")
    parser.add_argument("--resume-step", type=int, default=-1)
    parser.add_argument("--expect-trainable", type=int, default=EXPECTED_TRAINABLE,
                        help="0 disables the trainable-parameter count check")
    parser.add_argument("--lora", action="store_true",
                        help="optional DDP LoRA path; the experiment requires full-rank")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--world-size", type=int, default=8,
                        help="world size assumed by --validate-only planning")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.train or not args.dev:
        parser.error("--train and --dev are required")
    if args.lambda_g < 0:
        parser.error("--lambda-g must be >= 0")
    if not math.isfinite(args.gold_weight) or not 0 <= args.gold_weight <= 1:
        parser.error("--gold-weight must be in [0,1]")
    if args.objective == "set" and (args.lambda_g != 0 or args.graph != "none"):
        parser.error("set objective does not mix in the legacy label-token graph penalty")
    if args.objective == "bce" and args.count_target != "teacher":
        parser.error("--count-target gold is a set-head ablation and requires --objective set")
    if args.global_batch < 1 or args.epochs < 1:
        parser.error("--global-batch and --epochs must be positive")
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    if args.max_input_tokens < 1 or args.max_batch_tokens < 1:
        parser.error("token limits must be positive")
    if args.graph != "none" and args.lambda_g == 0:
        log("[warn] --graph has no effect with --lambda-g 0")
    if args.graph == "none" and args.lambda_g > 0:
        log("[warn] --graph none makes the graph term zero")

    if args.validate_only:
        return validate_only(args)

    if not args.output:
        parser.error("--output is required for training")
    if args.eval_every_frac <= 0:
        parser.error("--eval-every-frac must be positive")
    if args.global_batch % 64 != 0 and args.global_batch != 64:
        log(f"[warn] non-standard global batch {args.global_batch}; official is 64")

    mode = args.target or ("file" if args.target_file else "gold")
    if mode == "file" and not args.target_file:
        parser.error("--target file requires --target-file")

    # ---- data / tokenizer / plan (all ranks) ----
    train_rows = read_dataset(args.train, "train")
    if args.limit:
        train_rows = train_rows[:args.limit]
    dev_rows = read_dataset(args.dev, "dev")
    if args.limit_dev:
        dev_rows = dev_rows[:args.limit_dev]

    target_errors, target_stats = prepare_training_targets(train_rows, args)
    if target_errors:
        raise ValueError("target validation failed:\n  " + "\n  ".join(target_errors[:20]))

    from transformers import AutoConfig, AutoTokenizer
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    labels = option_labels(tokenizer)
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

    log(f"[data] train={len(train_rows)} dev={len(dev_rows)} targets={target_stats}")

    train_prompts = [make_prompt(tokenizer, row, labels) for row in train_rows]
    dev_prompts = [make_prompt(tokenizer, row, labels) for row in dev_rows]
    train_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                     for prompt in train_prompts]
    dev_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                   for prompt in dev_prompts]
    if max(train_lengths + dev_lengths) > args.max_input_tokens:
        raise ValueError(f"prompt length {max(train_lengths + dev_lengths)} exceeds "
                         f"--max-input-tokens {args.max_input_tokens}; inputs are never truncated")

    batches = batch_plan(train_rows, train_lengths, args.seed, args.epochs, args.global_batch)
    planned_steps = len(batches)
    stop_step = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps
    eval_steps = {min(stop_step, max(1, math.ceil(planned_steps * args.eval_every_frac * i)))
                  for i in range(1, int(round(1 / args.eval_every_frac)) + 1)}
    eval_steps.add(stop_step)
    eval_steps = {step for step in eval_steps if step <= stop_step}
    log(f"[plan] rows={len(train_rows)} planned_steps={planned_steps} stop_step={stop_step} "
        f"eval_steps={sorted(eval_steps)} max_tokens={max(train_lengths)}")

    # ---- distributed setup ----
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; use --validate-only or --self-test on CPU")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if args.global_batch % world:
        raise ValueError(f"global batch {args.global_batch} is not divisible by world {world}")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.manual_seed(args.seed)

    base_path = Path(args.model)
    load_path = base_path
    start_step = 0
    if args.resume:
        if args.lora:
            raise ValueError("--resume is not supported for the optional LoRA path")
        load_path, start_step = resolve_resume(args.resume, args.resume_step)
        if start_step >= stop_step:
            raise ValueError(f"resume step {start_step} >= stop step {stop_step}")
        log(f"[resume] from {load_path} at step {start_step}")
        log("[resume] weights and step schedule restored; optimizer starts fresh")
    if args.objective == "bce" and any((load_path / name).exists() for name in
                                       ("decision_readout.json", "decision_count_head.pt")):
        raise ValueError("set checkpoint requires --objective set; refusing to discard its count head")

    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            if (output / "training_complete.json").exists():
                raise ValueError(f"{output} already contains a completed run")
            if (output / "training.jsonl").exists():
                raise ValueError(f"{output}/training.jsonl exists; use a new output or --resume")
        save_json(output / "training_config.json", {
            "train": str(args.train), "dev": str(args.dev), "model": str(args.model),
            "modules": str(args.modules), "output": str(output),
            "target_mode": mode, "target_file": str(args.target_file) if args.target_file else None,
            "objective": args.objective, "count_target": args.count_target,
            "gold_weight": args.gold_weight, "allow_legacy_targets": args.allow_legacy_targets,
            "lambda_g": args.lambda_g, "graph": args.graph, "lr": args.lr,
            "global_batch": args.global_batch, "epochs": args.epochs,
            "max_input_tokens": args.max_input_tokens,
            "max_batch_tokens": args.max_batch_tokens,
            "eval_every_frac": args.eval_every_frac, "max_steps": args.max_steps,
            "limit": args.limit, "limit_dev": args.limit_dev, "seed": args.seed,
            "world_size": world, "planned_steps": planned_steps, "stop_step": stop_step,
            "eval_steps": sorted(eval_steps), "start_step": start_step,
            "lora": args.lora, "targets": target_stats,
        })
    torch.distributed.barrier()

    # ---- model ----
    config = AutoConfig.from_pretrained(load_path)
    cls = (Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5"
           else Qwen3_5ForCausalLM)
    targets = json.loads(Path(args.modules).read_text())
    full = not args.lora
    base = cls.from_pretrained(
        load_path, dtype=torch.float32 if full else torch.bfloat16,
        device_map="cpu" if full else str(device), attn_implementation="sdpa")
    base.requires_grad_(False)
    if full:
        modules = dict(base.named_modules())
        for name in targets:
            if not isinstance(modules[name], torch.nn.Linear):
                raise TypeError(f"{name} is not a language linear layer")
            modules[name].requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(args.seed)
        base = get_peft_model(base, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=targets, bias="none"))
    trainable = sum(parameter.numel() for parameter in base.parameters()
                    if parameter.requires_grad)
    if full and args.expect_trainable and trainable != args.expect_trainable:
        raise ValueError(f"trainable parameter count {trainable} != "
                         f"expected {args.expect_trainable}")
    log(f"[model] trainable_parameters={trainable} full_rank={full}")

    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False

    # graph cache (built before FSDP from frozen label embeddings)
    graph_cache = {}
    if args.lambda_g != 0.0 and args.graph != "none":
        head = (base.get_output_embeddings() if full
                else base.get_base_model().get_output_embeddings())
        vectors = head.weight.detach().float()[torch.tensor(label_ids)]
        kernel = cosine_kernel(vectors) if args.graph == "cosine" else None
        train_ks = sorted({len(row["options"]) for row in train_rows})
        graph_cache = build_graph_cache(args.graph, train_ks, kernel, device)
        log(f"[graph] kind={args.graph} lambda_g={args.lambda_g} Ks={train_ks}")

    wrapper = Readout(base, label_ids, full, objective=args.objective)
    if args.objective == "set":
        load_count_head(wrapper, load_path, required=bool(args.resume))
        count_trainable = sum(p.numel() for p in wrapper.count_head.parameters())
        log(f"[model] additional_count_head_parameters={count_trainable}")
        trainable += count_trainable
    params = None
    if full:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        model = FSDP(
            wrapper,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={type(base.model.language_model.layers[0])}),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16),
            device_id=device, use_orig_params=True, limit_all_gathers=True)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(wrapper.to(device), device_ids=[local], broadcast_buffers=False)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.0, foreach=False)

    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"))

    # resume selection state if present
    best_key = (float("-inf"), float("-inf"))
    best_record = None
    if args.resume and (output / "selection.json").exists():
        previous = json.loads((output / "selection.json").read_text())
        if "question_em" in previous:
            best_key = (float(previous["question_em"]), -float(previous.get("bce", float("inf"))))
            best_record = previous

    # Encode the exact pre-rendered SII prompt strings (chat template + answer cue).
    def encode_texts(indices, prompts):
        encoded = tokenizer([prompts[index] for index in indices], padding=True,
                            return_tensors="pt", add_special_tokens=False)
        return {name: value.to(device) for name, value in encoded.items()}

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        order = sorted(range(len(dev_rows)), key=lambda index: (dev_lengths[index],
                                                               dev_rows[index]["id"]))
        predictions = []
        offset = 0
        while offset < len(order):
            micro = 8
            group = order[offset:offset + world * micro]
            maximum = max(dev_lengths[index] for index in group)
            while micro > 1 and micro * maximum > args.max_batch_tokens:
                micro //= 2
            if micro * maximum > args.max_batch_tokens:
                raise ValueError(f"dev batch length {maximum} exceeds token budget")
            indices = order[offset:offset + world * micro]
            count = len(indices)
            padded = indices + [indices[-1]] * (world * micro - count)
            own = padded[rank * micro:(rank + 1) * micro]
            result = model(**encode_texts(own, dev_prompts))
            logits, count_logits = result if args.objective == "set" else (result, None)
            for position, index in enumerate(own):
                if rank * micro + position >= count:
                    continue
                row = dev_rows[index]
                k = len(row["options"])
                scores = logits[position, :k].float()
                target = torch.as_tensor(row["target"], dtype=torch.float32,
                                         device=scores.device)
                if args.objective == "set":
                    item = structured_prediction(scores.cpu().tolist(),
                                                 count_logits[position, :k + 1].cpu().tolist(), row)
                    item["candidate_logits"] = scores.cpu().tolist()
                    # Shared metrics compute marginal BCE identically to standalone eval.
                else:
                    item = {"p": torch.sigmoid(scores).cpu().tolist(),
                            "logits": scores.cpu().tolist()}
                    bce = F.binary_cross_entropy_with_logits(scores, target, reduction="mean")
                    item["bce"] = float(bce)
                predictions.append({"id": row["id"], **item})
            offset += count
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, predictions)
        merged = {}
        for part in gathered:
            for item in part:
                merged[item["id"]] = item
        record = compute_metrics(dev_rows, list(merged.values()), threshold=0.5)
        record["step"] = step
        if args.objective == "bce":
            record["threshold_tuning"] = tune_threshold(
                dev_rows, list(merged.values()), 0.05, 0.95, 0.01)
        record["selection"] = {
            "metric": "question_em:set_map" if args.objective == "set" else "question_em@0.5",
            "tie_break": "lower_bce",
            "question_em": record["question_em"],
            "bce": record["bce"],
        }
        model.train()
        return record

    def export(step):
        destination = output / f"checkpoint-{step}"
        if rank == 0 and destination.exists():
            shutil.rmtree(destination)
        torch.distributed.barrier()
        if full:
            from torch.distributed.fsdp import (
                FullStateDictConfig, StateDictType, FullyShardedDataParallel as FSDP)
            with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state = model.state_dict()
            if rank == 0:
                exported = {}
                count_state = {}
                for name, value in state.items():
                    if name.startswith("count_head."):
                        count_state[name.removeprefix("count_head.")] = value
                    elif name.startswith("base."):
                        exported[name.removeprefix("base.")] = (
                            value.to(torch.bfloat16) if value.is_floating_point() else value)
                    else:
                        raise ValueError(f"unexpected wrapper parameter in export: {name}")
                base.save_pretrained(destination, state_dict=exported,
                                     safe_serialization=True, max_shard_size="5GB")
                if args.objective == "set":
                    save_count_head(wrapper, destination, count_state)
                tokenizer.save_pretrained(destination)
                save_json(output / f"export_check-{step:06d}.json", verify_export(destination))
            del state
        else:
            if rank == 0:
                base.save_pretrained(destination)
                tokenizer.save_pretrained(destination)
                save_count_head(wrapper, destination)
        torch.distributed.barrier()

    model.train()
    for index, indices in enumerate(batches):
        step = index + 1
        if step <= start_step:
            continue
        if step > stop_step:
            break
        begin = time.monotonic()
        count = len(indices)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_factor(step, planned_steps)
        optimizer.zero_grad(set_to_none=True)
        partitions, micro = split_batch(indices, train_lengths, world,
                                        args.max_batch_tokens, args.global_batch)
        sums = torch.zeros(5, device=device, dtype=torch.float64)
        for part in partitions[rank]:
            batch_indices = [row_index for _, row_index in part]
            batch_targets = [train_rows[row_index]["train_target"] for row_index in batch_indices]
            result = model(**encode_texts(batch_indices, train_prompts))
            if args.objective == "set":
                losses, bces, graphs = set_loss_terms(
                    result[0], result[1], [train_rows[i] for i in batch_indices],
                    count_target=args.count_target)
            else:
                losses, bces, graphs = loss_terms(
                    result, batch_targets, args.lambda_g, graph_cache)
            active = torch.tensor([position < count for position, _ in part],
                                  device=device, dtype=losses.dtype)
            (losses * active).sum().mul(world / count).backward()
            sums[0] += (losses.detach() * active).sum()
            sums[1] += (bces.detach() * active).sum()
            sums[2] += (graphs.detach() * active).sum()
            sums[3] += active.sum()
        torch.distributed.all_reduce(sums)
        rows_seen = float(sums[3].item())
        grad = (model.clip_grad_norm_(args.grad_clip) if full
                else torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        if not torch.isfinite(sums).all() or not math.isfinite(float(grad)):
            raise FloatingPointError(f"non-finite loss/gradient at step {step}")
        optimizer.step()
        seconds = time.monotonic() - begin
        real_tokens = sum(train_lengths[row_index] for row_index in indices)
        record = {
            "step": step,
            "planned_steps": planned_steps,
            "stop_step": stop_step,
            "loss": float(sums[0] / rows_seen),
            ("bce" if args.objective == "bce" else "score_bce_diagnostic"):
                float(sums[1] / rows_seen),
            ("g_term" if args.objective == "bce" else "count_ce"):
                float(sums[2] / rows_seen),
            "objective": args.objective,
            "grad": float(grad),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": seconds,
            "real_tokens": real_tokens,
            "tokens_per_s": real_tokens / seconds if seconds else 0.0,
            "max_tokens": max(train_lengths[row_index] for row_index in indices),
            "micro": micro,
            "rows": count,
        }
        if rank == 0:
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            for key in ("loss", "bce", "g_term", "score_bce_diagnostic", "count_ce",
                        "grad", "lr", "seconds", "tokens_per_s"):
                if key in record:
                    writer.add_scalar("train/" + key, record[key], step)
            print(json.dumps(record), flush=True)

        if step in eval_steps:
            optimizer.zero_grad(set_to_none=True)
            evaluation = evaluate(step)
            key = (evaluation["question_em"], -evaluation["bce"])
            improved = key > best_key
            if improved:
                best_key = key
                best_record = {
                    "step": step,
                    "checkpoint": f"checkpoint-{step}",
                    "question_em": evaluation["question_em"],
                    "bce": evaluation["bce"],
                    "f1": evaluation["f1"],
                    "precision": evaluation["precision"],
                    "recall": evaluation["recall"],
                    "threshold": None if args.objective == "set" else 0.5,
                    "decoder": "set_map" if args.objective == "set" else "protocol_or_threshold",
                    "selection": "dev question EM; tie-break lower marginal BCE",
                }
                if "threshold_tuning" in evaluation:
                    best_record["best_threshold_by_question_em"] = (
                        evaluation["threshold_tuning"]["best_by_question_em"])
            if rank == 0:
                save_json(output / f"validation_{step:06d}.json", evaluation)
                for name, value in (
                        ("question_em", evaluation["question_em"]),
                        ("precision", evaluation["precision"]),
                        ("recall", evaluation["recall"]),
                        ("f1", evaluation["f1"]),
                        ("bce", evaluation["bce"])):
                    writer.add_scalar("validation/" + name, value, step)
                if "threshold_tuning" in evaluation:
                    best_threshold = evaluation["threshold_tuning"]["best_by_question_em"]
                    writer.add_scalar("validation/best_threshold_em", best_threshold["question_em"], step)
                    writer.add_scalar("validation/best_threshold", best_threshold["threshold"], step)
                writer.flush()
            if improved:
                log(f"[eval] step {step}: EM={evaluation['question_em']:.4f} "
                    f"F1={evaluation['f1']:.4f} BCE={evaluation['bce']:.4f} -> checkpoint")
                export(step)
            elif rank == 0:
                log(f"[eval] step {step}: EM={evaluation['question_em']:.4f} "
                    f"F1={evaluation['f1']:.4f} BCE={evaluation['bce']:.4f} (no improvement)")

    # ---- final export / selection ----
    if best_record is None:
        raise RuntimeError("training finished without a validation entry")
    selected = output / best_record["checkpoint"]
    if not selected.exists():
        raise RuntimeError(f"selected checkpoint is missing: {selected}")
    if rank == 0:
        final = output / "final"
        if final.exists():
            if final.is_symlink() or final.is_file():
                final.unlink()
            else:
                shutil.rmtree(final)
        if full:
            shutil.copytree(selected, final)
        else:
            # intermediate LoRA checkpoints hold adapters; merge the selected one
            materialize_lora(args.model, selected, final)
            tokenizer.save_pretrained(final)
            if args.objective == "set":
                for name in ("decision_readout.json", "decision_count_head.pt"):
                    shutil.copy2(selected / name, final / name)
        save_json(output / "export_check.json", verify_export(final))
        save_json(output / "selection.json", best_record)
        save_json(output / "training_complete.json", {
            "steps": stop_step,
            "planned_steps": planned_steps,
            "start_step": start_step,
            "world_size": world,
            "trainable_parameters": trainable,
            "selected": best_record,
            "final_export": str(final),
            "lambda_g": args.lambda_g,
            "graph": args.graph,
            "objective": args.objective, "count_target": args.count_target,
            "gold_weight": args.gold_weight,
        })
        if writer is not None:
            writer.close()
        print(json.dumps({"selected": best_record, "final_export": str(final)}), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

