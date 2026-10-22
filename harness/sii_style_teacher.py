#!/usr/bin/env python3
"""Reasoning-chain teacher with candidate-conditioned Yes/No membership readout.

Phase A (``generate``) samples finite reasoning chains with vLLM. The ``direct``
and ``verify`` views use different instructions, retain the full question and all
options, and never include gold answers or the gold answer count. Every cache row
records its view, checkpoint identity, input digest, chain digest, and finish state.

Phase B (``readout``) builds one full-context prompt per chain and candidate. It
names the candidate and asks whether that candidate belongs in the correct answer
set. The tokenizer must map the explicit Yes/No spellings to two distinct single
tokens, including at the actual answer boundary. Only those two output-head rows
are projected. ``q_i = softmax([yes_logit, no_logit])[0]`` is averaged across
complete valid chain vectors; no sampled answer string is parsed.

The old ``sigmoid``, ``centered_sigmoid`` and ``softmax`` label-slot transforms are
available only as explicitly marked diagnostics. Centering label logits by their
logsumexp gives ``a_i/(1+a_i)`` for a label softmax probability ``a_i``; it does not
produce candidate membership events. Diagnostic q values are not soft targets.

Output preserves the older q fields and adds ``per_chain_membership``, aligned
``per_chain_ids``/``per_chain_metadata``, separate excluded-chain metadata,
``option_keys``, ``q_semantics``, teacher/readout identity, and provenance. Each
membership vector contains marginal readouts; it is not an empirically measured
joint set distribution. Truncated chains follow ``--truncated-policy`` and remain
visible in coverage. An absent/invalid candidate invalidates the whole chain;
rows without any valid chain have ``q: null``, never fabricated negatives.

The native modes are generate, readout, merge-reasoning, merge-readout,
validate-only (tokenizer/prompts; no model forward), and check-model. For example:

  python -I harness/sii_style_teacher.py readout --model /path/to/Qwen3.5-4B \
      --teacher-id qwen-base --teacher-view direct --q-transform membership \
      --input data/multiselect_pilot/train.jsonl \
      --reasoning runs/teacher/direct/reasoning/train.jsonl \
      --output runs/teacher/direct/q/membership/exclude/parts/train/part-00000.jsonl \
      --samples 2 --readout-batch-size 4 --truncated-policy exclude

``harness/run_sii_style_teacher.sh`` defaults to two chains and one pilot shard;
``DRY_RUN=1`` prints commands without launching generation or readout. Outputs are
isolated by teacher view, q transform and truncation policy. An optional explicit
``--checkpoint-id`` records an immutable revision supplied by the operator; the
default local model path is identified as a path, not a verified weight hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Native CLI invocations use python -I, which omits the script directory.
HARNESS_DIR = str(Path(__file__).resolve().parent)
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)
from set_distillation import input_digest

DEFAULT_MODEL = ROOT / "models/Qwen3.5-4B"
DEFAULT_DATA = ROOT / "data/multiselect"
DEFAULT_RUN_ROOT = ROOT / "runs/qz/sii_style_teacher"
ANSWER_CUE = "The correct answer is ("
MEMBERSHIP_ANSWER_CUE = "Membership answer:\n"
MEMBERSHIP_READOUT = "candidate_yes_no_v1"
REASONING_PROMPT_VERSION = "candidate_reasoning_v1"
VIEW_INSTRUCTIONS = {
    "direct": (
        "Reason through the question directly. Evaluate every option against the "
        "given facts and explain which options are supported. Keep the reasoning "
        "specific to each option and follow the declared selection protocol."),
    "verify": (
        "Independently verify every option. Search for a counterexample, a "
        "contradiction, or an unmet condition for each candidate, then explain "
        "which candidates survive those checks. Follow the declared selection protocol."),
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Chains that never produced reasoning text; phase B skips them when averaging.
UNUSABLE_FINISH_REASONS = {"missing", "error", "abort", "prompt_too_long"}

PROTECTED_PATHS = {
    (DEFAULT_DATA / name).resolve()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "summary.json")
}


def log(message):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# SII prompts (verbatim from innerjev/sii_decision/prompts.py)
# ---------------------------------------------------------------------------
def text(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True)


def option_labels(tokenizer):
    """SII's 255 single-token option labels (A..Z, AA..JT)."""
    candidates = list(string.ascii_uppercase)
    candidates += [a + b for a in string.ascii_uppercase
                   for b in string.ascii_uppercase]
    labels = [label for label in candidates
              if len(tokenizer.encode(label, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise ValueError("The tokenizer must provide the original 255 single-token "
                         f"option labels, got {len(labels)}.")
    return labels


def make_body(row, labels):
    """Byte-identical to ``prompts.body(row, labels)``."""
    choices = "\n".join(f"({labels[i]}) {o['key']}: {o['desc']}"
                        for i, o in enumerate(row["options"]))
    return (f'State:\n{text(row["state"])}\nQuestion: {text(row["question"])}\n'
            f"Options:\n{choices}").rstrip()


def selection_protocol_text(row):
    """Use only task protocol, never a gold set or its cardinality."""
    selection_mode = row.get("selection_mode", "multi")
    if selection_mode == "single":
        selection = "Task protocol: exactly one option is correct."
    elif selection_mode == "multi":
        selection = "Task protocol: multiple options may be correct."
    else:
        raise ValueError(f"{row.get('id')}: invalid selection_mode {selection_mode!r}")
    if "allow_empty" in row and not isinstance(row["allow_empty"], bool):
        raise ValueError(f"{row.get('id')}: allow_empty must be boolean")
    if selection_mode == "single" and row.get("allow_empty", False):
        raise ValueError(f"{row.get('id')}: single-choice protocol cannot allow an empty set")
    if row.get("allow_empty", False) is False:
        selection += " At least one option must be selected."
    elif row.get("allow_empty") is True and selection_mode != "single":
        selection += " An empty selection is permitted."
    return selection


def make_think_prefix(tokenizer, row, labels, teacher_view=None):
    # None retains the old SII formatting for diagnostic prompt comparisons.
    content = make_body(row, labels)
    if teacher_view is not None:
        if teacher_view not in VIEW_INSTRUCTIONS:
            raise ValueError(f"unknown teacher_view: {teacher_view!r}")
        content += "\n\n" + selection_protocol_text(row) + "\n" + VIEW_INSTRUCTIONS[teacher_view]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)


def make_sii_prompt(tokenizer, row, labels):
    """SII's no-thinking answer-slot prompt (the reference format)."""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": make_body(row, labels)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prefix + ANSWER_CUE


def extract_reasoning(raw_text):
    """Generated reasoning with chat control tokens removed.

    vLLM's ``stop=["</think>"]`` normally excludes the marker from ``text``; if a
    build includes it, keep only the part before it.  A leading ``<think>`` (the
    template opener echoed back) is stripped.  Trailing newlines are kept out so
    the read prompt joins cleanly; an empty chain yields the empty block.
    """
    if not isinstance(raw_text, str):
        return ""
    text = raw_text
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    text = text.strip()
    while text.startswith("<think>"):
        text = text[len("<think>"):].lstrip()
    text = text.replace("<think>", " ").replace("</think>", " ")
    return text.strip()


def make_read_prompt(think_prefix, reasoning):
    """think_prefix + reasoning + closing think marker + SII answer cue."""
    return think_prefix + reasoning + "\n</think>\n\n" + ANSWER_CUE


def binary_token_info(tokenizer, yes_token="Yes", no_token="No"):
    """Resolve the named binary events from this tokenizer; never assume IDs."""
    resolved = {}
    for event, token in (("yes", yes_token), ("no", no_token)):
        if not isinstance(token, str) or not token or token.strip().lower() != event:
            raise ValueError(f"{event}_token must explicitly spell {event!r}")
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or isinstance(ids[0], bool) or not isinstance(ids[0], int):
            raise ValueError(f"{event} token {token!r} must encode to one token, got {ids}")
        resolved[f"{event}_token"] = token
        resolved[f"{event}_token_id"] = ids[0]
    if resolved["yes_token_id"] == resolved["no_token_id"]:
        raise ValueError("yes and no must resolve to distinct token IDs")
    return resolved


def validate_binary_token_boundary(tokenizer, prompt, token_info):
    """Ensure each answer is one continuation token after the actual cue."""
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for event in ("yes", "no"):
        token = token_info[f"{event}_token"]
        actual = tokenizer.encode(prompt + token, add_special_tokens=False)
        if actual != prefix_ids + [token_info[f"{event}_token_id"]]:
            raise ValueError(f"{event} token {token!r} is not a single continuation "
                             "after the membership answer cue")


def make_membership_prompt(tokenizer, row, labels, reasoning, candidate_index,
                           yes_token="Yes", no_token="No", teacher_view="direct"):
    """Full context, shared reasoning, and one explicitly named membership event."""
    if not 0 <= candidate_index < len(row["options"]):
        raise ValueError("candidate index outside the input option order")
    if teacher_view not in VIEW_INSTRUCTIONS:
        raise ValueError(f"unknown teacher_view: {teacher_view!r}")
    candidate = row["options"][candidate_index]
    content = (
        make_body(row, labels) + "\n\n" + selection_protocol_text(row)
        + f"\n\nReasoning evidence (teacher view: {teacher_view}):\n{reasoning}"
        + "\n\nCandidate under evaluation:\n"
        + f"({labels[candidate_index]}) {candidate['key']}: {candidate['desc']}"
        + "\n\nDoes this specific candidate belong in the correct answer set for "
        "the question? Consider the full question, all options, and the reasoning "
        "above. Judge this candidate only. Reply with exactly "
        + repr(yes_token) + " if it belongs, or " + repr(no_token)
        + " if it does not belong. Do not output an option label or any explanation.")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return prefix + MEMBERSHIP_ANSWER_CUE


def membership_probability(logits_row):
    """P(Yes | Yes or No, full context, one candidate), stable to logit offsets."""
    values = _finite_values(logits_row)
    if len(values) != 2:
        raise ValueError("membership readout requires exactly [yes_logit, no_logit]")
    maximum = max(values)
    yes, no = [math.exp(value - maximum) for value in values]
    return yes / (yes + no)


def _finite_values(values):
    values = values.tolist() if hasattr(values, "tolist") else list(values)
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or not math.isfinite(value) for value in values):
        raise ValueError("readout values must be a nonempty finite numeric vector")
    return [float(value) for value in values]


def q_semantics(transform):
    return "candidate_membership" if transform == "membership" else f"diagnostic_{transform}"


def teacher_identity(args):
    return {
        "teacher_id": args.teacher_id or f"{Path(args.model).name}:{args.teacher_view}",
        "teacher_view": args.teacher_view,
        "checkpoint": args.checkpoint_id or str(Path(args.model).resolve()),
        "reasoning_prompt_version": REASONING_PROMPT_VERSION,
    }


def input_fingerprint(row):
    """Compatibility name for the shared teacher/target input identity."""
    return input_digest(row)


def chain_provenance(args, row):
    return {
        **teacher_identity(args), "schema_version": 2,
        "input_sha256": input_fingerprint(row), "input_digest": input_digest(row),
        "option_keys": [option["key"] for option in row["options"]],
        "option_alignment": "input_option_order",
        "generation": {key: getattr(args, key) for key in
                       ("seed", "temperature", "top_p", "top_k", "max_tokens")},
    }


def _validate_identity(record, expected_identity, row=None, allow_legacy=False):
    ident = record.get("id", "unknown")
    legacy = not any(key in record for key in expected_identity)
    if legacy and allow_legacy:
        return False
    for key, expected in expected_identity.items():
        if record.get(key) != expected:
            raise ValueError(f"{ident}: cache {key} mismatch: expected {expected!r}, "
                             f"got {record.get(key)!r}")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{ident}: cache provenance missing")
    if row is not None:
        expected_digest = input_digest(row)
        if (provenance.get("input_sha256") != expected_digest
                or provenance.get("input_digest") != expected_digest
                or record.get("input_digest") != expected_digest):
            raise ValueError(f"{ident}: cache input fingerprint mismatch")
        keys = [option["key"] for option in row["options"]]
        if record.get("option_keys") != keys:
            raise ValueError(f"{ident}: cache option_keys mismatch")
    return True


# ---------------------------------------------------------------------------
# json / atomic io
# ---------------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        path.write_bytes(data[: cut + 1] if cut >= 0 else b"")
        log(f"truncated a partial last line in {path}")


def _guard_output(path):
    resolved = Path(path).resolve()
    if resolved in PROTECTED_PATHS:
        raise SystemExit(f"refusing to overwrite curated dataset file: {resolved}")


def atomic_write_text(path, payload):
    path = Path(path)
    _guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class JsonlAppender:
    """Append JSON rows as one os.write per record on an O_APPEND fd."""

    def __init__(self, path, resume=False):
        self.path = Path(path)
        _guard_output(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.path.unlink(missing_ok=True)
        else:
            sanitize_jsonl(self.path)
        self.fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)

    def append(self, records):
        for record in records:
            data = (json.dumps(record, ensure_ascii=False, allow_nan=False)
                    + "\n").encode("utf-8")
            view = memoryview(data)
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError(f"short write to {self.path}")
                view = view[written:]
        os.fsync(self.fd)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def validate_readout_record(record, row=None, expected_identity=None, args=None,
                            require_q=True):
    """Fail closed on stale provenance, invalid targets, or broken chain alignment."""
    ident = record.get("id", "unknown")
    if expected_identity is not None:
        _validate_identity(record, expected_identity, row=row)
    n = record.get("n_options")
    keys = record.get("option_keys")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError(f"{ident}: invalid n_options")
    if not isinstance(keys, list) or len(keys) != n or len(set(keys)) != n:
        raise ValueError(f"{ident}: missing or invalid option_keys")
    if row is not None and keys != [option["key"] for option in row["options"]]:
        raise ValueError(f"{ident}: option_keys mismatch")
    transform = record.get("q_transform")
    if transform not in ("membership", "sigmoid", "centered_sigmoid", "softmax"):
        raise ValueError(f"{ident}: unknown q_transform")
    if record.get("q_semantics") != q_semantics(transform):
        raise ValueError(f"{ident}: q_semantics mismatch")
    if args is not None:
        if transform != args.q_transform or record.get("n_chains") != args.samples:
            raise ValueError(f"{ident}: cached transform or sample count mismatch")
        provenance = record.get("provenance") or {}
        if provenance.get("truncated_policy") != args.truncated_policy:
            raise ValueError(f"{ident}: cached truncated_policy mismatch")
        if transform == "membership":
            for key in ("yes_token", "no_token"):
                if (provenance.get("binary_tokens") or {}).get(key) != getattr(args, key):
                    raise ValueError(f"{ident}: cached binary {key} mismatch")
    q = record.get("q")
    if q is None:
        if require_q:
            raise ValueError(f"{ident}: cached q is missing; rerun without --resume")
        if record.get("valid_chains") != 0:
            raise ValueError(f"{ident}: null q with nonzero valid_chains")
        return
    values = _finite_values(q)
    if len(values) != n or any(not 0 <= value <= 1 for value in values):
        raise ValueError(f"{ident}: invalid q probability vector")
    field = {"membership": "per_chain_membership", "sigmoid": "per_chain_probs",
             "centered_sigmoid": "per_chain_centered", "softmax": "per_chain_softmax"}[transform]
    vectors = record.get(field)
    if not isinstance(vectors, list) or not vectors or len(vectors) != record.get("valid_chains"):
        raise ValueError(f"{ident}: missing or inconsistent {field}")
    for vector in vectors:
        vector = _finite_values(vector)
        if len(vector) != n or any(not 0 <= value <= 1 for value in vector):
            raise ValueError(f"{ident}: invalid {field} vector")
    mean = [sum(column) / len(vectors) for column in zip(*vectors)]
    if any(abs(value - expected) > 1e-7 for value, expected in zip(values, mean)):
        raise ValueError(f"{ident}: q differs from the per-chain mean")
    ids = record.get("per_chain_ids")
    metadata = record.get("per_chain_metadata")
    if (not isinstance(ids, list) or len(ids) != len(vectors)
            or len(set(ids)) != len(ids) or not isinstance(metadata, list)
            or len(metadata) != len(ids)
            or any(item.get("chain") != chain or item.get("included") is not True
                   for chain, item in zip(ids, metadata))):
        raise ValueError(f"{ident}: invalid per-chain alignment")
    if transform == "membership" and record.get("readout") != MEMBERSHIP_READOUT:
        raise ValueError(f"{ident}: invalid membership readout version")


def load_done_ids(path, *, expected_identity=None, input_rows=None, args=None,
                  reasoning_digests=None):
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        ident = record.get("id")
        if not isinstance(ident, str) or not ident or ident in done:
            raise ValueError(f"{path}: invalid or duplicate cached id {ident!r}")
        if expected_identity is not None:
            if input_rows is not None and ident not in input_rows:
                raise ValueError(f"{ident}: cached id absent from input")
            validate_readout_record(record, row=(input_rows or {}).get(ident),
                                    expected_identity=expected_identity, args=args)
            if (reasoning_digests is not None and
                    record.get("provenance", {}).get("reasoning_cache_sha256") != reasoning_digests):
                raise ValueError(f"{ident}: cached readout uses different reasoning content")
        done.add(ident)
    return done


def load_done_chains(path, *, expected_identity=None, input_rows=None, args=None):
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        ident, chain = record.get("id"), record.get("chain")
        if not isinstance(ident, str) or not ident or type(chain) is not int or chain < 0:
            raise ValueError(f"{path}: invalid cached reasoning id/chain")
        if (ident, chain) in done:
            raise ValueError(f"{ident}: duplicate cached chain {chain}")
        if expected_identity is not None:
            if input_rows is not None and ident not in input_rows:
                raise ValueError(f"{ident}: cached reasoning id absent from input")
            _validate_identity(record, expected_identity, row=(input_rows or {}).get(ident))
            if args is not None:
                generation = (record.get("provenance") or {}).get("generation", {})
                for key in ("seed", "temperature", "top_p", "top_k", "max_tokens"):
                    if generation.get(key) != getattr(args, key):
                        raise ValueError(f"{ident}: cached generation {key} mismatch")
                if chain >= args.samples:
                    raise ValueError(f"{ident}: cached chain exceeds --samples")
        done.add((ident, chain))
    return done


# ---------------------------------------------------------------------------
# input validation / model check
# ---------------------------------------------------------------------------
def validate_row(row):
    ident = row.get("id")
    if not isinstance(ident, str) or not ident:
        raise ValueError("each row needs a nonempty string id")
    if not isinstance(row.get("source"), str) or not row["source"]:
        raise ValueError(f"{ident}: missing source")
    for field in ("state", "question"):
        if not isinstance(row.get(field), str):
            raise ValueError(f"{ident}: {field} must be a string")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        raise ValueError(f"{ident}: requires 2..255 options")
    keys = []
    for option in options:
        key = option.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{ident}: option keys must be nonempty strings")
        if not isinstance(option.get("desc"), str):
            raise ValueError(f"{ident}: option description must be a string")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{ident}: duplicate option keys")
    selection_protocol_text(row)
    return row


def check_model_complete(model_path):
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"{model_path}: model.safetensors.index.json not found")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_path / name).exists()]
    if missing:
        raise SystemExit(f"{model_path}: missing {len(missing)}/{len(shards)} "
                         f"weight shards: {missing[:5]}")
    return shards


# ---------------------------------------------------------------------------
# phase A: vLLM reasoning generation
# ---------------------------------------------------------------------------
def build_engine(args):
    import torch
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=False,
        seed=args.seed,
        disable_log_stats=True,
    )
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    llm = LLM(**kwargs)
    sampling = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["</think>"],
        skip_special_tokens=False,
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return llm, sampling, gpu_name


def _generate_with_retry(llm, prompts, sampling):
    try:
        outputs = llm.generate(prompts, sampling, use_tqdm=False,
                               tokenization_kwargs={"add_special_tokens": False})
        if len(outputs) != len(prompts):
            raise ValueError("generation engine returned fewer or extra prompt results")
        return outputs
    except Exception as error:  # noqa: BLE001 - vLLM raises many engine types
        if len(prompts) > 1:
            middle = len(prompts) // 2
            log(f"[warn] batch of {len(prompts)} failed ({error}); splitting")
            return (_generate_with_retry(llm, prompts[:middle], sampling)
                    + _generate_with_retry(llm, prompts[middle:], sampling))
        log(f"[error] prompt failed: {error}")
        return [None]


def chain_record(row, chain, reasoning, finish_reason, prompt_tokens,
                 completion_tokens, *, provenance=None):
    record = {
        "id": row["id"],
        "chain": chain,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "source": row.get("source"),
        "lang": row.get("lang"),
        "truncated": finish_reason == "length",
    }
    if provenance is not None:
        record.update({key: provenance[key] for key in
                       ("teacher_id", "teacher_view", "checkpoint", "reasoning_prompt_version")})
        record["option_keys"] = provenance["option_keys"]
        record["input_digest"] = provenance["input_digest"]
        record["provenance"] = dict(provenance)
        record["reasoning_sha256"] = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
    return record


def generate_split(llm, sampling, tokenizer, labels, args, input_path, output_path):
    rows = read_jsonl(input_path)
    seen_ids = set()
    for row in rows:
        validate_row(row)
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate input id {row['id']!r}")
        seen_ids.add(row["id"])
        if len(row["options"]) > len(labels):
            raise ValueError(f"{row['id']}: more options than SII labels")
    if args.limit:
        rows = rows[: args.limit]

    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.shards == args.shard]
    output_path = Path(output_path)
    _guard_output(output_path)
    profile_path = output_path.with_name(
        output_path.stem.replace("part-", "profile-") + ".json")

    done = set()
    if args.resume:
        sanitize_jsonl(output_path)
        done = load_done_chains(output_path, expected_identity=teacher_identity(args),
                                input_rows={row["id"]: row for row in rows}, args=args)
    if done:
        selected = [(index, row) for index, row in selected
                    if any((row["id"], chain) not in done
                           for chain in range(args.samples))]
        log(f"[{input_path.name} shard {args.shard}] resume: {len(done)} chains done, "
            f"{len(selected)} rows left")

    profile = {
        "phase": "reasoning",
        **teacher_identity(args),
        "input": str(input_path), "output": str(output_path),
        "model": str(args.model), "shard": args.shard, "shards": args.shards,
        "samples": args.samples, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "rows_selected": len(rows) // args.shards
        + (1 if args.shard < len(rows) % args.shards else 0),
        "chains": 0, "truncated": 0, "errors": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
        "gen_seconds": 0.0, "completion_tokens_per_s": 0.0,
        "gpu_name": None, "status": "empty",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not selected:
        atomic_write_json(profile_path, profile)
        log(f"[{input_path.name} shard {args.shard}] nothing left to do")
        return profile

    prompts = [make_think_prefix(tokenizer, row, labels, teacher_view=args.teacher_view)
               for _, row in selected]
    lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
               for prompt in prompts]
    budget = args.max_model_len - args.max_tokens
    slots = []  # (row, prompt, length, chains_to_generate)
    over_budget = []
    for (index, row), prompt, length in zip(selected, prompts, lengths):
        needed = [chain for chain in range(args.samples)
                  if (row["id"], chain) not in done]
        if length > budget:
            over_budget.append((row, length, needed))
        elif needed:
            slots.append((row, prompt, length, needed))
    if over_budget:
        log(f"[{input_path.name} shard {args.shard}] {len(over_budget)} prompts exceed "
            f"max_model_len-max_tokens={budget}")

    begin = time.monotonic()
    chains_written = 0
    with JsonlAppender(output_path, resume=args.resume and output_path.exists()) as out:
        for row, length, needed in over_budget:
            out.append([chain_record(row, chain, "", "prompt_too_long", length, 0,
                                     provenance=chain_provenance(args, row))
                        for chain in needed])
            chains_written += len(needed)
            profile["errors"] += len(needed)
            profile["prompt_tokens"] += length

        for offset in range(0, len(slots), args.batch_size):
            batch = slots[offset: offset + args.batch_size]
            outputs = _generate_with_retry(llm, [item[1] for item in batch], sampling)
            records = []
            for (row, _prompt, length, needed), request_output in zip(batch, outputs):
                # vLLM prefills once and decodes n branches, so count the prompt once.
                profile["prompt_tokens"] += length
                if request_output is None:
                    for chain in needed:
                        records.append(chain_record(row, chain, "", "error", length, 0,
                                                     provenance=chain_provenance(args, row)))
                    profile["errors"] += len(needed)
                    chains_written += len(needed)
                    continue
                completions = list(getattr(request_output, "outputs", None) or [])
                for chain in needed:
                    if chain >= len(completions):
                        records.append(chain_record(row, chain, "", "missing", length, 0,
                                                     provenance=chain_provenance(args, row)))
                        profile["errors"] += 1
                        chains_written += 1
                        continue
                    completion = completions[chain]
                    finish = completion.finish_reason or "unknown"
                    token_ids = getattr(completion, "token_ids", None)
                    tokens = len(token_ids) if token_ids is not None else 0
                    records.append(chain_record(
                        row, chain, extract_reasoning(completion.text or ""),
                        finish, length, tokens, provenance=chain_provenance(args, row)))
                    profile["truncated"] += 1 if finish == "length" else 0
                    profile["errors"] += 1 if finish in ("error", "abort") else 0
                    profile["completion_tokens"] += tokens
                    chains_written += 1
                chains_written += 0
            out.append(records)
            profile["chains"] += len(records)
            if args.progress_every and chains_written % args.progress_every < len(records):
                elapsed = time.monotonic() - begin
                log(f"[{input_path.name} shard {args.shard}] {chains_written} chains, "
                    f"{profile['completion_tokens']/max(elapsed,1e-9):.0f} tok/s")

    gen_seconds = time.monotonic() - begin
    profile.update({
        "chains": chains_written,
        "gen_seconds": round(gen_seconds, 3),
        "completion_tokens_per_s": (round(profile["completion_tokens"] / gen_seconds, 2)
                                    if gen_seconds else 0.0),
        "status": "ok" if slots else "all_over_budget",
    })
    atomic_write_json(profile_path, profile)
    log(f"[{input_path.name} shard {args.shard}] done: {chains_written} chains, "
        f"{profile['truncated']} truncated, {profile['errors']} errors, "
        f"{profile['completion_tokens_per_s']:.0f} tok/s")
    return profile


def run_generate(args):
    check_model_complete(args.model)
    inputs = [Path(path) for path in args.input]
    outputs = [Path(path) for path in args.output]
    if not inputs:
        raise SystemExit("generate requires at least one --input")
    if len(outputs) != len(inputs):
        raise SystemExit("--output must be given once per --input")
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    labels = option_labels(tokenizer)

    pending = []
    for input_path, output_path in zip(inputs, outputs):
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[: args.limit]
        if args.resume:
            sanitize_jsonl(output_path)
        done = (load_done_chains(output_path, expected_identity=teacher_identity(args),
                                 input_rows={row["id"]: row for row in rows}, args=args)
                if args.resume else set())
        shard_rows = [row for index, row in enumerate(rows)
                      if index % args.shards == args.shard]
        pending.append(any((row["id"], chain) not in done
                           for row in shard_rows for chain in range(args.samples)))
    if not any(pending):
        log(f"[shard {args.shard}] all inputs complete; skipping model load")
        return 0

    load_begin = time.monotonic()
    llm, sampling, gpu_name = build_engine(args)
    load_seconds = time.monotonic() - load_begin
    log(f"[shard {args.shard}] model loaded in {load_seconds:.1f}s on {gpu_name}")

    status = 0
    for position, (input_path, output_path) in enumerate(zip(inputs, outputs)):
        try:
            profile = generate_split(llm, sampling, tokenizer, labels, args,
                                     input_path, output_path)
            profile["gpu_name"] = gpu_name
            profile["load_seconds"] = round(load_seconds, 3) if position == 0 else 0.0
            atomic_write_json(output_path.with_name(
                output_path.stem.replace("part-", "profile-") + ".json"), profile)
        except Exception as error:  # noqa: BLE001 - keep the next split
            log(f"[error] {input_path.name}: {error}")
            status = 1
    return status


# ---------------------------------------------------------------------------
# phase B: transformers SII readout
# ---------------------------------------------------------------------------
class ReadoutEngine:
    """Last-slot hidden state -> explicitly selected output-token logits."""

    def __init__(self, model_path, device="cuda", gpu_index=0, max_input_tokens=8192,
                 q_transform="membership", yes_token="Yes", no_token="No"):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"
        self.labels = option_labels(self.tokenizer)
        self.q_transform = q_transform
        self.binary_tokens = binary_token_info(self.tokenizer, yes_token, no_token)
        self.label_ids_cpu = [self.tokenizer.encode(label, add_special_tokens=False)[0]
                             for label in self.labels]
        output_ids = ([self.binary_tokens["yes_token_id"], self.binary_tokens["no_token_id"]]
                      if q_transform == "membership" else self.label_ids_cpu)

        config = AutoConfig.from_pretrained(model_path)
        cls = (Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5"
               else Qwen3_5ForCausalLM)
        kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
        if device == "cpu":
            kwargs["device_map"] = "cpu"
        else:
            torch.cuda.set_device(gpu_index)
            kwargs["device_map"] = {"": gpu_index}
        model, info = cls.from_pretrained(model_path, output_loading_info=True, **kwargs)
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError("incomplete model loading: " + str({
                "missing_keys": info.get("missing_keys"),
                "mismatched_keys": info.get("mismatched_keys")}))
        model.eval()
        self.model = model
        self.backbone = getattr(model.model, "language_model", model.model)
        self.head = model.get_output_embeddings()
        self.input_device = model.get_input_embeddings().weight.device
        self.output_ids = torch.tensor(output_ids, device=self.head.weight.device)
        # Qwen's output head is linear: project only the two requested events
        # (or the 255 labels for legacy diagnostics), not the entire vocabulary.
        self.readout_weight = self.head.weight.index_select(0, self.output_ids).detach()
        head_bias = getattr(self.head, "bias", None)
        self.readout_bias = (head_bias.index_select(0, self.output_ids).detach()
                             if head_bias is not None else None)
        self.device = device

    def logits(self, prompts):
        """Return [B, 2] in [Yes, No] order, or legacy [B, 255] logits."""
        torch = self.torch
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt",
                                 add_special_tokens=False)
        lengths = encoded["attention_mask"].sum(-1)
        longest = int(lengths.max())
        if longest > self.max_input_tokens:
            raise ValueError(f"input length {longest} exceeds "
                             f"--max-input-tokens {self.max_input_tokens}")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.backbone(**encoded, use_cache=False,
                                   return_dict=True).last_hidden_state
            end = (lengths - 1).to(hidden.device)
            slot = hidden[torch.arange(len(prompts), device=hidden.device), end]
            slot = slot.to(self.head.weight.device)
            logits = torch.nn.functional.linear(
                slot, self.readout_weight, self.readout_bias).float()
        return logits.cpu()


def chain_probabilities(logits_row, n_options):
    """Legacy diagnostics only; none is a candidate membership probability.

    If a_i is the label softmax probability over all labels, the centered
    transform is a_i / (1 + a_i), so even the sum over all labels is below one.
    Subtracting logsumexp does not turn label evidence into independent events.
    """
    full = _finite_values(logits_row)
    if not 0 < n_options <= len(full):
        raise ValueError("not enough label logits for the requested options")
    values = full[:n_options]
    maximum = max(full)
    lse = maximum + math.log(sum(math.exp(value - maximum) for value in full))
    def sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)
    exps = [math.exp(value - max(values)) for value in values]
    return ([sigmoid(value) for value in values],
            [sigmoid(value - lse) for value in values],
            [value / sum(exps) for value in exps])


def _readout_with_retry(engine, prompts):
    """Preserve prompt positions when splitting failures; one None per failure."""
    if not prompts:
        return []
    try:
        values = engine.logits(prompts)
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        if len(values) != len(prompts):
            raise ValueError("engine returned a different number of logits than prompts")
        return values
    except Exception as error:  # noqa: BLE001 - keep the shard alive
        if len(prompts) > 1:
            middle = len(prompts) // 2
            log(f"[warn] readout batch of {len(prompts)} failed ({error}); splitting")
            left = _readout_with_retry(engine, prompts[:middle])
            right = _readout_with_retry(engine, prompts[middle:])
            return left + right
        log(f"[error] readout prompt failed: {error}")
        return [None]


def load_reasoning_by_id(paths, wanted_ids, *, expected_identity=None,
                         input_rows=None, samples=None, allow_legacy=False):
    grouped = defaultdict(list)
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            ident = row.get("id")
            if ident not in wanted_ids:
                continue
            chain = row.get("chain")
            if type(chain) is not int or chain < 0 or (samples is not None and chain >= samples):
                raise ValueError(f"{ident}: cached chain outside requested sample range")
            if (ident, chain) in seen:
                raise ValueError(f"{ident}: duplicate cached reasoning chain {chain}")
            seen.add((ident, chain))
            if not isinstance(row.get("reasoning"), str):
                raise ValueError(f"{ident}: cached reasoning must be text")
            if expected_identity is not None:
                verified = _validate_identity(row, expected_identity,
                                               row=(input_rows or {}).get(ident),
                                               allow_legacy=allow_legacy)
                if verified and row.get("reasoning_sha256") != hashlib.sha256(
                        row["reasoning"].encode("utf-8")).hexdigest():
                    raise ValueError(f"{ident}: cached reasoning digest mismatch")
            if ("truncated" in row and (not isinstance(row["truncated"], bool)
                    or row["truncated"] != (row.get("finish_reason") == "length"))):
                raise ValueError(f"{ident}: inconsistent truncated flag")
            grouped[ident].append(row)
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry["chain"])
    return grouped


def _chain_metadata(entry, included, exclusion_reason=None):
    metadata = {
        "chain": entry["chain"], "finish_reason": entry.get("finish_reason"),
        "truncated": entry.get("truncated", entry.get("finish_reason") == "length"),
        "included": included, "prompt_tokens": entry.get("prompt_tokens"),
        "completion_tokens": entry.get("completion_tokens"),
        "reasoning_sha256": entry.get("reasoning_sha256"),
        "generation": (entry.get("provenance") or {}).get("generation"),
    }
    if exclusion_reason is not None:
        metadata["exclusion_reason"] = exclusion_reason
    return metadata


def readout_split(engine, tokenizer, labels, args, input_path, reasoning_paths,
                  output_path):
    rows = read_jsonl(input_path)
    row_ids = set()
    for row in rows:
        validate_row(row)
        if row["id"] in row_ids:
            raise ValueError(f"duplicate input id {row['id']!r}")
        row_ids.add(row["id"])
        if len(row["options"]) > len(labels):
            raise ValueError(f"{row['id']}: more options than SII labels")
    if args.limit:
        rows = rows[:args.limit]
    input_by_id = {row["id"]: row for row in rows}
    identity = teacher_identity(args)
    membership = args.q_transform == "membership"
    if membership and args.allow_legacy_reasoning_cache:
        raise ValueError("membership readout cannot use unverified legacy reasoning caches")
    token_info = binary_token_info(tokenizer, args.yes_token, args.no_token) if membership else None
    reasoning_digests = {str(path): file_sha256(path) for path in reasoning_paths}
    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.shards == args.shard]
    output_path = Path(output_path)
    _guard_output(output_path)
    profile_path = output_path.with_name(
        output_path.stem.replace("part-", "profile-") + ".json")
    done = set()
    if args.resume:
        sanitize_jsonl(output_path)
        done = load_done_ids(output_path, expected_identity=identity,
                             input_rows=input_by_id, args=args,
                             reasoning_digests=reasoning_digests)
    selected = [(index, row) for index, row in selected if row["id"] not in done]
    reasoning = load_reasoning_by_id(
        reasoning_paths, {row["id"] for _, row in selected}, expected_identity=identity,
        input_rows=input_by_id, samples=args.samples,
        allow_legacy=args.allow_legacy_reasoning_cache)
    profile = {
        "phase": "readout", **identity, "q_transform": args.q_transform,
        "q_semantics": q_semantics(args.q_transform),
        "readout": MEMBERSHIP_READOUT if membership else "sii_label_slot_diagnostic_v1",
        "truncated_policy": args.truncated_policy,
        "input": str(input_path), "reasoning": [str(p) for p in reasoning_paths],
        "output": str(output_path), "model": str(args.model),
        "shard": args.shard, "shards": args.shards, "samples": args.samples,
        "rows": 0, "rows_with_q": 0, "rows_zero_chain": 0, "resumed_rows": len(done),
        "chains": 0, "requested_chains": len(selected) * args.samples,
        "missing_requested_chains": 0, "skipped_unusable": 0,
        "excluded_truncated_chains": 0, "included_truncated_chains": 0,
        "invalid_readout_chains": 0, "seconds": 0.0, "status": "empty",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not selected:
        atomic_write_json(profile_path, profile)
        log(f"[{input_path.name} shard {args.shard}] nothing left to do")
        return profile
    begin = time.monotonic()
    with JsonlAppender(output_path, resume=args.resume and output_path.exists()) as out:
        for offset in range(0, len(selected), args.readout_batch_size):
            batch = selected[offset:offset + args.readout_batch_size]
            prompts, slots = [], []
            entries_by_id, excluded_by_id = {}, defaultdict(list)
            for _index, row in batch:
                ident = row["id"]
                available = {entry["chain"]: entry for entry in reasoning.get(ident, [])}
                usable = []
                for chain in range(args.samples):
                    entry = available.get(chain)
                    if entry is None:
                        excluded_by_id[ident].append({
                            "chain": chain, "finish_reason": "missing", "truncated": False,
                            "included": False, "exclusion_reason": "missing_reasoning"})
                        profile["missing_requested_chains"] += 1
                        continue
                    if (entry.get("finish_reason") in UNUSABLE_FINISH_REASONS
                            or not entry["reasoning"].strip()):
                        excluded_by_id[ident].append(_chain_metadata(entry, False, "unusable_reasoning"))
                        profile["skipped_unusable"] += 1
                        continue
                    if entry.get("truncated", entry.get("finish_reason") == "length") \
                            and args.truncated_policy == "exclude":
                        excluded_by_id[ident].append(_chain_metadata(entry, False, "truncated_policy"))
                        profile["excluded_truncated_chains"] += 1
                        continue
                    usable.append(entry)
                    if membership:
                        for candidate in range(len(row["options"])):
                            prompt = make_membership_prompt(
                                tokenizer, row, labels, entry["reasoning"], candidate,
                                args.yes_token, args.no_token, args.teacher_view)
                            validate_binary_token_boundary(tokenizer, prompt, token_info)
                            prompts.append(prompt)
                            slots.append((ident, chain, candidate))
                    else:
                        # Legacy transforms are reproducible diagnostics only.
                        view = args.teacher_view if "teacher_view" in entry else None
                        prefix = make_think_prefix(tokenizer, row, labels, teacher_view=view)
                        prompts.append(make_read_prompt(prefix, entry["reasoning"]))
                        slots.append((ident, chain, 0))
                entries_by_id[ident] = usable
            logits_by_chain = defaultdict(dict)
            for begin_prompt in range(0, len(prompts), args.readout_batch_size):
                end_prompt = begin_prompt + args.readout_batch_size
                values = _readout_with_retry(engine, prompts[begin_prompt:end_prompt])
                for slot, value in zip(slots[begin_prompt:end_prompt], values):
                    ident, chain, candidate = slot
                    logits_by_chain[(ident, chain)][candidate] = value
            records = []
            for _index, row in batch:
                ident, n = row["id"], len(row["options"])
                source_vectors, metadata, chain_ids, chain_logits = [], [], [], []
                sigmoids, centereds, softmaxes, memberships = [], [], [], []
                for entry in entries_by_id[ident]:
                    values = logits_by_chain[(ident, entry["chain"])]
                    try:
                        required = n if membership else 1
                        if set(values) != set(range(required)) or any(value is None for value in values.values()):
                            raise ValueError("candidate readout failed")
                        if membership:
                            vector = [round(membership_probability(values[i]), 8) for i in range(n)]
                            memberships.append(vector)
                            logits_out = [[round(value, 6) for value in _finite_values(values[i])]
                                          for i in range(n)]
                        else:
                            sig, centered, soft = chain_probabilities(values[0], n)
                            sig, centered, soft = ([round(value, 8) for value in vector]
                                                   for vector in (sig, centered, soft))
                            sigmoids.append(sig)
                            centereds.append(centered)
                            softmaxes.append(soft)
                            vector = {"sigmoid": sig, "centered_sigmoid": centered,
                                      "softmax": soft}[args.q_transform]
                            logits_out = [round(value, 6) for value in _finite_values(values[0])[:n]]
                    except (ValueError, TypeError, OverflowError) as error:
                        excluded_by_id[ident].append(_chain_metadata(entry, False, "invalid_readout"))
                        profile["invalid_readout_chains"] += 1
                        log(f"[error] {ident} chain {entry['chain']}: {error}")
                        continue
                    source_vectors.append(vector)
                    chain_ids.append(entry["chain"])
                    metadata.append(_chain_metadata(entry, True))
                    chain_logits.append(logits_out)
                    profile["included_truncated_chains"] += int(metadata[-1]["truncated"])
                q = ([round(sum(column) / len(source_vectors), 8) for column in zip(*source_vectors)]
                     if source_vectors else None)
                provenance = {
                    "schema_version": 2, "input_sha256": input_fingerprint(row),
                    "input_digest": input_digest(row),
                    "reasoning_prompt_version": REASONING_PROMPT_VERSION,
                    "option_alignment": "input_option_order", "option_permutation": None,
                    "reasoning_paths": [str(path) for path in reasoning_paths],
                    "reasoning_cache_sha256": reasoning_digests,
                    "model_path": str(args.model),
                    "checkpoint_identity_kind": "explicit" if args.checkpoint_id else "local_path",
                    "truncated_policy": args.truncated_policy,
                    "legacy_reasoning_cache_allowed": args.allow_legacy_reasoning_cache,
                    "binary_tokens": token_info,
                    "joint_semantics": "marginal_readouts_only; no empirical joint distribution",
                }
                record = {
                    "id": ident, "source": row.get("source"), "lang": row.get("lang"),
                    **identity, "n_options": n,
                    "input_digest": input_digest(row),
                    "option_keys": [option["key"] for option in row["options"]],
                    "n_chains": args.samples, "valid_chains": len(source_vectors),
                    "missing_chains": args.samples - len(source_vectors),
                    "q": q, "q_transform": args.q_transform,
                    "q_semantics": q_semantics(args.q_transform),
                    "readout": profile["readout"], "provenance": provenance,
                    "per_chain_ids": chain_ids, "per_chain_membership": memberships,
                    "per_chain_probs": sigmoids, "per_chain_centered": centereds,
                    "per_chain_softmax": softmaxes,
                    "per_chain_finish_reasons": [item["finish_reason"] for item in metadata],
                    "per_chain_metadata": metadata,
                    "excluded_chain_metadata": sorted(excluded_by_id[ident], key=lambda item: item["chain"]),
                }
                for field in ("selection_mode", "allow_empty"):
                    if field in row:
                        record[field] = row[field]
                if args.store_logits:
                    record["per_chain_logits"] = chain_logits
                    record["logit_order"] = [args.yes_token, args.no_token] if membership else labels[:n]
                if q is None:
                    record["reason"] = "no_valid_membership_chain" if membership else "no_valid_diagnostic_chain"
                validate_readout_record(record, row=row, expected_identity=identity,
                                        args=args, require_q=False)
                records.append(record)
            out.append(records)
            profile["rows"] += len(records)
            profile["rows_with_q"] += sum(record["q"] is not None for record in records)
            profile["rows_zero_chain"] += sum(record["q"] is None for record in records)
            profile["chains"] += sum(record["valid_chains"] for record in records)
            if args.progress_every and profile["rows"] % args.progress_every < len(records):
                log(f"[{input_path.name} shard {args.shard}] {profile['rows']} rows, "
                    f"{profile['chains']} valid chains")
    profile.update({
        "seconds": round(time.monotonic() - begin, 3),
        "valid_chain_coverage": (profile["chains"] / profile["requested_chains"]
                                 if profile["requested_chains"] else None),
        "status": "incomplete" if profile["rows_zero_chain"] else "ok",
    })
    atomic_write_json(profile_path, profile)
    log(f"[{input_path.name} shard {args.shard}] done: {profile['rows']} rows, "
        f"{profile['chains']} chains, {profile['rows_zero_chain']} without q")
    return profile


def run_readout(args):
    check_model_complete(args.model)
    inputs = [Path(path) for path in args.input]
    outputs = [Path(path) for path in args.output]
    reasoning = [Path(path) for path in args.reasoning]
    if not inputs:
        raise SystemExit("readout requires at least one --input")
    if len(outputs) != len(inputs):
        raise SystemExit("--output must be given once per --input")
    if len(reasoning) != len(inputs):
        raise SystemExit("--reasoning must be given once per --input")
    for path in inputs + reasoning:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")

    pending = []
    for input_path, output_path, reasoning_path in zip(inputs, outputs, reasoning):
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[: args.limit]
        if args.resume:
            sanitize_jsonl(output_path)
        done = (load_done_ids(output_path, expected_identity=teacher_identity(args),
                              input_rows={row["id"]: row for row in rows}, args=args,
                              reasoning_digests={str(reasoning_path): file_sha256(reasoning_path)})
                if args.resume else set())
        shard_rows = [row for index, row in enumerate(rows)
                      if index % args.shards == args.shard]
        pending.append(any(row["id"] not in done for row in shard_rows))
    if not any(pending):
        log(f"[shard {args.shard}] all inputs complete; skipping model load")
        return 0

    engine = ReadoutEngine(str(args.model), device=args.device,
                           gpu_index=0, max_input_tokens=args.max_input_tokens,
                           q_transform=args.q_transform, yes_token=args.yes_token,
                           no_token=args.no_token)
    tokenizer = engine.tokenizer
    labels = engine.labels

    status = 0
    for input_path, reasoning_path, output_path in zip(inputs, reasoning, outputs):
        try:
            profile = readout_split(engine, tokenizer, labels, args, input_path,
                                    [reasoning_path], output_path)
            if profile["rows_zero_chain"]:
                status = 1
            atomic_write_json(output_path.with_name(
                output_path.stem.replace("part-", "profile-") + ".json"), profile)
        except Exception as error:  # noqa: BLE001 - keep the next split
            log(f"[error] {input_path.name}: {error}")
            status = 1
    return status


# ---------------------------------------------------------------------------
# merge / summary
# ---------------------------------------------------------------------------
def _merge_records(parts, key_fn):
    merged = {}
    for part in parts:
        sanitize_jsonl(part)
        for row in read_jsonl(part):
            key = key_fn(row)
            if key in merged:
                raise ValueError(f"{part}: duplicate merged record key {key!r}")
            merged[key] = row
    return merged


def merge_reasoning(args):
    parts_root = Path(args.parts_root)
    output_dir = Path(args.output_dir or DEFAULT_RUN_ROOT / "reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = {}
    for split_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        split = split_dir.name
        parts = sorted(path for path in split_dir.glob("part-*.jsonl")
                       if not path.name.endswith(".invalid.jsonl"))
        merged = _merge_records(parts, lambda row: (row.get("id"), row.get("chain")))
        input_path = Path(args.data_dir) / f"{split}.jsonl"
        if not input_path.exists():
            raise ValueError(f"reasoning merge input not found: {input_path}")
        input_rows = read_jsonl(input_path)
        if args.limit:
            input_rows = input_rows[:args.limit]
        input_by_id = {row["id"]: row for row in input_rows}
        order = [(row["id"], chain) for row in input_rows for chain in range(args.samples)]
        for (ident, chain), record in merged.items():
            if ident not in input_by_id or type(chain) is not int or not 0 <= chain < args.samples:
                raise ValueError(f"{split}: unexpected reasoning id/chain {(ident, chain)!r}")
            verified = _validate_identity(record, teacher_identity(args), row=input_by_id[ident],
                                            allow_legacy=args.allow_legacy_reasoning_cache)
            if verified and record.get("reasoning_sha256") != hashlib.sha256(
                    record.get("reasoning", "").encode("utf-8")).hexdigest():
                raise ValueError(f"{ident}: reasoning cache digest mismatch")
        written = set()
        lines = []
        for key in order:
            if key in merged:
                lines.append(json.dumps(merged[key], ensure_ascii=False,
                                        allow_nan=False))
                written.add(key)
        for key in sorted(merged, key=lambda value: (str(value[0]), value[1])):
            if key not in written:
                lines.append(json.dumps(merged[key], ensure_ascii=False,
                                        allow_nan=False))
        target = output_dir / f"{split}.jsonl"
        atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))
        coverage[split] = {
            "input_rows": len(input_rows), "requested_chains": len(order),
            "output_chains": len(merged), "missing_chains": len(set(order) - set(merged)),
            "truncated_chains": sum(record.get("finish_reason") == "length" for record in merged.values()),
            "finish_reasons": dict(Counter(record.get("finish_reason", "unknown") for record in merged.values())),
        }
        log(f"[merge-reasoning] {len(parts)} parts -> {target} ({len(merged)} chains)")
    atomic_write_json(output_dir / "reasoning_summary.json", {
        **teacher_identity(args), "samples": args.samples, "coverage": coverage})
    return int(not coverage or any(item["missing_chains"] for item in coverage.values()))


def _token_stats(values):
    if not values:
        return {"n": 0, "sum": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {"n": len(values), "sum": sum(values),
            "mean": round(statistics.mean(values), 1),
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            "max": ordered[-1]}


def _summarize(records, input_rows, profiles):
    total = {"rows": 0, "rows_with_chains": 0, "rows_zero_chain": 0,
             "chains": 0, "full_chains": 0}
    q_values = []
    by_source = defaultdict(lambda: {"rows": 0, "chains": 0, "rows_zero_chain": 0})
    zero_ids = []
    finish_reasons = Counter()
    predict_hist = Counter()
    gold_rows = 0
    exact = 0
    jaccard = 0.0
    gold_pos, gold_neg = [], []
    input_by_id = {row["id"]: row for row in input_rows}

    for record in records:
        valid = int(record.get("valid_chains") or 0)
        total["rows"] += 1
        total["chains"] += valid
        if valid:
            total["rows_with_chains"] += 1
            total["full_chains"] += 1 if valid == record.get("n_chains") else 0
        else:
            total["rows_zero_chain"] += 1
            zero_ids.append(record["id"])
        for reason in record.get("per_chain_finish_reasons") or []:
            finish_reasons[str(reason)] += 1
        if record.get("q"):
            q_values.extend(record["q"])
            predict_hist[sum(1 for value in record["q"] if value >= 0.5)] += 1
        source = record.get("source") or "unknown"
        by_source[source]["rows"] += 1
        by_source[source]["chains"] += valid
        by_source[source]["rows_zero_chain"] += 0 if valid else 1

        row = input_by_id.get(record["id"])
        gold = row.get("target") if row else None
        if gold and record.get("q"):
            gold_rows += 1
            keys = [option["key"] for option in row["options"]]
            for value, flag in zip(record["q"], gold):
                (gold_pos if flag else gold_neg).append(value)
            predicted = {key for key, value in zip(keys, record["q"]) if value >= 0.5}
            gold_set = {key for key, flag in zip(keys, gold) if flag}
            exact += 1 if predicted == gold_set else 0
            union = predicted | gold_set
            jaccard += len(predicted & gold_set) / len(union) if union else 1.0

    observed = None
    if profiles:
        seconds = sum(profile.get("gen_seconds", 0.0) + profile.get("seconds", 0.0)
                      for profile in profiles)
        observed = {"profiles": len(profiles), "gpu_seconds": round(seconds, 1),
                    "gpu_hours": round(seconds / 3600, 4)}

    transforms = {record.get("q_transform", "sigmoid") for record in records}
    if len(transforms) > 1:
        raise ValueError("cannot summarize mixed q transforms")
    transform = next(iter(transforms), "membership")
    if transform == "membership":
        method = ("For each chain and candidate, retain the full question/options/reasoning "
                  "and compute q_i = softmax([Yes logit, No logit])[Yes]; average "
                  "complete candidate vectors across valid chains.")
        transform_note = ("The binary events are explicitly candidate membership. "
                          "These are model readouts, without a calibration guarantee.")
    else:
        method = f"Legacy SII option-label {transform} diagnostic, averaged over chains."
        transform_note = ("This transform is not a candidate membership probability and "
                          "must not be used as a membership soft target. For centered_sigmoid, "
                          "q_i = a_i/(1+a_i), where a_i is label softmax over all labels; "
                          "the total is strictly below one.")
    included_metadata = [item for record in records for item in record.get("per_chain_metadata", [])]
    excluded_metadata = [item for record in records for item in record.get("excluded_chain_metadata", [])]
    excluded_reasons = Counter(item.get("exclusion_reason", "unknown") for item in excluded_metadata)
    requested_chains = sum(record.get("n_chains", 0) for record in records)
    identities = {tuple(record.get(key) for key in ("teacher_id", "teacher_view", "checkpoint", "readout"))
                  for record in records}
    if len(identities) > 1:
        raise ValueError("cannot summarize mixed teacher identities or readout versions")
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": method,
        "q_transform": transform, "q_semantics": q_semantics(transform),
        "teacher_components": [dict(zip(("teacher_id", "teacher_view", "checkpoint", "readout"), identity))
                               for identity in sorted(identities, key=str)],
        "totals": total,
        "requested_chains": requested_chains,
        "valid_rate_chains": (round(total["chains"] / requested_chains, 6)
                              if requested_chains else None),
        "excluded_chain_reasons": dict(sorted(excluded_reasons.items())),
        "truncation": {
            "included": sum(bool(item.get("truncated")) for item in included_metadata),
            "excluded": sum(bool(item.get("truncated")) for item in excluded_metadata),
            "policies": sorted({record.get("provenance", {}).get("truncated_policy", "legacy_unspecified")
                                for record in records}),
        },
        "by_source": dict(sorted(by_source.items())),
        "q_value_stats": _token_stats([round(value, 6) for value in q_values]),
        "q_histogram": _histogram(q_values),
        "predicted_set_size_histogram": dict(sorted(predict_hist.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "zero_chain": {"count": len(zero_ids), "ids": zero_ids[:100]},
        "gold_diagnostics": {
            "rows_with_gold": gold_rows,
            "mean_q_gold_positive": (round(statistics.mean(gold_pos), 6)
                                     if gold_pos else None),
            "mean_q_gold_negative": (round(statistics.mean(gold_neg), 6)
                                     if gold_neg else None),
            "exact_set_agreement_at_q_0.5": (round(exact / gold_rows, 6)
                                             if gold_rows else None),
            "mean_jaccard_at_q_0.5": (round(jaccard / gold_rows, 6)
                                      if gold_rows else None),
        },
        "observed": observed,
        "notes": [
            transform_note,
            "Each per_chain_membership vector contains separate candidate readouts. "
            "A product-Bernoulli joint over those marginals is a downstream modeling "
            "assumption, not an empirical joint distribution measured by this teacher.",
            "Truncated chains follow the explicitly recorded include/exclude policy. "
            "Excluded and failed chains remain visible in coverage and metadata.",
            "q is null when no complete valid chain remains; missing candidate "
            "readouts are never replaced with negative targets.",
        ],
    }


def _histogram(values, bins=10):
    if not values:
        return {}
    counts = Counter()
    for value in values:
        index = min(bins - 1, int(max(0.0, min(1.0, value)) * bins))
        counts[format(index / bins, ".1f")] += 1
    return dict(sorted(counts.items()))


def merge_readout(args):
    parts_root = Path(args.parts_root)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir or data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths, all_records, all_inputs, profiles = [], [], [], []
    coverage = {}
    for split_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        split = split_dir.name
        parts = sorted(path for path in split_dir.glob("part-*.jsonl")
                       if not path.name.endswith(".invalid.jsonl"))
        merged = _merge_records(parts, lambda row: row.get("id"))
        input_path = data_dir / f"{split}.jsonl"
        target = output_dir / f"sii_teacher_{split}.jsonl"
        input_rows = read_jsonl(input_path) if input_path.exists() else []
        if args.limit:
            input_rows = input_rows[:args.limit]
        input_by_id = {row["id"]: row for row in input_rows}
        order = list(input_by_id)
        for ident, record in merged.items():
            if ident not in input_by_id:
                raise ValueError(f"{split}: output id {ident!r} absent from input")
            validate_readout_record(record, row=input_by_id[ident],
                                    expected_identity=teacher_identity(args), args=args,
                                    require_q=False)
        written = set()
        lines = []
        for ident in order:
            if ident in merged:
                lines.append(json.dumps(merged[ident], ensure_ascii=False,
                                        allow_nan=False))
                written.add(ident)
        for ident in sorted(set(merged) - written):
            lines.append(json.dumps(merged[ident], ensure_ascii=False, allow_nan=False))
        atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))
        log(f"[merge-readout] {len(parts)} parts -> {target} ({len(merged)} ids)")

        if input_path.exists():
            split_records = read_jsonl(target)
            input_paths.append(input_path)
            all_records.extend(split_records)
            all_inputs.extend(input_rows)
            input_ids = {row["id"] for row in input_rows}
            output_ids = {record["id"] for record in split_records}
            coverage[split] = {
                "input_rows": len(input_rows),
                "output_rows": len(split_records),
                "missing": len(input_ids - output_ids),
                "extra": len(output_ids - input_ids),
                "rows_with_q": sum(record.get("q") is not None for record in split_records),
                "rows_without_q": sum(record.get("q") is None for record in split_records),
            }
            profiles.extend(json.loads(path.read_text())
                            for path in sorted(split_dir.glob("profile-*.json"))
                            if path.exists())
    if not input_paths:
        raise SystemExit(f"no split inputs found under {data_dir}")
    summary = _summarize(all_records, all_inputs, profiles)
    summary["coverage"] = coverage
    atomic_write_json(args.summary, summary)
    log(f"[summary] wrote {args.summary}")
    return int(any(item["missing"] or item["extra"] or item["rows_without_q"]
                   for item in coverage.values()))


# ---------------------------------------------------------------------------
# CPU validation
# ---------------------------------------------------------------------------
def validate_only(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    labels = option_labels(tokenizer)
    membership = args.q_transform == "membership"
    binary_tokens = binary_token_info(tokenizer, args.yes_token, args.no_token) if membership else None
    report = {"model": str(args.model), "option_labels": len(labels),
              **teacher_identity(args), "q_transform": args.q_transform,
              "q_semantics": q_semantics(args.q_transform), "binary_tokens": binary_tokens,
              "model_forward_executed": False,
              "validation_scope": "tokenizer and prompts only; real model forward remains pending",
              "labels_head": labels[:10], "labels_tail": labels[-3:],
              "label_token_ids_head": [tokenizer.encode(label, add_special_tokens=False)[0]
                                       for label in labels[:10]],
              "inputs": {}, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    failed = False
    for path in args.input:
        path = Path(path)
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[: args.limit]
        stage1, stage2, by_source = [], [], defaultdict(list)
        errors = []
        sample_output = None
        for position, row in enumerate(rows):
            try:
                validate_row(row)
            except ValueError as error:
                errors.append({"row": position, "error": str(error)})
                failed = True
                continue
            prefix = make_think_prefix(tokenizer, row, labels, teacher_view=args.teacher_view)
            if membership:
                candidate_prompts = [make_membership_prompt(
                    tokenizer, row, labels, "", candidate, args.yes_token,
                    args.no_token, args.teacher_view) for candidate in range(len(row["options"]))]
                try:
                    for prompt in candidate_prompts:
                        validate_binary_token_boundary(tokenizer, prompt, binary_tokens)
                except ValueError as error:
                    errors.append({"row": position, "error": str(error)})
                    failed = True
                empty_read = candidate_prompts[0]
                sample_with_reasoning = make_membership_prompt(
                    tokenizer, row, labels, "short reasoning", 0,
                    args.yes_token, args.no_token, args.teacher_view)
            else:
                legacy_prefix = make_think_prefix(tokenizer, row, labels)
                reference = make_sii_prompt(tokenizer, row, labels)
                if make_read_prompt(legacy_prefix, "") != reference:
                    errors.append({"row": position, "error": "legacy empty-reasoning prompt "
                                  "differs from the SII no-thinking reference"})
                    failed = True
                empty_read = make_read_prompt(prefix, "")
                candidate_prompts = [empty_read]
                sample_with_reasoning = make_read_prompt(prefix, "short reasoning")
            first_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            read_len = max(len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                           for prompt in candidate_prompts)
            stage1.append(first_len)
            stage2.append(read_len)
            by_source[row["source"]].append(first_len)
            if sample_output is None:
                sample_output = {"id": row["id"], "think_prefix": prefix,
                                 "read_prompt_empty_reasoning": empty_read,
                                 "read_prompt_with_reasoning": sample_with_reasoning}
        report["inputs"][str(path)] = {
            "rows": len(rows), "invalid_rows": errors[:20],
            "invalid_row_count": len(errors),
            "by_source": {source: {"n": len(values), **_token_stats(values)}
                          for source, values in sorted(by_source.items())},
            "think_prefix_tokens": _token_stats(stage1),
            "read_prompt_empty_tokens": _token_stats(stage2),
            "estimated_read_prompt_tokens_max_reasoning": (
                stage2 and max(stage2) + args.max_tokens),
            "prompt_sample": sample_output,
        }
    atomic_write_json(args.report, report)
    log(f"wrote {args.report}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reasoning-chain teacher: candidate Yes/No membership readout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--teacher-id", default=None,
                        help="teacher identity; defaults to checkpoint directory name plus :view")
    parser.add_argument("--teacher-view", choices=tuple(VIEW_INSTRUCTIONS), default="direct",
                        help="distinct reasoning instruction, recorded in every cache row")
    parser.add_argument("--checkpoint-id", default=None,
                        help="immutable checkpoint/revision identifier; otherwise records the local path")
    parser.add_argument("--yes-token", default="Yes",
                        help="explicit single-token affirmative spelling, checked with the tokenizer")
    parser.add_argument("--no-token", default="No",
                        help="explicit single-token negative spelling, checked with the tokenizer")
    parser.add_argument("--input", action="append", default=[],
                        help="data JSONL; repeat for train+dev")
    parser.add_argument("--output", action="append", default=[],
                        help="output JSONL; one per --input")
    parser.add_argument("--reasoning", action="append", default=[],
                        help="phase-A merged reasoning JSONL; one per --input")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2,
                        help="independent thinking chains per question")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="prompts per phase-A generate call")
    parser.add_argument("--readout-batch-size", type=int, default=4,
                        help="prompts per phase-B forward call")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="phase-A thinking cap")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=8192,
                        help="phase-B readout length guard")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--q-transform", choices=("membership", "sigmoid", "centered_sigmoid", "softmax"),
                        default="membership",
                        help="candidate-specific Yes/No membership; old label transforms are diagnostics only")
    parser.add_argument("--truncated-policy", choices=("include", "exclude"), default="exclude",
                        help="include or exclude length-truncated chains and report both counts")
    parser.add_argument("--allow-legacy-reasoning-cache", action="store_true",
                        help="explicitly allow unverified old reasoning only for diagnostic transforms")
    parser.add_argument("--store-logits", action="store_true",
                        help="also write per-chain candidate [Yes, No] logits (label logits for diagnostics)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)

    parser.add_argument("mode", nargs="?", default=None,
                        choices=("generate", "readout", "merge-reasoning",
                                 "merge-readout", "validate-only", "check-model"),
                        help="optional subcommand form of the mode flags")
    parser.add_argument("--generate", action="store_true", help="phase A")
    parser.add_argument("--readout", action="store_true", help="phase B")
    parser.add_argument("--merge-reasoning", action="store_true")
    parser.add_argument("--merge-readout", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-model", action="store_true")

    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=DEFAULT_DATA / "sii_teacher_summary.json")
    parser.add_argument("--parts-root", type=Path, default=DEFAULT_RUN_ROOT / "q/parts")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="merged output directory; defaults to the legacy location for the selected mode")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)
    if args.mode:
        setattr(args, args.mode.replace("-", "_"), True)

    if args.shards < 1 or args.samples < 1 or args.batch_size < 1 \
            or args.readout_batch_size < 1:
        parser.error("--shards, --samples, --batch-size and --readout-batch-size "
                     "must be >= 1")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if min(args.max_tokens, args.max_model_len, args.max_input_tokens) < 1:
        parser.error("token limits must be positive")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be smaller than --max-model-len")
    if args.generate and not args.output:
        parser.error("generate needs --output (one per --input)")
    if args.readout and len(args.reasoning) < len(args.input):
        parser.error("readout needs --reasoning (one per --input)")
    if args.allow_legacy_reasoning_cache and args.q_transform == "membership":
        parser.error("--allow-legacy-reasoning-cache requires a diagnostic --q-transform")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check_model:
        shards = check_model_complete(args.model)
        print(f"model complete: {args.model} ({len(shards)} safetensors shards)")
        return 0
    if args.validate_only:
        if args.report is None:
            args.report = ROOT / "logs/cpu_checks/sii_teacher_validate.json"
        if not args.input:
            raise SystemExit("--validate-only requires --input")
        return validate_only(args)
    if args.merge_reasoning:
        return merge_reasoning(args)
    if args.merge_readout:
        return merge_readout(args)
    if args.generate:
        return run_generate(args)
    if args.readout:
        if not args.input:
            raise SystemExit("readout requires --input")
        return run_readout(args)
    raise SystemExit("choose --generate, --readout, --merge-reasoning, "
                     "--merge-readout or --validate-only")


if __name__ == "__main__":
    sys.exit(main())

