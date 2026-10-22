#!/usr/bin/env python3
"""SII-style self-teacher: thinking chains + SII answer-slot readout.

This replaces free-form ``Answer: AC`` parsing with the SII decision readout used
by ``innerjev/sii_decision`` (prompts.py / infer.py / learning.py):

  Phase A (``generate``, vLLM)
    body      = ``prompts.body(row, labels)``: State/Question/Options with
                ``(<label i>) <option key>: <description>`` lines.
    prefix    = chat template of the user turn, ``add_generation_prompt=True``,
                ``enable_thinking=True`` (ends with ``<|im_start|>assistant\\n<think>\\n``).
    Sample ``--samples`` chains with vLLM, ``max_tokens`` tokens,
    ``stop=["</think>"]``.  The reasoning is the generated text with chat control
    tokens stripped and, when present, everything after ``</think>`` dropped.
    Truncated reasoning is kept: the readout still works.
    One JSONL row per (id, chain):
      {"id","chain","reasoning","finish_reason","prompt_tokens",
       "completion_tokens","source","lang"}.

  Phase B (``readout``, transformers)
    read_prompt = think_prefix + reasoning + "\\n</think>\\n\\n" + ANSWER_CUE
    where ANSWER_CUE = "The correct answer is (".
    With empty reasoning this is byte-identical to SII's
    ``prompts.prompt(..., enable_thinking=False)`` (the template then emits the
    empty ``<think>\\n\\n</think>\\n\\n`` block).  One forward pass; the hidden
    state at the last non-pad token feeds the output head; logits are gathered
    for SII's 255 single-token option labels and cut to the first K.

    Single-choice SII takes ``softmax(logits[:K])``.  Multi-select targets do not
    form a simplex, so the teacher uses the per-candidate sigmoid
    ``q_chain = sigmoid(logits[:K])`` and averages q over the chains
    (``--q-transform softmax`` switches q to SII's exact readout).  Softmax is
    always recorded per chain for comparison.  No answer text is parsed.

    Calibration warning: on the *base* Qwen3.5-4B head the label logits are all
    positive (CPU smoke: 0.07..24) so the raw sigmoid saturates near 1.0 for
    every option; ``q_value_stats`` and ``per_chain_softmax`` expose this.

    One JSONL row per input id:
      {"id","source","lang","n_options","n_chains","valid_chains","q",
       "q_transform","per_chain_probs","per_chain_softmax",
       "per_chain_finish_reasons"}

Modes
  generate           phase A, one process per visible GPU and data shard
  readout            phase B, one process per visible GPU and data shard
  merge-reasoning    merge phase-A parts per split into <split>.jsonl
  merge-readout      merge phase-B parts, write sii_teacher_<split>.jsonl + summary
  --validate-only    CPU: render/compare prompts, check labels, report lengths
  --check-model      verify all safetensors shards are present

Examples
  python -I sii_style_teacher.py --validate-only --model ../models/Qwen3.5-4B \
      --input ../data/multiselect/dev.jsonl --report ../logs/cpu_checks/sii_teacher_validate.json

  CUDA_VISIBLE_DEVICES=3 python -I sii_style_teacher.py generate \
      --model ../models/Qwen3.5-4B --input ../data/multiselect/train.jsonl \
      --output ../runs/qz/sii_style_teacher/reasoning/parts/train/part-00003.jsonl \
      --shard 3 --shards 8 --samples 2 --max-tokens 1024 --max-model-len 8192 --resume

  CUDA_VISIBLE_DEVICES=3 ../.venv-sii/bin/python -I sii_style_teacher.py readout \
      --model ../models/Qwen3.5-4B --input ../data/multiselect/train.jsonl \
      --reasoning ../runs/qz/sii_style_teacher/reasoning/train.jsonl \
      --output ../runs/qz/sii_style_teacher/q/parts/train/part-00003.jsonl \
      --shard 3 --shards 8 --samples 2 --batch-size 4 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/Qwen3.5-4B"
DEFAULT_DATA = ROOT / "data/multiselect"
DEFAULT_RUN_ROOT = ROOT / "runs/qz/sii_style_teacher"
ANSWER_CUE = "The correct answer is ("
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


def make_think_prefix(tokenizer, row, labels):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": make_body(row, labels)}],
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


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    for row in read_jsonl(path):
        if isinstance(row.get("id"), str):
            done.add(row["id"])
    return done


def load_done_chains(path):
    done = set()
    if not Path(path).exists():
        return done
    for row in read_jsonl(path):
        if isinstance(row.get("id"), str) and isinstance(row.get("chain"), int):
            done.add((row["id"], row["chain"]))
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
        return llm.generate(prompts, sampling, use_tqdm=False,
                            tokenization_kwargs={"add_special_tokens": False})
    except Exception as error:  # noqa: BLE001 - vLLM raises many engine types
        if len(prompts) > 1:
            middle = len(prompts) // 2
            log(f"[warn] batch of {len(prompts)} failed ({error}); splitting")
            return (_generate_with_retry(llm, prompts[:middle], sampling)
                    + _generate_with_retry(llm, prompts[middle:], sampling))
        log(f"[error] prompt failed: {error}")
        return [None]


def chain_record(row, chain, reasoning, finish_reason, prompt_tokens,
                 completion_tokens):
    return {
        "id": row["id"],
        "chain": chain,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "source": row.get("source"),
        "lang": row.get("lang"),
    }


def generate_split(llm, sampling, tokenizer, labels, args, input_path, output_path):
    rows = read_jsonl(input_path)
    for row in rows:
        validate_row(row)
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
        done = load_done_chains(output_path)
    if done:
        selected = [(index, row) for index, row in selected
                    if any((row["id"], chain) not in done
                           for chain in range(args.samples))]
        log(f"[{input_path.name} shard {args.shard}] resume: {len(done)} chains done, "
            f"{len(selected)} rows left")

    profile = {
        "phase": "reasoning",
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

    prompts = [make_think_prefix(tokenizer, row, labels) for _, row in selected]
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
            out.append([chain_record(row, chain, "", "prompt_too_long", length, 0)
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
                        records.append(chain_record(row, chain, "", "error", length, 0))
                    profile["errors"] += len(needed)
                    chains_written += len(needed)
                    continue
                completions = list(getattr(request_output, "outputs", None) or [])
                for chain in needed:
                    if chain >= len(completions):
                        records.append(chain_record(row, chain, "", "missing", length, 0))
                        profile["errors"] += 1
                        chains_written += 1
                        continue
                    completion = completions[chain]
                    finish = completion.finish_reason or "unknown"
                    token_ids = getattr(completion, "token_ids", None)
                    tokens = len(token_ids) if token_ids is not None else 0
                    records.append(chain_record(
                        row, chain, extract_reasoning(completion.text or ""),
                        finish, length, tokens))
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
        done = load_done_chains(output_path) if args.resume else set()
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
    """SII readout: last-slot hidden state -> label-token logits."""

    def __init__(self, model_path, device="cuda", gpu_index=0, max_input_tokens=8192):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"
        self.labels = option_labels(self.tokenizer)
        self.label_ids_cpu = [self.tokenizer.encode(label, add_special_tokens=False)[0]
                              for label in self.labels]

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
        self.label_ids = torch.tensor(self.label_ids_cpu, device=self.head.weight.device)
        self.device = device

    def logits(self, prompts):
        """Return a CPU float tensor [B, 255] of label-token logits."""
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
            logits = self.head(slot).float().index_select(-1, self.label_ids)
        return logits.cpu()


def chain_probabilities(logits_row, n_options):
    """(sigmoid, centered_sigmoid, softmax) of the first K label logits.

    centered_sigmoid subtracts the log-sum-exp over all 255 label tokens before
    the per-candidate sigmoid. Raw sigmoid saturates on an untrained base head
    (all label logits are high); centering keeps per-candidate independence
    without forcing the K options onto a simplex.
    """
    import torch
    full = logits_row.float()
    values = full[:n_options]
    lse = torch.logsumexp(full, -1)
    return (torch.sigmoid(values).tolist(),
            torch.sigmoid(values - lse).tolist(),
            torch.softmax(values, -1).tolist())


def _readout_with_retry(engine, prompts):
    """Split a failed forward batch down to single prompts; None on failure."""
    try:
        return engine.logits(prompts)
    except Exception as error:  # noqa: BLE001 - keep the shard alive
        if len(prompts) > 1:
            import torch
            middle = len(prompts) // 2
            log(f"[warn] readout batch of {len(prompts)} failed ({error}); splitting")
            left = _readout_with_retry(engine, prompts[:middle])
            right = _readout_with_retry(engine, prompts[middle:])
            if left is None or right is None:
                return None
            return torch.cat([left, right], dim=0)
        log(f"[error] readout prompt failed: {error}")
        return None


def load_reasoning_by_id(paths, wanted_ids):
    grouped = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            ident = row.get("id")
            if ident in wanted_ids and isinstance(row.get("chain"), int):
                grouped[ident].append(row)
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry["chain"])
    return grouped


def readout_split(engine, tokenizer, labels, args, input_path, reasoning_paths,
                  output_path):
    rows = read_jsonl(input_path)
    for row in rows:
        validate_row(row)
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
        done = load_done_ids(output_path)
    selected = [(index, row) for index, row in selected if row["id"] not in done]
    reasoning = load_reasoning_by_id(reasoning_paths, {row["id"] for _, row in selected})

    profile = {
        "phase": "readout",
        "input": str(input_path), "reasoning": [str(p) for p in reasoning_paths],
        "output": str(output_path), "model": str(args.model),
        "shard": args.shard, "shards": args.shards, "samples": args.samples,
        "rows": 0, "chains": 0, "rows_zero_chain": 0,
        "skipped_unusable": 0,
        "seconds": 0.0, "status": "empty",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not selected:
        atomic_write_json(profile_path, profile)
        log(f"[{input_path.name} shard {args.shard}] nothing left to do")
        return profile

    begin = time.monotonic()
    with JsonlAppender(output_path, resume=args.resume and output_path.exists()) as out:
        for offset in range(0, len(selected), args.readout_batch_size):
            batch = selected[offset: offset + args.readout_batch_size]
            flat = []  # (row, entry)
            prompts = []
            prefix_cache = {}
            for _index, row in batch:
                available = reasoning.get(row["id"], [])
                entries = [entry for entry in available
                           if entry.get("finish_reason") not in UNUSABLE_FINISH_REASONS
                           and (row["id"], entry["chain"]) not in done]
                profile["skipped_unusable"] += len(available) - len(entries)
                entries = entries[: args.samples]
                if not entries:
                    out.append([{
                        "id": row["id"], "source": row.get("source"),
                        "lang": row.get("lang"), "n_options": len(row["options"]),
                        "n_chains": args.samples, "valid_chains": 0, "q": None,
                        "per_chain_probs": [], "per_chain_softmax": [],
                        "per_chain_finish_reasons": [],
                        "reason": "no_reasoning",
                    }])
                    profile["rows"] += 1
                    profile["rows_zero_chain"] += 1
                    continue
                prefix = prefix_cache.get(row["id"])
                if prefix is None:
                    prefix = make_think_prefix(tokenizer, row, labels)
                    prefix_cache[row["id"]] = prefix
                for entry in entries:
                    prompts.append(make_read_prompt(
                        prefix, entry.get("reasoning") or ""))
                    flat.append((row, entry))

            if not prompts:
                continue
            logits = _readout_with_retry(engine, prompts)
            if logits is None:
                for row, entry in flat:
                    out.append([{
                        "id": row["id"], "source": row.get("source"),
                        "lang": row.get("lang"), "n_options": len(row["options"]),
                        "n_chains": args.samples, "valid_chains": 0, "q": None,
                        "per_chain_probs": [], "per_chain_softmax": [],
                        "per_chain_finish_reasons": [],
                        "reason": "readout_error",
                    }])
                    profile["rows"] += 1
                    profile["rows_zero_chain"] += 1
                continue

            # Regroup the flat results by input id.
            by_id = defaultdict(list)
            for position, (row, entry) in enumerate(flat):
                by_id[row["id"]].append((position, row, entry))
            records = []
            for _index, row in batch:
                items = by_id.get(row["id"])
                if not items:
                    continue
                n = len(row["options"])
                sigmoids, centereds, softmaxes, finishes, logits_out = [], [], [], [], []
                for position, _row, entry in items:
                    sig, centered, soft = chain_probabilities(logits[position], n)
                    sigmoids.append([round(value, 8) for value in sig])
                    centereds.append([round(value, 8) for value in centered])
                    softmaxes.append([round(value, 8) for value in soft])
                    finishes.append(entry.get("finish_reason"))
                    logits_out.append([round(value, 6) for value in
                                       logits[position][:n].tolist()])
                if args.q_transform == "softmax":
                    source_probs = softmaxes
                elif args.q_transform == "centered_sigmoid":
                    source_probs = centereds
                else:
                    source_probs = sigmoids
                q = [round(sum(values) / len(values), 8)
                     for values in zip(*source_probs)] if source_probs else None
                record = {
                    "id": row["id"], "source": row.get("source"),
                    "lang": row.get("lang"), "n_options": n,
                    "n_chains": args.samples, "valid_chains": len(sigmoids),
                    "q": q, "q_transform": args.q_transform,
                    "per_chain_probs": sigmoids,
                    "per_chain_centered": centereds,
                    "per_chain_softmax": softmaxes,
                    "per_chain_finish_reasons": finishes,
                }
                if args.store_logits:
                    record["per_chain_logits"] = logits_out
                if len(sigmoids) < args.samples:
                    record["missing_chains"] = args.samples - len(sigmoids)
                records.append(record)
            out.append(records)
            profile["rows"] += len(records)
            profile["chains"] += sum(record["valid_chains"] for record in records)
            if args.progress_every and profile["rows"] % args.progress_every < len(records):
                elapsed = time.monotonic() - begin
                log(f"[{input_path.name} shard {args.shard}] {profile['rows']} rows, "
                    f"{profile['chains']} chains read in {elapsed:.0f}s")

    seconds = time.monotonic() - begin
    profile.update({"seconds": round(seconds, 3), "status": "ok"})
    atomic_write_json(profile_path, profile)
    log(f"[{input_path.name} shard {args.shard}] done: {profile['rows']} rows, "
        f"{profile['chains']} chains, {profile['rows_zero_chain']} zero-chain")
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

    engine = ReadoutEngine(str(args.model), device=args.device,
                           gpu_index=0, max_input_tokens=args.max_input_tokens)
    tokenizer = engine.tokenizer
    labels = engine.labels

    pending = []
    for input_path, output_path in zip(inputs, outputs):
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[: args.limit]
        if args.resume:
            sanitize_jsonl(output_path)
        done = load_done_ids(output_path) if args.resume else set()
        shard_rows = [row for index, row in enumerate(rows)
                      if index % args.shards == args.shard]
        pending.append(any(row["id"] not in done for row in shard_rows))
    if not any(pending):
        log(f"[shard {args.shard}] all inputs complete; skipping model load")
        return 0

    status = 0
    for input_path, reasoning_path, output_path in zip(inputs, reasoning, outputs):
        try:
            profile = readout_split(engine, tokenizer, labels, args, input_path,
                                    [reasoning_path], output_path)
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
            merged[key_fn(row)] = row
    return merged


def merge_reasoning(args):
    parts_root = Path(args.parts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        split = split_dir.name
        parts = sorted(path for path in split_dir.glob("part-*.jsonl")
                       if not path.name.endswith(".invalid.jsonl"))
        merged = _merge_records(parts, lambda row: (row.get("id"), row.get("chain")))
        input_path = Path(args.data_dir) / f"{split}.jsonl"
        order = []
        if input_path.exists():
            order = [(row["id"], chain) for row in read_jsonl(input_path)
                     for chain in range(args.samples)]
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
        log(f"[merge-reasoning] {len(parts)} parts -> {target} ({len(merged)} chains)")
    return 0


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

    transform = (records[0].get("q_transform", "sigmoid") if records else "sigmoid")
    if transform == "softmax":
        method = ("SII answer-slot readout over thinking chains; q_chain = "
                  "softmax(label logits[:K]) (SII's exact readout), averaged over "
                  "chains.")
        transform_note = ("q uses SII's exact softmax over the first K label "
                          "logits; per_chain_probs still records the independent "
                          "sigmoid alternative.")
    elif transform == "centered_sigmoid":
        method = ("SII answer-slot readout over thinking chains; q_chain = "
                  "sigmoid(label logits[:K] - logsumexp over all 255 labels), "
                  "averaged over chains.")
        transform_note = ("Raw sigmoid saturates on an untrained base head; "
                          "centering by the 255-label logsumexp keeps independent "
                          "per-candidate probabilities without forcing a simplex. "
                          "per_chain_probs/per_chain_softmax are recorded.")
    else:
        method = ("SII answer-slot readout over thinking chains; q_chain = "
                  "sigmoid(label logits[:K]) (multi-select adaptation of SII's "
                  "single-choice softmax), averaged over chains.")
        transform_note = ("SII single-choice uses softmax over the first K label "
                          "logits; this multi-select teacher uses the "
                          "per-candidate sigmoid and averages it over chains. "
                          "per_chain_softmax is recorded for comparison, and "
                          "--q-transform softmax switches q to SII's exact readout.")
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": method,
        "q_transform": transform,
        "totals": total,
        "valid_rate_chains": (round(total["chains"] / (total["rows"]
                                                       * (records[0].get("n_chains") or 1)
                                                       ), 6)
                              if records and total["rows"] else None),
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
            "Raw sigmoid on a base-model output head can saturate when every "
            "label logit is positive (CPU smoke on Qwen3.5-4B: logits 0.07..24, "
            "sigmoid ~1.0 for all options); check q_value_stats and "
            "per_chain_softmax, and use --q-transform softmax if it does.",
            "Reasoning truncated at max_tokens is still read out; no answer text is "
            "parsed anywhere in this pipeline.",
            "q is null only when a row has no reasoning chain at all.",
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
    data_dir.mkdir(parents=True, exist_ok=True)
    input_paths, all_records, all_inputs, profiles = [], [], [], []
    coverage = {}
    for split_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        split = split_dir.name
        parts = sorted(path for path in split_dir.glob("part-*.jsonl")
                       if not path.name.endswith(".invalid.jsonl"))
        merged = _merge_records(parts, lambda row: row.get("id"))
        input_path = data_dir / f"{split}.jsonl"
        target = data_dir / f"sii_teacher_{split}.jsonl"
        order = [row["id"] for row in read_jsonl(input_path)] if input_path.exists() else []
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
            input_rows = read_jsonl(input_path)
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
    return 0


# ---------------------------------------------------------------------------
# CPU validation
# ---------------------------------------------------------------------------
def validate_only(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    labels = option_labels(tokenizer)
    report = {"model": str(args.model), "option_labels": len(labels),
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
            prefix = make_think_prefix(tokenizer, row, labels)
            reference = make_sii_prompt(tokenizer, row, labels)
            empty_read = make_read_prompt(prefix, "")
            if empty_read != reference:
                errors.append({"row": position, "error": "empty-reasoning read prompt "
                              "differs from prompts.prompt(enable_thinking=False)"})
                failed = True
            first_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            read_len = len(tokenizer(empty_read, add_special_tokens=False)["input_ids"])
            stage1.append(first_len)
            stage2.append(read_len)
            by_source[row["source"]].append(first_len)
            if sample_output is None:
                sample_output = {"id": row["id"], "think_prefix": prefix,
                                 "read_prompt_empty_reasoning": empty_read,
                                 "read_prompt_with_reasoning": make_read_prompt(
                                     prefix, "short reasoning")}
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
        description="SII-style self-teacher: thinking chains + answer-slot readout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
    parser.add_argument("--q-transform", choices=("sigmoid", "centered_sigmoid", "softmax"),
                        default="centered_sigmoid",
                        help="multi-select q: centered per-option sigmoid (default), "
                             "raw sigmoid (saturates on base heads), or SII's exact softmax")
    parser.add_argument("--store-logits", action="store_true",
                        help="also write per-chain label logits")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_ROOT / "reasoning")
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
