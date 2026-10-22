"""Sharded GPU evaluation harness for SII-Decision on JEVal-style inputs.

This is an independent reimplementation of the documented recipe in
``innerjev/sii_decision`` (prompt rendering + last-slot label logits + softmax).
It does NOT import the downloaded repository at runtime.

One process owns one visible GPU set and one data shard.  A launcher
(``harness/run_sii_eval.sh``) starts one process per GPU, passes disjoint
``--shard/--shards``, then merges the per-shard part files.

Key behaviour
  * exact prompt: chat template with enable_thinking=False + "The correct answer is ("
  * BF16 weights; the released LoRA checkpoints are already merged standalone models
  * legal labels are the original 255 single-token A..Z/AA..ZZ labels, in option order
  * gathers the hidden state at the last non-pad token per row, reads label-token logits
  * dynamic length-bucketed batching with a padded-token budget (long inputs run alone)
  * resume: ids already present in the part file are skipped
  * per-item error capture to ``<output>.errors.jsonl``
  * GPU utilisation/memory sampling via pynvml when installed, else ``nvidia-smi``

Examples
  # validate prompts/token lengths on CPU (no weights)
  python -I sii_gpu_eval.py --validate-only --model ../models/SII-4B-LoRA \
      --input ../data/jeval/sii_input.jsonl

  # one shard on one visible GPU
  CUDA_VISIBLE_DEVICES=3 python -I sii_gpu_eval.py --model ../models/SII-4B-LoRA \
      --input ../data/jeval/sii_input.jsonl --output ../runs/x/parts/part-00003.jsonl \
      --shard 3 --shards 8 --batch-size 8 --max-batch-tokens 32768 --resume \
      --profile ../runs/x/parts/profile-00003.json

  # merge shards
  python -I sii_gpu_eval.py --merge-parts ../runs/x/parts --output ../runs/x/predictions.jsonl

  # merge per-shard profile JSONs
  python -I sii_gpu_eval.py --merge-profiles ../runs/x/parts --output ../runs/x/profile.json
"""
import argparse
import gc
import json
import os
import string
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ANSWER_CUE = "The correct answer is ("


# ---------------------------------------------------------------------------
# prompt rendering (byte-for-byte equivalent to innerjev/sii_decision/prompts.py)
# ---------------------------------------------------------------------------
def text(value):
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
    return (f'State:\n{text(row["state"])}\nQuestion: {text(row["question"])}\n'
            f"Options:\n{choices}").rstrip()


def make_prompt(tokenizer, row, labels, prompt_style="chat"):
    body = make_body(row, labels)
    if prompt_style == "plain":
        # Paper Appendix F.1: plain-text template with no chat template.
        return body + "\n" + ANSWER_CUE
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prefix + ANSWER_CUE


# ---------------------------------------------------------------------------
# input / model helpers
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


def validate_row(row):
    ident = row.get("id")
    if not isinstance(ident, str) or not ident:
        raise ValueError("each row needs a nonempty string id")
    if row.get("type") not in ("choice", "noul", "score"):
        raise ValueError(f"{ident}: unsupported type {row.get('type')!r}")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        raise ValueError(f"{ident}: requires 2..255 options")
    keys = [o.get("key") for o in options]
    if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError(f"{ident}: option keys must be unique nonempty strings")
    if any(not isinstance(o.get("desc"), str) for o in options):
        raise ValueError(f"{ident}: option descriptions must be strings")


def check_model_complete(model_path):
    """Fail fast on a partial download instead of letting from_pretrained stall."""
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        if not list(model_path.glob("*.safetensors")):
            raise FileNotFoundError(f"{model_path}: no safetensors weights found")
        return []
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{model_path}: missing {len(missing)}/{len(shards)} weight shards: "
            + ", ".join(missing))
    return shards


def is_cuda_oom(error):
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(error).lower()


# ---------------------------------------------------------------------------
# GPU sampler
# ---------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """Sample utilization/memory for the physical GPUs backing this process."""

    def __init__(self, physical_indices, interval=0.5):
        super().__init__(daemon=True)
        self.indices = physical_indices
        self.interval = interval
        self.samples = []
        self.source = "none"
        self._stop_event = threading.Event()

    def _sample_pynvml(self, nvml):
        values = []
        for index in self.indices:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            values.append({
                "gpu": index,
                "util_percent": float(util.gpu),
                "memory_used_mib": mem.used / (1024 * 1024),
                "memory_total_mib": mem.total / (1024 * 1024),
            })
        return values

    def _sample_smi(self):
        command = ["nvidia-smi",
                   "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            index = int(parts[0])
            if index not in self.indices:
                continue
            values.append({
                "gpu": index,
                "util_percent": float(parts[1]),
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
            })
        return values or None

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.source = "pynvml"
        except Exception:
            pynvml = None
        while not self._stop_event.is_set():
            try:
                values = self._sample_pynvml(pynvml) if pynvml else self._sample_smi()
                if values:
                    self.samples.append({"t": time.monotonic(), "gpus": values})
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    def summary(self):
        utils = [g["util_percent"] for s in self.samples for g in s["gpus"]]
        mems = [g["memory_used_mib"] for s in self.samples for g in s["gpus"]]
        if not utils:
            return {"source": self.source, "samples": 0}
        utils_sorted = sorted(utils)
        return {
            "source": self.source,
            "samples": len(self.samples),
            "util_mean": sum(utils) / len(utils),
            "util_p50": utils_sorted[len(utils_sorted) // 2],
            "util_p90": utils_sorted[min(len(utils_sorted) - 1, int(0.9 * len(utils_sorted)))],
            "util_max": max(utils),
            "memory_mean_mib": sum(mems) / len(mems),
            "memory_max_mib": max(mems),
        }


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, model_path, device="cuda", gpu_index=0, model_parallel=1,
                 max_input_tokens=100000):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        shards = check_model_complete(model_path)
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
        elif model_parallel > 1:
            torch.cuda.set_device(gpu_index)
            kwargs["device_map"] = "balanced"
            kwargs["max_memory"] = {i: "76GiB" for i in range(model_parallel)}
        else:
            torch.cuda.set_device(gpu_index)
            kwargs["device_map"] = {"": gpu_index}

        model, info = cls.from_pretrained(model_path, output_loading_info=True, **kwargs)
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError("incomplete model loading: " + str({
                "missing_keys": info.get("missing_keys"),
                "mismatched_keys": info.get("mismatched_keys"),
            }))
        if device != "cpu":
            offloaded = [name for name, place in getattr(model, "hf_device_map", {}).items()
                         if place in ("cpu", "disk")]
            if offloaded:
                raise RuntimeError("model was offloaded to CPU/disk; refusing silent offload")
        model.eval()
        self.model = model
        self.backbone = getattr(model.model, "language_model", model.model)
        self.head = model.get_output_embeddings()
        self.input_device = model.get_input_embeddings().weight.device
        self.label_ids = torch.tensor(self.label_ids_cpu, device=self.head.weight.device)
        self.device = device
        self.shards = shards

    def token_lengths(self, prompts):
        return [len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    def predict(self, rows, prompts):
        torch = self.torch
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt",
                                 add_special_tokens=False)
        lengths = encoded["attention_mask"].sum(-1)
        longest = int(lengths.max())
        if longest > self.max_input_tokens:
            raise ValueError(
                f"input length {longest} exceeds --max-input-tokens {self.max_input_tokens}")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.backbone(**encoded, use_cache=False, return_dict=True).last_hidden_state
            end = (lengths - 1).to(hidden.device)
            slot = hidden[torch.arange(len(rows), device=hidden.device), end]
            slot = slot.to(self.head.weight.device)
            logits = self.head(slot).float().index_select(-1, self.label_ids)
        results = []
        for i, row in enumerate(rows):
            n = len(row["options"])
            probs = logits[i, :n].softmax(-1).cpu().tolist()
            index = max(range(n), key=probs.__getitem__)
            keys = [o["key"] for o in row["options"]]
            result = {
                "id": row["id"],
                "model": self.model_name,
                "type": row["type"],
                "precision": "bf16",
                "probs": probs,
                "probabilities": dict(zip(keys, probs)),
                "predicted_index": index,
                "predicted_key": keys[index],
                "input_tokens": int(lengths[i]),
            }
            if row["type"] == "noul":
                yes = next((j for j, key in enumerate(keys) if key in ("yes", "true")), None)
                if yes is not None:
                    result["noul"] = probs[yes]
            if row["type"] == "score":
                result["score"] = sum(j * p for j, p in enumerate(probs))
                if "score_values" in row:
                    if len(row["score_values"]) != n:
                        raise ValueError("score_values must match the ordered options")
                    result["expected_score_value"] = sum(
                        v * p for v, p in zip(row["score_values"], probs))
            results.append(result)
        return results, longest


# ---------------------------------------------------------------------------
# batching / resume
# ---------------------------------------------------------------------------
def make_batches(indices, lengths, batch_size, max_batch_tokens):
    """indices are already length-sorted; cap both row count and padded tokens."""
    current, current_max = [], 0
    for index in indices:
        length = lengths[index]
        new_max = max(current_max, length)
        if current and ((len(current) + 1) * new_max > max_batch_tokens
                        or len(current) >= batch_size):
            yield current
            current, current_max = [], 0
            new_max = length
        current.append(index)
        current_max = new_max
    if current:
        yield current


def load_done_ids(path):
    """Return valid ids, rewriting the file if a crashed writer left a torn line."""
    path = Path(path)
    if not path.exists():
        return set()
    good, malformed = [], 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    if malformed:
        temporary = path.with_suffix(path.suffix + ".repair")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in good:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
        print(f"[resume] dropped {malformed} torn line(s) from {path}", file=sys.stderr)
    return {row["id"] for row in good if isinstance(row.get("id"), str)}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def validate_only(args):
    from transformers import AutoTokenizer
    check_model_complete(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    labels = option_labels(tokenizer)
    rows = read_jsonl(args.input)
    lengths, errors, types = [], [], {}
    for row in rows:
        try:
            validate_row(row)
            prompt = make_prompt(tokenizer, row, labels, args.prompt_style)
            lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
            types[row["type"]] = types.get(row["type"], 0) + 1
        except Exception as error:
            errors.append({"id": row.get("id"), "error": str(error)})
    if not lengths:
        raise SystemExit("no valid rows")
    ordered = sorted(lengths)
    def pct(p):
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    report = {
        "model": args.model,
        "input": args.input,
        "rows": len(rows),
        "valid": len(lengths),
        "types": types,
        "option_labels": len(labels),
        "tokens": {
            "total": sum(lengths),
            "mean": sum(lengths) / len(lengths),
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
            "max": max(lengths),
            "over_6000": sum(1 for n in lengths if n > 6000),
            "over_32768": sum(1 for n in lengths if n > 32768),
            "over_max_input": sum(1 for n in lengths if n > args.max_input_tokens),
        },
        "errors": errors[:20],
        "error_count": len(errors),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


def parse_indices(value):
    return [int(part) for part in value.split(",") if part.strip()]


def evaluate(args):
    import torch

    check_model_complete(args.model)
    rows = read_jsonl(args.input)
    if args.limit and args.limit_mode == "stride":
        step = max(1, len(rows) // args.limit)
        rows = rows[::step][:args.limit]
    elif args.limit:
        rows = rows[:args.limit]
    for row in rows:
        validate_row(row)

    shard_rows = [(i, row) for i, row in enumerate(rows) if i % args.shards == args.shard]
    print(f"[shard {args.shard}/{args.shards}] {len(shard_rows)} of {len(rows)} rows",
          file=sys.stderr, flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = Path(str(output) + ".errors.jsonl")
    if not args.resume:
        output.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)
    done = load_done_ids(output) if args.resume else set()
    if done:
        shard_rows = [(i, row) for i, row in shard_rows if row["id"] not in done]
        print(f"[resume] {len(done)} ids already present; {len(shard_rows)} left",
              file=sys.stderr, flush=True)
    if not shard_rows:
        profile = {
            "model": args.model, "model_name": Path(args.model.rstrip("/")).name,
            "input": args.input, "output": str(output), "shard": args.shard,
            "shards": args.shards, "device": args.device, "device_name": None,
            "model_parallel": args.model_parallel, "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "max_input_tokens": args.max_input_tokens,
            "items": 0, "errors": 0, "skipped_resume": len(done),
            "load_seconds": 0.0, "tokenize_seconds": 0.0, "infer_seconds": 0.0,
            "items_per_s": 0.0, "real_tokens": 0, "padded_tokens": 0,
            "real_tokens_per_s": 0.0, "padded_tokens_per_s": 0.0,
            "gpu": {"source": "skipped"}, "batches": [],
        }
        print(f"[shard {args.shard}] nothing left to do", file=sys.stderr, flush=True)
        if args.profile:
            Path(args.profile).parent.mkdir(parents=True, exist_ok=True)
            Path(args.profile).write_text(
                json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        return 0

    load_begin = time.monotonic()
    engine = Engine(args.model, args.device, args.gpu_index, args.model_parallel,
                    args.max_input_tokens)
    engine.model_name = Path(args.model.rstrip("/")).name
    load_seconds = time.monotonic() - load_begin

    tokenize_begin = time.monotonic()
    prompts = [make_prompt(engine.tokenizer, row, engine.labels, args.prompt_style) for _, row in shard_rows]
    lengths = engine.token_lengths(prompts)
    tokenize_seconds = time.monotonic() - tokenize_begin

    # Filter over-limit items before touching the GPU; they are data errors, not crashes.
    keep, too_long = [], []
    for (position, row), prompt, length in zip(shard_rows, prompts, lengths):
        if length > args.max_input_tokens:
            too_long.append({"id": row["id"], "type": row["type"],
                             "error": f"input_tokens {length} > max_input_tokens "
                                      f"{args.max_input_tokens}"})
        else:
            keep.append((position, row, prompt, length))
    if too_long:
        with errors_path.open("a", encoding="utf-8") as stream:
            for record in too_long:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[shard {args.shard}] skipped {len(too_long)} over-limit rows",
              file=sys.stderr, flush=True)

    if args.no_sort:
        order = list(range(len(keep)))
    else:
        order = sorted(range(len(keep)), key=lambda i: (keep[i][3], keep[i][1]["id"]))

    physical = args.physical_gpu if args.physical_gpu is not None else list(range(args.model_parallel))
    sampler = GpuSampler(physical, args.profile_interval) if args.device != "cpu" else None
    if sampler:
        sampler.start()

    infer_begin = time.monotonic()
    processed = 0
    padded_tokens = 0
    real_tokens = 0
    batches = []
    error_count = len(too_long)

    def record_errors(batch, error):
        nonlocal error_count
        with errors_path.open("a", encoding="utf-8") as stream:
            for i in batch:
                record = {"id": keep[i][1]["id"], "type": keep[i][1]["type"],
                          "error": str(error)}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                error_count += 1

    def run_batch(batch):
        nonlocal processed, padded_tokens, real_tokens
        begin = time.monotonic()
        results, longest = engine.predict([keep[i][1] for i in batch],
                                          [keep[i][2] for i in batch])
        elapsed = time.monotonic() - begin
        with output.open("a", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        processed += len(results)
        padded_tokens += longest * len(results)
        real_tokens += sum(keep[i][3] for i in batch)
        batches.append({
            "items": len(results),
            "max_tokens": longest,
            "real_tokens": sum(keep[i][3] for i in batch),
            "padded_tokens": longest * len(results),
            "seconds": elapsed,
            "items_per_s": len(results) / elapsed if elapsed else 0.0,
        })

    def safe_batch(batch):
        try:
            run_batch(batch)
            return
        except Exception as error:
            if is_cuda_oom(error):
                gc.collect()
                torch.cuda.empty_cache()
                if len(batch) > 1:
                    middle = len(batch) // 2
                    print(f"[oom] split batch {len(batch)} -> {middle}+{len(batch) - middle}",
                          file=sys.stderr, flush=True)
                    safe_batch(batch[:middle])
                    safe_batch(batch[middle:])
                    return
            else:
                # A whole-batch failure may still be row-specific (e.g. tokenizer edge);
                # retry rows individually so one bad item cannot lose the batch.
                if len(batch) > 1:
                    print(f"[error] batch failed ({error}); retrying rows individually",
                          file=sys.stderr, flush=True)
                    for i in batch:
                        safe_batch([i])
                    return
            record_errors(batch, error)

    total = len(order)
    last_report = 0
    for batch in make_batches(order, [keep[i][3] for i in range(len(keep))],
                              args.batch_size, args.max_batch_tokens):
        safe_batch(batch)
        if args.progress_every and processed - last_report >= args.progress_every:
            last_report = processed
            elapsed = time.monotonic() - infer_begin
            print(f"[shard {args.shard}] {processed}/{total} items "
                  f"({processed / elapsed:.2f} items/s, {real_tokens / elapsed:.0f} tok/s)",
                  file=sys.stderr, flush=True)

    infer_seconds = time.monotonic() - infer_begin
    if sampler:
        time.sleep(min(0.2, args.profile_interval))
        sampler.stop()

    profile = {
        "model": args.model,
        "model_name": Path(args.model.rstrip("/")).name,
        "input": args.input,
        "output": str(output),
        "shard": args.shard,
        "shards": args.shards,
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(args.gpu_index)
                        if args.device == "cuda" else "cpu"),
        "model_parallel": args.model_parallel,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_input_tokens": args.max_input_tokens,
        "items": processed,
        "errors": error_count,
        "skipped_resume": len(done),
        "load_seconds": load_seconds,
        "tokenize_seconds": tokenize_seconds,
        "infer_seconds": infer_seconds,
        "items_per_s": processed / infer_seconds if infer_seconds else 0.0,
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "real_tokens_per_s": real_tokens / infer_seconds if infer_seconds else 0.0,
        "padded_tokens_per_s": padded_tokens / infer_seconds if infer_seconds else 0.0,
        "gpu": sampler.summary() if sampler else {"source": "cpu"},
        "batches": batches,
    }
    print(json.dumps({key: profile[key] for key in (
        "items", "errors", "load_seconds", "tokenize_seconds", "infer_seconds",
        "items_per_s", "real_tokens_per_s", "padded_tokens_per_s", "gpu")},
        ensure_ascii=False), flush=True)
    if args.profile:
        Path(args.profile).parent.mkdir(parents=True, exist_ok=True)
        Path(args.profile).write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    return 0 if error_count == 0 else 0


def merge_parts(args):
    parts = sorted(Path(args.merge_parts).glob("part-*.jsonl"))
    if not parts:
        raise SystemExit(f"no part-*.jsonl files under {args.merge_parts}")
    seen = {}
    for part in parts:
        for line in part.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[row["id"]] = row
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for ident in sorted(seen):
            stream.write(json.dumps(seen[ident], ensure_ascii=False, allow_nan=False) + "\n")
    print(f"merged {len(parts)} parts -> {output} ({len(seen)} unique ids)")
    return 0


def merge_profiles(args):
    profiles = sorted(Path(args.merge_profiles).glob("profile-*.json"))
    if not profiles:
        raise SystemExit(f"no profile-*.json files under {args.merge_profiles}")
    runs = [json.loads(path.read_text()) for path in profiles]
    items = sum(run["items"] for run in runs)
    infer_seconds = sum(run["infer_seconds"] for run in runs)
    utils = [run["gpu"].get("util_mean") for run in runs if run["gpu"].get("samples")]
    merged = {
        "shards": len(runs),
        "items": items,
        "errors": sum(run["errors"] for run in runs),
        "aggregate_items_per_s": items / infer_seconds if infer_seconds else 0.0,
        "shard_items_per_s": [run["items_per_s"] for run in runs],
        "shard_infer_seconds": [run["infer_seconds"] for run in runs],
        "shard_load_seconds": [run["load_seconds"] for run in runs],
        "shard_real_tokens_per_s": [run["real_tokens_per_s"] for run in runs],
        "util_mean_of_shards": sum(utils) / len(utils) if utils else None,
        "gpu": [run["gpu"] for run in runs],
        "batches": [batch for run in runs for batch in run["batches"]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: merged[key] for key in (
        "shards", "items", "aggregate_items_per_s", "util_mean_of_shards")},
        ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sharded SII-Decision GPU evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=100000)
    parser.add_argument("--prompt-style", choices=["chat", "plain"], default="chat",
                        help="chat: released prompts.py template; plain: paper Appendix F.1 template")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="visible CUDA index within this process")
    parser.add_argument("--model-parallel", type=int, default=1,
                        help="spread one model over N visible GPUs (device_map=balanced)")
    parser.add_argument("--physical-gpu", default=None,
                        help="comma-separated physical indices for nvidia-smi sampling")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-mode", choices=["head", "stride"], default="head")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-model", action="store_true",
                        help="only verify that every index shard is present")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-interval", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--merge-parts", type=Path)
    parser.add_argument("--merge-profiles", type=Path)
    args = parser.parse_args()

    if args.merge_parts:
        if not args.output:
            parser.error("--merge-parts requires --output")
        return merge_parts(args)
    if args.merge_profiles:
        if not args.output:
            parser.error("--merge-profiles requires --output")
        return merge_profiles(args)
    if args.check_model:
        if not args.model:
            parser.error("--check-model requires --model")
        shards = check_model_complete(args.model)
        print(f"model complete: {args.model} ({len(shards)} safetensors shards)")
        return 0
    if not args.model or not args.input:
        parser.error("--model and --input are required")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.batch_size < 1 or args.max_batch_tokens < 1 or args.max_input_tokens < 1:
        parser.error("batch size and token limits must be positive")
    if args.model_parallel < 1:
        parser.error("--model-parallel must be >= 1")
    if args.physical_gpu is not None:
        args.physical_gpu = parse_indices(args.physical_gpu)
    if args.validate_only:
        return validate_only(args)
    if not args.output:
        parser.error("--output is required for inference")
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
