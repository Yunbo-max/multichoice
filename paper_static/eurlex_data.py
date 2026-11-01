#!/usr/bin/env python3
"""Convert official LexGLUE EUR-LEX into full-100-option Qwen decision rows.

Deliberately reads TRAIN and VALIDATION only unless --include-test is given.
Exact official ClassLabel EuroVoc IDs and pinned descriptors are mandatory;
gold never chooses an option, a prompt, or a decoding rule.
The full 100-choice list is never truncated. Only legal-document tokens may
be truncated, with a deterministic head/tail policy verified on the actual
Qwen3.5 chat template / 255-label readout. No Qwen weights are loaded.

Usage:
  python -m paper_static.eurlex_data \
    --output data/eurlex_decision_v1 --tokenizer /path/to/Qwen3.5-4B \
    --pilot-train 5000 --pilot-dev 1000
  # For the FINAL locked evaluation, run separately into a NEW directory
  # with --include-test. NEVER tune on its test.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import urllib.request

DATASET_ID = "coastalcph/lex_glue"
CONFIG = "eurlex"
DESCRIPTORS_COMMIT = "af2da31b54ca4272d880a46aee8dbbf346712e90"
DESCRIPTORS_URL = (
    "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/"
    + DESCRIPTORS_COMMIT + "/data/eurovoc_descriptors.json"
)
EXPECTED = {"train": 55000, "validation": 5000, "test": 5000}
QUESTION = "Select all EuroVoc concepts applicable to the legal document."
TRUNCATION_MARKER = "\n[... middle of legal document omitted ...]\n"
ANSWER_CUE = "The correct answer is ("


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def digest_file(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1 << 20), b""):
            dig.update(part)
    return dig.hexdigest()


def stable_text_key(text):
    return sha_bytes(re.sub(r"\s+", " ", text).strip().encode("utf-8"))


def load_descriptors(path):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("descriptor JSON must be EuroVoc-ID -> nonempty English string")
    return mapping


def label_catalog(class_names, descriptors):
    if not isinstance(class_names, (list, tuple)) or len(class_names) != 100:
        raise ValueError("EUR-LEX must expose EXACTLY 100 official ordered ClassLabel names")
    names = [str(s) for s in class_names]
    if any(not s.isdigit() for s in names) or len(set(names)) != 100:
        raise ValueError("official EuroVoc IDs must be 100 unique numeric names")
    options = []
    for eurovoc_id in names:
        value = descriptors.get(eurovoc_id)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing verified EuroVoc descriptor for {eurovoc_id}")
        options.append({"key": eurovoc_id, "desc": value.strip()})
    return options


def readout_labels(tokenizer):
    all_labels = list(string.ascii_uppercase) + [
        x + y for x in string.ascii_uppercase for y in string.ascii_uppercase
    ]
    labels = [s for s in all_labels if len(tokenizer.encode(s, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise ValueError("tokenizer does not provide 255 legal single-token answer slots")
    token_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in labels]
    if len(set(token_ids)) != 255:
        raise ValueError("answer-slot token IDs are not unique")
    return labels, token_ids


def render_prompt(tokenizer, row, labels):
    # Byte-for-byte match harness/train_multiselect.py make_body/make_prompt.
    choices = "\n".join(
        f"({labels[i]}) {o['key']}: {o['desc']}"
        for i, o in enumerate(row["options"])
    )
    body = (
        f"State:\n{row['state']}\nQuestion: {row['question']}\n"
        f"Options:\n{choices}"
    ).rstrip()
    rule = ("Select all correct options; an empty answer is allowed."
            if row["allow_empty"]
            else "Select all correct options; at least one option is correct.")
    body += "\nDecision rule: " + rule
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) + ANSWER_CUE


def prompt_length(tokenizer, row, labels):
    return len(tokenizer.encode(render_prompt(tokenizer, row, labels),
                                add_special_tokens=False))


def shorten_document(tokenizer, row, labels, max_prompt_tokens, head_ratio=0.75):
    """Return a copy with only document tokens reduced; check ACTUAL prompt."""
    if not (isinstance(max_prompt_tokens, int) and max_prompt_tokens >= 512):
        raise ValueError("max_prompt_tokens must be >=512")
    if not 0 < head_ratio < 1:
        raise ValueError("head_ratio must be strictly between zero and one")
    empty_row = {**row, "state": ""}
    base = prompt_length(tokenizer, empty_row, labels)
    if base + 32 >= max_prompt_tokens:
        raise ValueError(f"100 options + chat template already require {base} tokens; "
                         f"max={max_prompt_tokens} leaves no document budget")
    document = row["state"]
    original_ids = tokenizer.encode(document, add_special_tokens=False)
    original_tokens = len(original_ids)
    if original_tokens + base + 8 < max_prompt_tokens:
        result = {**row}
        actual = prompt_length(tokenizer, result, labels)
        if actual <= max_prompt_tokens:
            return result, {"original_document_tokens": original_tokens,
                            "kept_document_tokens": original_tokens,
                            "prompt_tokens": actual, "truncated": False}
    # The marker is accounted for by the final exact-prompt verification.
    low, high, chosen, length = 0, min(original_tokens, max_prompt_tokens - base), None, None
    while low <= high:
        keep = (low + high) // 2
        left = min(keep, int(math.ceil(keep * head_ratio)))
        right = keep - left
        candidate = (
            tokenizer.decode(original_ids[:left], skip_special_tokens=False)
            + TRUNCATION_MARKER
            + (tokenizer.decode(original_ids[-right:], skip_special_tokens=False) if right else "")
        )
        item = {**row, "state": candidate}
        n = prompt_length(tokenizer, item, labels)
        if n <= max_prompt_tokens:
            chosen, length = (item, (keep, n))
            low = keep + 1
        else:
            high = keep - 1
    if chosen is None or length[0] == 0:
        raise ValueError("token budget left no usable legal document")
    return chosen, {"original_document_tokens": original_tokens,
                    "kept_document_tokens": length[0],
                    "prompt_tokens": length[1], "truncated": True}


def convert_split(records, split, options, *, tokenizer=None, labels=None,
                  max_prompt_tokens=8192):
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown official split")
    if (tokenizer is None) != (labels is None):
        raise ValueError("tokenizer and answer slots must be provided together")
    seen_ids, output, stats = set(), [], []
    for n, source in enumerate(records):
        text = source.get("text")
        target_ids = source.get("labels")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{split}:{n}: missing legal document text")
        if not isinstance(target_ids, (tuple, list)) or any(
            type(i) is not int or not 0 <= i < 100 for i in target_ids
        ) or len(set(target_ids)) != len(target_ids):
            raise ValueError(f"{split}:{n}: invalid ClassLabel indices")
        txt_key = stable_text_key(text)
        if txt_key in seen_ids:
            raise ValueError(f"{split}:{n}: duplicate legal document in the split")
        seen_ids.add(txt_key)
        row = {
            "id": f"eurlex:{split}:{n:06d}",
            "source": "lexglue_eurlex", "lang": "en",
            "split": "dev" if split == "validation" else split,
            "group_id": txt_key,
            "state": text.strip(), "question": QUESTION,
            "options": options, "n_options": 100,
            "selection_mode": "multi", "allow_empty": True,
            "target": [int(i in target_ids) for i in range(100)],
        }
        if tokenizer is not None:
            row, info = shorten_document(tokenizer, row, labels, max_prompt_tokens)
            stats.append(info)
        output.append(row)
    return output, {"n": len(output), "positive_count_histogram": dict(sorted(
        Counter(sum(r["target"]) for r in output).items())),
        "document_fingerprints": sorted(seen_ids),
        "truncation": {
            "n_checked": len(stats),
            "n_truncated": sum(s["truncated"] for s in stats),
            "max_prompt_tokens": max((s["prompt_tokens"] for s in stats), default=None),
            "mean_retained_fraction": (sum(s["kept_document_tokens"] /
                                            max(1, s["original_document_tokens"])
                                            for s in stats) / len(stats)) if stats else None,
        }}


def _choose_pilot(rows, n, seed):
    if not isinstance(n, int) or n < 0:
        raise ValueError("pilot count must be nonnegative integer")
    if n == 0:
        return []
    if n > len(rows):
        raise ValueError("pilot requests more examples than official split contains")
    subset = set(random.Random(seed).sample(range(len(rows)), n))
    return [row for idx, row in enumerate(rows) if idx in subset]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def save_package(directory, records, options, *, provenance, pilot_train=5000,
                 pilot_dev=1000, tokenizer=None, labels=None,
                 max_prompt_tokens=8192):
    """Never overwrite a preexisting output. Atomic-ish one-shot offline conversion."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError("output exists; refusing to replace existing train/val/test")
    required = ("train", "validation")
    if any(k not in records for k in required):
        raise ValueError("requires official train AND validation")
    if any(k not in ("train", "validation", "test") for k in records):
        raise ValueError("unknown split name")
    converted = {}
    audit = {}
    keys = {}
    for part in ("train", "validation", "test"):
        if part not in records:
            continue
        rows, report = convert_split(records[part], part, options, tokenizer=tokenizer,
                                     labels=labels, max_prompt_tokens=max_prompt_tokens)
        converted[part] = rows
        keys[part] = set(report.pop("document_fingerprints"))
        audit[part] = report
    for a in converted:
        for b in converted:
            if a < b and keys[a] & keys[b]:
                raise ValueError(f"cross-split document leakage: {a}/{b}: {len(keys[a] & keys[b])}")
    directory.mkdir(parents=True)
    paths = {}
    for part, rows in converted.items():
        fname = "dev.jsonl" if part == "validation" else part + ".jsonl"
        path = directory / fname
        write_jsonl(path, rows)
        paths[part] = {"path": fname, "sha256": digest_file(path), "n": len(rows)}
    pilot = directory / "pilot"
    pilot.mkdir()
    for name, items in (("train", _choose_pilot(converted["train"], pilot_train, 43)),
                        ("dev", _choose_pilot(converted["validation"], pilot_dev, 44))):
        if items:
            write_jsonl(pilot / f"{name}.jsonl", items)
    manifest = {
        "dataset": DATASET_ID, "config": CONFIG, "provenance": provenance,
        "allowed_counts": "0..100 (no label-informed restrictions)",
        "label_ids": [o["key"] for o in options],
        "ordered_concept_descriptions": options,
        "label_map_sha256": sha_bytes(json.dumps(options, ensure_ascii=False,
                        separators=(",", ":")).encode()),
        "answer_slots": labels[:100] if labels else None,
        "test_loaded": "test" in records,
        "splits": paths, "audit": audit,
        "pilot": {"train": pilot_train, "dev": pilot_dev},
        "max_input_tokens": max_prompt_tokens if tokenizer else None,
        "status": "DATA_PREPARED_NOT_MODEL_TRAINED",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_from_hf(revision, *, include_test=False):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    pinned = revision or HfApi().dataset_info(DATASET_ID).sha
    if not isinstance(pinned, str) or len(pinned) < 12:
        raise ValueError("could not resolve a pinned Hugging Face dataset commit SHA")
    splits = ["train", "validation"] + (["test"] if include_test else [])
    result = {}
    classnames = None
    for split in splits:
        dataset = load_dataset(DATASET_ID, CONFIG, split=split, revision=pinned)
        if len(dataset) != EXPECTED[split]:
            raise ValueError(f"{split}: official sample count changed: {len(dataset)}")
        names = list(dataset.features["labels"].feature.names)
        if classnames is None:
            classnames = names
        elif classnames != names:
            raise ValueError("train/val/test ClassLabel IDs differ")
        result[split] = dataset
    return result, classnames, pinned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--descriptors", type=Path, help="pinned EuroVoc descriptors JSON")
    parser.add_argument("--download-descriptors", action="store_true",
                        help="download descriptor JSON from fixed Git commit")
    parser.add_argument("--hf-revision", type=str,
                        help="exact HF dataset revision; default resolves SHA and saves it")
    parser.add_argument("--include-test", action="store_true",
                        help="explicit one-time held-out test materialization")
    parser.add_argument("--pilot-train", type=int, default=5000)
    parser.add_argument("--pilot-dev", type=int, default=1000)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists")
    if args.download_descriptors and args.descriptors:
        parser.error("choose --descriptors or --download-descriptors")
    if not args.download_descriptors and not args.descriptors:
        parser.error("need --descriptors or --download-descriptors")
    if args.download_descriptors:
        # Pin source to an immutable Git SHA and fingerprint exact downloaded bytes.
        with urllib.request.urlopen(DESCRIPTORS_URL, timeout=120) as stream:
            raw = stream.read()
        descriptor_sha = sha_bytes(raw)
        descriptors = json.loads(raw.decode("utf-8"))
    else:
        descriptors = load_descriptors(args.descriptors)
        descriptor_sha = digest_file(args.descriptors)
    datasets, names, pinned = load_from_hf(args.hf_revision, include_test=args.include_test)
    options = label_catalog(names, descriptors)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    labels, ids = readout_labels(tokenizer)
    manifest = save_package(args.output, datasets, options,
        provenance={"hf_revision": pinned, "descriptor_commit": DESCRIPTORS_COMMIT,
                    "descriptor_sha256": descriptor_sha, "tokenizer": args.tokenizer,
                    "answer_slot_token_ids": ids[:100]},
        pilot_train=args.pilot_train, pilot_dev=args.pilot_dev,
        tokenizer=tokenizer, labels=labels, max_prompt_tokens=args.max_input_tokens)
    print(json.dumps({"status": manifest["status"], "path": str(args.output),
                      "splits": {k: v["n"] for k, v in manifest["splits"].items()},
                      "hf_revision": pinned, "descriptors_sha256": descriptor_sha,
                      "test_loaded": manifest["test_loaded"]}, indent=2))


if __name__ == "__main__":
    main()
