#!/usr/bin/env python3
"""Self-teacher soft-target generator for the multi-select decision experiment.

The teacher is the *same* base model as the student (``models/Qwen3.5-4B``) run with
thinking enabled.  For every question it samples ``--samples`` independent reasoning
chains, parses each chain's final answer set, and writes the per-candidate soft target

    q_i = (# valid chains that mark candidate i correct) / (# valid chains)

For the default two chains ``q_i`` is in ``{0, 0.5, 1}``.  Invalid / truncated chains
are excluded from the denominator and logged; a row with zero valid chains is written
with ``q = null`` so the training harness can drop it.

Input rows (``data/multiselect/{train,dev}.jsonl``):
  {"id","source","lang","split","state","question",
   "options":[{"key","desc"}],"target":[0/1,...],"n_options"}

Output rows (one per input id, in input order after merging):
  {"id","source","lang","n_options","n_chains","valid_chains","sets","q",
   "finish_reasons","prompt_tokens","completion_tokens"}

  sets            only the *valid* chains, each a list of option keys in option order
  q               length == n_options, or null when valid_chains == 0
  finish_reasons  one entry per sampled chain (including invalid ones)
  completion_tokens  one entry per sampled chain (including invalid ones)

Two-stage mode (``--two-stage``)
  vLLM 0.19.1 refuses ``thinking_token_budget`` unless a reasoning_config is set,
  and a single thinking pass capped at N tokens frequently never emits
  ``</think>``, so no answer is produced (61% zero-valid rows on the 4096-token
  pilot).  Two-stage mode splits the work:

  stage 1  the thinking prompt (``enable_thinking=True``) is sampled with
           ``--samples`` chains, ``max_tokens=--thinking-max-tokens`` (default
           1024) and ``stop=["</think>"]``.  finish_reason is ``stop`` when the
           model closed its thinking block and ``length`` when it hit the cap.
           The reasoning text is kept in both cases (leading ``<think>`` and chat
           control tokens stripped; text after ``</think>`` dropped).
  stage 2  the original State/Question/Options body plus that reasoning is
           rendered with ``enable_thinking=False`` and the model is asked for a
           single final line ``Answer: AC`` (greedy, ``--answer-max-tokens``,
           default 64).  Only this answer is parsed, so a truncated stage-1
           chain still yields a usable target.  If the stage-2 prompt would
           exceed the context budget the reasoning is truncated (tail kept); if
           even an empty reasoning does not fit the chain is invalid.

  Two-stage records add ``stage1_finish_reasons`` and ``stage2_finish_reasons``.
  ``finish_reasons`` mirrors stage 2 because it decides validity,
  ``prompt_tokens`` sums both prompts and ``completion_tokens`` sums both
  generations for the chain.

Prompt (chat template, ``enable_thinking=True``)::

    State:
    {state}
    Question: {question}
    Options:
    (A) A: desc
    (B) B: desc
    ...
    <instruction to reason and end with "Answer: AC">

The letter label and the option key coincide for this dataset (A, B, C, ...); the
renderer follows the project's existing SII prompt convention ``(label) key: desc``.

Modes
  generate (default)   one process owns one visible GPU and one data shard
  --validate-only      CPU: validate rows, render prompts, report token lengths
  --parser-selftest    CPU: exercise the answer parser on tricky outputs
  --estimate           CPU: project prompt tokens, GPU-hours and USD cost
  --merge-all          merge per-shard parts for every split, then write the summary
  --check-model        verify that every safetensors index shard is present

Examples
  # CPU validation (no GPU, no vLLM import)
  python -I generate_self_teacher.py --validate-only \
      --input ../data/multiselect/train.jsonl --input ../data/multiselect/dev.jsonl \
      --model ../models/Qwen3.5-4B --report ../logs/cpu_checks/ms_teacher_validate.json

  # one shard on one visible GPU (the launcher starts 8 of these)
  CUDA_VISIBLE_DEVICES=3 python -I generate_self_teacher.py \
      --model ../models/Qwen3.5-4B \
      --input ../data/multiselect/train.jsonl \
      --output ../runs/qz/ms_self_teacher/parts/train/part-00003.jsonl \
      --shard 3 --shards 8 --samples 2 --max-tokens 2048 --max-model-len 8192 \
      --resume

  # two-stage shard: 1024-token thinking pass, then a forced 64-token answer
  CUDA_VISIBLE_DEVICES=3 python -I generate_self_teacher.py \
      --two-stage --thinking-max-tokens 1024 --answer-max-tokens 64 \
      --model ../models/Qwen3.5-4B \
      --input ../data/multiselect/train.jsonl \
      --output ../runs/qz/ms_self_teacher/parts/train/part-00003.jsonl \
      --shard 3 --shards 8 --samples 2 --max-model-len 8192 --resume

  # merge all splits and write the summary
  python -I generate_self_teacher.py --merge-all \
      --parts-root ../runs/qz/ms_self_teacher/parts \
      --data-dir ../data/multiselect --invalid-dir ../logs/qz \
      --summary ../data/multiselect/self_teacher_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/Qwen3.5-4B"
DEFAULT_DATA = ROOT / "data/multiselect"
DEFAULT_PARTS = ROOT / "runs/qz/ms_self_teacher/parts"
DEFAULT_SUMMARY = DEFAULT_DATA / "self_teacher_summary.json"
DEFAULT_INVALID_DIR = ROOT / "logs/qz"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Never overwrite the curated dataset files.
PROTECTED_PATHS = {
    (DEFAULT_DATA / name).resolve()
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "summary.json")
}

ZH_INSTRUCTION = (
    "请先简要推理（不超过 200 字），然后最后一行只输出所有正确选项的字母，格式为：Answer: AC"
    "（只写字母，不要输出其他内容）。"
)
EN_INSTRUCTION = (
    "Please reason briefly (at most 200 words), then output the letters of all correct options on the "
    "final line exactly as: Answer: AC (letters only, no other text)."
)

# ---------------------------------------------------------------------------
# answer parsing
# ---------------------------------------------------------------------------
# Markers are tried from the last occurrence backwards.  English markers require a word
# boundary (so "answered" does not match); Chinese alternatives are longest-first.
ANSWER_RE = re.compile(
    r"(最终答案|参考答案|正确答案|答案|故选|应选|正确(?:的)?选项(?:是|为|包括)?"
    r"|correct\s+options?\s+(?:are|is)\b"
    r"|(?:(?:final|correct)\s+)?answers?\b)",
    re.IGNORECASE,
)

# Words that are prose, never option letters.  Single-letter words are deliberately
# absent: "A" must stay a candidate.  All-caps prose words are handled separately so
# that a genuine answer such as "BE" or "IS" is not swallowed by the lowercase word.
STOPWORDS = {
    "the", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "if", "then",
    "than", "that", "this", "these", "those", "it", "its", "all", "any", "both",
    "each", "either", "every", "none", "no", "not", "only", "also", "so", "such",
    "option", "options", "answer", "answers", "correct", "incorrect", "true",
    "false", "above", "below", "following", "select", "choose", "chosen", "letter",
    "letters", "choice", "choices", "statement", "statements", "question", "yes",
    "ok", "okay", "include", "includes", "included", "according", "based",
    "because", "since", "therefore", "thus", "hence", "which", "who", "whose",
    "where", "when", "why", "how", "can", "could", "should", "would", "may",
    "might", "must", "will", "shall", "do", "does", "did", "have", "has", "had",
    "we", "you", "they", "he", "she", "me", "my", "our", "your", "their", "them",
    "us", "item", "items", "result", "results", "but", "however", "although",
    "though", "while", "yet",
}
# All-caps prose words are handled separately from the lowercase stopword list so a
# genuine answer such as "BE" (options B and E) is still parsed as two letters.
CAPS_STOPWORDS = {
    "THE", "ALL", "NONE", "NO", "YES", "NOT", "OPTIONS", "OPTION", "ANSWER",
    "ANSWERS", "CORRECT", "INCORRECT", "TRUE", "FALSE", "BOTH", "ANY", "EACH",
    "ABOVE", "BELOW", "SELECT", "CHOOSE", "LETTER", "LETTERS", "OK", "OKAY",
    "ACCORDING", "BASED", "BECAUSE", "THEREFORE", "HOWEVER", "ALTHOUGH", "THOUGH",
}
CONNECTORS = {"and", "or", "plus", "also"}

MARKER_SEGMENT_MAX = 300
FALLBACK_LINE_MAX = 120
PUNCT_STRIP = " \t\r\n.,;:!?，。；：！？、|/\\-—–_*`~'\"“”‘’()[]{}<>《》【】"


def log(message):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def _clean_output(text):
    """Remove chat-template control tokens but keep <think> / </think> markers."""
    if not isinstance(text, str):
        return ""
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "\n")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _first_nonempty_line(after, max_lines=2):
    for line in after.split("\n")[:max_lines]:
        line = line.strip()
        if line:
            return line
    return ""


def _parse_segment(segment, valid_order, strict=False):
    """Parse one answer-like string into an ordered list of option keys.

    ``strict`` is used for the marker-less last-line fallback: once letter parsing has
    started, any prose aborts the parse instead of silently accepting a prefix.
    """
    valid_set = set(valid_order)
    text = unicodedata.normalize("NFKC", segment)

    # LaTeX wrappers: \boxed{AC}, \text{A, C}, ...
    match = re.search(r"\\(?:boxed|text|mathrm|mathbf)\s*\{([^{}]*)\}", text)
    if match:
        text = match.group(1)

    text = re.sub(r"[*_`~]+", " ", text)
    # Drop punctuation left by the marker (":", "：", "**", ...) before the lead-in
    # regexes run; "Answer: I choose A and C" must still match the "I choose" rule.
    text = text.strip(PUNCT_STRIP)
    if not text:
        return None

    # Remove explicitly negated options before collecting letters.  Without this,
    # "A, C. B and D are incorrect" would also pick up B and D.
    left = r"(?<![A-Za-z])"
    right = r"(?![A-Za-z])"
    text = re.sub(
        left + r"([A-Z](?:\s*(?:,|and|or|&)\s*[A-Z])*)\s+(?:is|are)\s+"
               r"(?:incorrect|wrong|false|excluded|invalid)" + right,
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        left + r"([A-Z](?:\s*(?:,|and|or|&)\s*[A-Z])*)\s+(?:is|are)"
               r"(?:n't|\s+not)\s+(?:a\s+)?correct" + right,
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        left + r"(?:not|except|excluding|exclude)\s+([A-Z])" + right,
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        left + r"(?:except|excluding|exclude)\s+"
               r"([A-Z](?:\s*(?:,|and|or|&)\s*[A-Z])*)" + right,
        " ", text, flags=re.IGNORECASE)
    # Chinese writes negations compactly: "A错误C正确", "排除A、C".  Drop the trailing
    # letter lookahead here so "A错误C正确" can remove A even though C follows directly.
    text = re.sub(
        left + r"([A-Z](?:\s*[、,，和与及]\s*[A-Z])*)\s*"
               r"(?:(?:是|为)?\s*(?:错误|不正确|不对|排除|不选)"
               r"|(?:不是|并非|非)\s*(?:正确|对))",
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        left + r"(?:错误|不正确|不对|排除|不选)\s*"
               r"([A-Z](?:\s*[、,，和与及]\s*[A-Z])*)",
        " ", text, flags=re.IGNORECASE)

    # Leading conversational wrapper: "I choose A and C", "I think ...".
    text = re.sub(
        r"^\s*i(?:'d|'ll|\s+would)?\s+"
        r"(?:choose|select|pick|think|believe|go\s+with)\s+",
        " ", text, flags=re.IGNORECASE)

    # Leading prose such as "the correct options are" / "正确答案是".
    text = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+|correct\s+)?(?:answers?|options?)\s*(?:is|are)\s*",
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:the\s+)?correct\s+options?\s*(?:are|is)\s*",
        " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:最终答案|参考答案|正确答案|答案|正确选项|应选|故选)\s*(?:是|为|包括)?\s*",
        " ", text)
    text = re.sub(r"^\s*(?:is|are|为|是)\s*", " ", text, flags=re.IGNORECASE)
    text = text.strip(PUNCT_STRIP)
    if not text:
        return None
    if strict and re.search(r"\d", text):
        return None

    letters = []
    started = False
    for match in re.finditer(r"[A-Za-z]+", text):
        word = match.group(0)
        lower = word.lower()

        # Connectors keep the answer running: "A and C", "A or C".
        if lower in CONNECTORS:
            continue

        # A single letter is always a candidate, whatever its case.
        if len(word) == 1:
            key = word.upper()
            if key in valid_set:
                letters.append(key)
                started = True
            else:
                return None
            continue

        # All-caps runs: "AC", "ABD".  Prose all-caps words are stopwords.
        if word == word.upper():
            if word in CAPS_STOPWORDS:
                if not started:
                    continue
                if strict:
                    return None
                break
            chars = list(word)
            if all(char in valid_set for char in chars):
                letters.extend(chars)
                started = True
                continue
            # An all-caps token with an out-of-range letter is a real parse error.
            return None

        # Lowercase / mixed-case prose.
        if lower in STOPWORDS:
            if not started:
                continue
            if strict:
                return None
            break
        # Tolerate a short all-lowercase letter run such as "ac".
        if word == lower and len(word) <= 4 and all(
                char.upper() in valid_set for char in word):
            letters.extend(char.upper() for char in word)
            started = True
            continue
        if not started:
            return None
        if strict:
            return None
        break

    if not letters:
        return None
    present = set(letters)
    return [key for key in valid_order if key in present]


def parse_final_answer(raw_text, valid_order):
    """Return (letter_list_or_None, metadata) for one sampled chain."""
    text = _clean_output(raw_text)
    if "</think>" in text:
        # The final answer must live after the thinking block.  Ignoring the reasoning
        # text entirely prevents a tentative "Answer: B" inside the thought from
        # beating the real final answer.
        regions = [("post_think", text.rsplit("</think>", 1)[1])]
    else:
        regions = [("full", text)]

    candidates = []
    for region_name, region in regions:
        if not region.strip():
            continue
        for match in reversed(list(ANSWER_RE.finditer(region))):
            segment = _first_nonempty_line(region[match.end():], max_lines=2)
            if segment and len(segment) <= MARKER_SEGMENT_MAX:
                candidates.append((region_name + "_marker", match.group(0), segment))

    # Fallback: a short, pure-answer last line even when the model forgot the marker.
    for region_name, region in regions:
        if not region.strip():
            continue
        lines = [line.strip() for line in region.split("\n") if line.strip()]
        for line in reversed(lines[-4:]):
            if len(line) <= FALLBACK_LINE_MAX:
                candidates.append((region_name + "_lastline", "", line))

    seen = set()
    for source, marker, segment in candidates:
        strict = source.endswith("_lastline")
        key = (strict, segment)
        if key in seen:
            continue
        seen.add(key)
        parsed = _parse_segment(segment, valid_order, strict=strict)
        if parsed is not None:
            return parsed, {"source": source, "marker": marker,
                            "segment": segment[:200]}
    return None, {"source": None, "reason": "no_valid_parse", "tail": text[-300:]}


# (expected, valid_options, raw_output)
PARSER_CASES = [
    (["A", "C"], "ABCDE", "答案：A、C"),
    (["A", "C"], "ABCDE", "答案: A、C"),
    (["A", "C"], "ABCDE", "Answer: AC"),
    (["A", "C"], "ABCDE", "Answer: A C"),
    (["A", "C"], "ABCDE", "Answer: A,C"),
    (["A", "C"], "ABCDE", "**Answer:** A, C."),
    (["A", "C"], "ABCDE", "最终答案：Ａ、Ｃ"),
    (["A", "C"], "ABCDE", "Answer: A and C"),
    (["A", "B", "D"], "ABCDE", "Answer: A, B, and D"),
    (["A", "C"], "ABCDE", "The correct options are A, C"),
    (["A", "C"], "ABCDE", "The answer is A and C."),
    (["A", "C"], "ABCDE", "答案：AC（正确）"),
    (["A", "C"], "ABCDE", "Answer: \\boxed{AC}"),
    (["A", "C"], "ABCDE", "Answer: `AC`"),
    (["A", "C"], "ABCDE", "Answer: (A) (C)"),
    (["A", "C"], "ABCDE", "Answer: A; C"),
    (["A", "C"], "ABCDE", "Answer: A / C"),
    (["A", "C"], "ABCDE", "Answer: A or C"),
    (["A", "C"], "ABCDE", "答案：A和C"),
    (["A", "C"], "ABCDE", "答案：选A、C"),
    (["A", "C"], "ABCDE", "答案：\nA、C"),
    (["A", "C"], "ABCDE",
     "<think>Let me check. Answer: B is plausible.</think>\nAnswer: AC"),
    (["A", "C"], "ABCDE",
     "<think>The answer might be B.</think>\nAC"),
    (["A", "B", "D"], "ABCDE", "Answer: ABD"),
    (["B", "E"], "ABCDE", "Answer: BE"),
    (["I", "S"], "ABCDEFGHIJKLMNOPQRS", "Answer: IS"),
    (None, "ABCDE", "Answer: ALL"),
    (None, "ABCDE", "Answer: be"),
    (["A"], "ABCDE", "Answer: A is correct, B is wrong"),
    (["A", "C"], "ABCDE", "Answer: A, C. Option B is incorrect."),
    (["A", "C"], "ABCDE", "Answer: A, C. B and D are incorrect."),
    (["A", "C"], "ABCDE", "Answer: B is wrong, so A and C"),
    (["A"], "ABCDE", "答案：A正确、C错误"),
    (["C"], "ABCDE", "答案：A错误C正确"),
    (["D"], "ABCDE", "答案：A、C错误D正确"),
    (None, "ABCDE", "答案：排除A、C"),
    (["A"], "ABCDE", "Answer: A is correct, C is incorrect"),
    (["C"], "ABCDE", "Answer: A is incorrect, C is correct"),
    (["C"], "ABCDE", "Answer: not A, C"),
    (["A", "C"], "ABCDE", "Answer: I choose A and C"),
    (["A", "C"], "ABCDE", "Answer: I think the correct options are A and C"),
    (["A", "C"], "ABCDE", "Answer: A, C. B is not a correct option."),
    (["A", "C"], "ABCDE", "答案：A、C。B不是正确选项"),
    (["A", "C"], "ABCDE", "答案：A、C。B并非正确选项"),
    (["A", "B", "C"], "ABCDE", "答案：A、C。B正确、D错误"),
    (None, "ABCDE", "Answer: G"),
    (None, "ABCDE", "Answer: A, G"),
    (None, "ABCDE", "Answer: None"),
    (None, "ABCDE", "Answer:"),
    (None, "ABCDE", "答案：以上都是"),
    (None, "ABCDE", "Answer: All of the above"),
    (None, "ABCDE", "I am not sure."),
    (None, "ABCDE", "Answer: 3"),
    (["A", "C"], "ABCDE", "答案：A（错误）、C（正确）"),  # known heuristic limitation
    # Stage-2 answers: no </think> in the generated text, optional chat control
    # tokens (skip_special_tokens=False), and a blank line before the answer.
    (["A", "C"], "ABCDE", "Answer: AC<|im_end|>"),
    (["A", "C"], "ABCDE", "Answer: A, C<|im_end|>\n"),
    (["B"], "ABCDE", "<think>\n\n</think>\n\nAnswer: B"),
    (["B"], "ABCDE", "\n\nAnswer: B"),
    (["A", "C"], "ABCDE", "The reasoning above identifies both. Answer: AC"),
    (None, "ABCDE", ""),
    (None, "ABCDE", "<|im_end|>"),
]


def parser_selftest():
    failures = []
    records = []
    for expected, valid, raw in PARSER_CASES:
        got, meta = parse_final_answer(raw, list(valid))
        ok = got == expected
        records.append({"raw": raw, "valid": valid, "expected": expected,
                        "got": got, "ok": ok, "meta": meta})
        if not ok:
            failures.append((expected, got, raw, meta))
    return records, failures


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
    """Drop a partial last line left by a killed process."""
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


def atomic_write_text(path, text):
    path = Path(path)
    _guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class JsonlAppender:
    """Append one JSON record per os.write on an O_APPEND fd.

    A crash can leave at most a partial final line; resume sanitizes it and the merge
    step dedupes by id.  The final merged file is written atomically.
    """

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
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            data = line.encode("utf-8")
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
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("id"), str):
                done.add(row["id"])
    return done


# ---------------------------------------------------------------------------
# data / prompt
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
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError(f"{ident}: requires 2..26 options")
    keys = []
    for option in options:
        key = option.get("key")
        if not isinstance(key, str) or len(key) != 1 or key not in LETTERS:
            raise ValueError(f"{ident}: option keys must be single letters A..Z, got {key!r}")
        if not isinstance(option.get("desc"), str):
            raise ValueError(f"{ident}: option description must be a string")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{ident}: duplicate option keys")
    if row.get("n_options") is not None and row["n_options"] != len(options):
        raise ValueError(f"{ident}: n_options {row['n_options']} != {len(options)} options")
    target = row.get("target")
    if target is not None:
        if not isinstance(target, list) or len(target) != len(options):
            raise ValueError(f"{ident}: target must have one value per option")
        if any(value not in (0, 1) for value in target):
            raise ValueError(f"{ident}: target must be multi-hot 0/1")
    return row


def instruction_for(row):
    return ZH_INSTRUCTION if row.get("lang") == "zh" else EN_INSTRUCTION


def make_prompt_body(row):
    """State/Question/Options block shared by the one-stage and two-stage prompts."""
    choices = "\n".join(
        f"({option['key']}) {option['key']}: {option['desc']}"
        for option in row["options"]
    )
    return (f"State:\n{row['state']}\nQuestion: {row['question']}\n"
            f"Options:\n{choices}")


def make_prompt_content(row):
    return make_prompt_body(row) + "\n\n" + instruction_for(row)


def make_prompt(tokenizer, row):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt_content(row)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)


def make_stage2_prompt_content(row, reasoning):
    return (make_prompt_body(row) + f"\n\nReasoning:\n{reasoning}"
            "\n\nNow output the final answer. Reply with the correct option letters "
            "only on the final line as: Answer: AC")


def make_stage2_prompt(tokenizer, row, reasoning):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": make_stage2_prompt_content(row, reasoning)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt_length(tokenizer, prompt):
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def prompt_lengths(tokenizer, rows):
    return [prompt_length(tokenizer, prompt)
            for prompt in (make_prompt(tokenizer, row) for row in rows)]


def extract_stage1_reasoning(text):
    """Reasoning text of one stage-1 chain, with chat control tokens removed.

    vLLM's ``stop=["</think>"]`` normally excludes the stop string from ``text``;
    if a build includes it, everything after the marker is dropped.  A leading
    ``<think>`` (the template opener echoed back) is stripped as well.
    """
    if not isinstance(text, str):
        return ""
    text = _clean_output(text)
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    text = text.strip()
    while text.startswith("<think>"):
        text = text[len("<think>"):].lstrip()
    text = text.replace("<think>", " ").replace("</think>", " ")
    return text.strip()


def fit_stage2_prompt(tokenizer, row, reasoning, budget):
    """Return (prompt, reasoning_used, prompt_tokens) fitting ``budget`` tokens.

    When the full reasoning does not fit, keep its tail (the part closest to the
    conclusion) and binary-search the largest tail that fits.  Returns None when
    even the empty reasoning does not fit the budget.
    """
    prompt = make_stage2_prompt(tokenizer, row, reasoning)
    length = prompt_length(tokenizer, prompt)
    if length <= budget:
        return prompt, reasoning, length

    ids = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
    best = None
    low, high = 0, len(ids)
    while low <= high:
        keep = (low + high) // 2
        if keep:
            candidate_text = tokenizer.decode(ids[len(ids) - keep:],
                                              skip_special_tokens=True)
        else:
            candidate_text = ""
        candidate = make_stage2_prompt(tokenizer, row, candidate_text)
        candidate_length = prompt_length(tokenizer, candidate)
        if candidate_length <= budget:
            best = (candidate, candidate_text, candidate_length)
            low = keep + 1
        else:
            high = keep - 1
    return best



def check_model_complete(model_path):
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"{model_path}: model.safetensors.index.json not found")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_path / name).exists()]
    if missing:
        raise SystemExit(
            f"{model_path}: missing {len(missing)}/{len(shards)} weight shards: "
            f"{missing[:5]}")
    return shards


# ---------------------------------------------------------------------------
# generation
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
        disable_log_stats=args.disable_log_stats,
    )
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    llm = LLM(**kwargs)

    sampling_kwargs = dict(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )
    if args.thinking_token_budget and not args.two_stage:
        # vLLM 0.19.1 rejects this without --reasoning-config; two-stage mode
        # supersedes it and must never send it.
        sampling_kwargs["thinking_token_budget"] = args.thinking_token_budget
    sampling = SamplingParams(**sampling_kwargs)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return llm, sampling, gpu_name


def build_stage_samplers(args, sampling):
    """Stage-1/stage-2 SamplingParams for two-stage mode.

    Stage 1 reuses the configured sampling distribution and ``n`` but caps the
    thinking at ``--thinking-max-tokens`` and stops at ``</think>``.  Stage 2 is
    greedy and only produces the ``Answer: ...`` line.
    """
    from vllm import SamplingParams
    stage1 = SamplingParams(
        n=args.samples,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        max_tokens=args.thinking_max_tokens,
        stop=["</think>"],
        skip_special_tokens=False,
    )
    stage2 = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.answer_max_tokens,
        skip_special_tokens=False,
    )
    return stage1, stage2



def build_record(row, request_output, requested_n):
    """Turn one RequestOutput (n chains) into the output line plus invalid logs."""
    valid_order = [option["key"] for option in row["options"]]
    outputs = list(getattr(request_output, "outputs", None) or [])
    prompt_tokens = getattr(request_output, "prompt_token_ids", None)
    prompt_tokens = len(prompt_tokens) if prompt_tokens else None

    sets = []
    finish_reasons = []
    completion_tokens = []
    invalid = []
    for chain in range(requested_n):
        if chain < len(outputs):
            completion = outputs[chain]
            text = completion.text or ""
            finish_reason = completion.finish_reason or "unknown"
            token_ids = getattr(completion, "token_ids", None)
            chain_tokens = len(token_ids) if token_ids is not None else None
        else:
            text = ""
            finish_reason = "missing"
            chain_tokens = 0
        finish_reasons.append(finish_reason)
        completion_tokens.append(chain_tokens)

        if finish_reason in ("length", "abort", "missing"):
            meta = {"source": None,
                    "reason": "truncated" if finish_reason == "length" else finish_reason,
                    "tail": text[-300:]}
            parsed = None
        else:
            parsed, meta = parse_final_answer(text, valid_order)

        if parsed is None:
            invalid.append({
                "id": row["id"], "chain": chain,
                "reason": meta.get("reason") or "unparsed",
                "finish_reason": finish_reason,
                "parse_source": meta.get("source"),
                "marker": meta.get("marker"),
                "segment": meta.get("segment"),
                "tail": meta.get("tail"),
            })
        else:
            sets.append(parsed)

    valid_chains = len(sets)
    if valid_chains:
        q = [sum(1 for item in sets if key in item) / valid_chains
             for key in valid_order]
        q = [round(value, 6) for value in q]
    else:
        q = None
        invalid.append({
            "id": row["id"], "chain": None, "reason": "no_valid_chains",
            "n_chains": requested_n, "finish_reasons": finish_reasons,
        })

    record = {
        "id": row["id"],
        "source": row["source"],
        "lang": row.get("lang"),
        "n_options": len(valid_order),
        "n_chains": requested_n,
        "valid_chains": valid_chains,
        "sets": sets,
        "q": q,
        "finish_reasons": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return record, invalid


BAD_FINISH_REASONS = ("length", "abort", "missing")


def _chain_texts(stage2_entry):
    """(text, finish_reason, completion_tokens) of one stage-2 chain."""
    error = stage2_entry.get("stage2_error")
    if error:
        return "", error, 0
    request_output = stage2_entry.get("stage2_output")
    if request_output is None:
        return "", "missing", 0
    outputs = list(getattr(request_output, "outputs", None) or [])
    if not outputs:
        return "", "missing", 0
    completion = outputs[0]
    token_ids = getattr(completion, "token_ids", None)
    return (completion.text or "",
            completion.finish_reason or "unknown",
            len(token_ids) if token_ids is not None else 0)


def build_two_stage_record(row, chains, requested_n):
    """Assemble one two-stage output line from per-chain stage records.

    ``chains[i]`` holds the stage-1 outcome (finish reason, completion tokens,
    reasoning text), the fitted stage-2 prompt length and either the stage-2
    ``RequestOutput`` or a ``stage2_error`` string.  A chain is valid iff stage 2
    produced a parseable answer.
    """
    valid_order = [option["key"] for option in row["options"]]
    sets = []
    finish_reasons = []
    stage1_finish_reasons = []
    stage2_finish_reasons = []
    completion_tokens = []
    prompt_tokens = (chains[0].get("stage1_prompt_tokens") or 0) if chains else 0
    invalid = []

    for chain in range(requested_n):
        entry = chains[chain] if chain < len(chains) else {}
        stage1_finish = entry.get("stage1_finish_reason") or "missing"
        stage1_tokens = entry.get("stage1_completion_tokens") or 0
        text, stage2_finish, stage2_tokens = _chain_texts(entry)

        stage1_finish_reasons.append(stage1_finish)
        stage2_finish_reasons.append(stage2_finish)
        # Stage 2 decides validity; mirror it in the legacy field.
        finish_reasons.append(stage2_finish)
        completion_tokens.append(stage1_tokens + stage2_tokens)
        prompt_tokens += entry.get("stage2_prompt_tokens") or 0

        if entry.get("stage2_error"):
            meta = {"source": None, "reason": entry["stage2_error"]}
            parsed = None
        elif stage2_finish in BAD_FINISH_REASONS:
            meta = {"source": None,
                    "reason": ("truncated_answer" if stage2_finish == "length"
                               else stage2_finish),
                    "tail": text[-300:]}
            parsed = None
        else:
            parsed, meta = parse_final_answer(text, valid_order)

        if parsed is None:
            invalid.append({
                "id": row["id"], "chain": chain,
                "reason": meta.get("reason") or "unparsed",
                "finish_reason": stage2_finish,
                "stage1_finish_reason": stage1_finish,
                "stage2_finish_reason": stage2_finish,
                "stage1_reasoning": (entry.get("reasoning") or "")[-300:],
                "parse_source": meta.get("source"),
                "marker": meta.get("marker"),
                "segment": meta.get("segment"),
                "tail": meta.get("tail"),
            })
        else:
            sets.append(parsed)

    valid_chains = len(sets)
    if valid_chains:
        q = [sum(1 for item in sets if key in item) / valid_chains
             for key in valid_order]
        q = [round(value, 6) for value in q]
    else:
        q = None
        invalid.append({
            "id": row["id"], "chain": None, "reason": "no_valid_chains",
            "n_chains": requested_n, "finish_reasons": finish_reasons,
            "stage1_finish_reasons": stage1_finish_reasons,
            "stage2_finish_reasons": stage2_finish_reasons,
        })

    record = {
        "id": row["id"],
        "source": row["source"],
        "lang": row.get("lang"),
        "n_options": len(valid_order),
        "n_chains": requested_n,
        "valid_chains": valid_chains,
        "sets": sets,
        "q": q,
        "finish_reasons": finish_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stage1_finish_reasons": stage1_finish_reasons,
        "stage2_finish_reasons": stage2_finish_reasons,
    }
    return record, invalid



def _generate_with_retry(llm, prompts, sampling, tokenization_kwargs):
    """Split a failed batch down to single prompts; return None for a failed prompt."""
    try:
        return llm.generate(prompts, sampling, use_tqdm=False,
                            tokenization_kwargs=tokenization_kwargs)
    except Exception as error:  # noqa: BLE001 - vLLM raises many engine-specific types
        if len(prompts) > 1:
            middle = len(prompts) // 2
            log(f"[warn] batch of {len(prompts)} failed ({error}); splitting")
            left = _generate_with_retry(llm, prompts[:middle], sampling,
                                        tokenization_kwargs)
            right = _generate_with_retry(llm, prompts[middle:], sampling,
                                         tokenization_kwargs)
            return left + right
        log(f"[error] prompt failed: {error}")
        return [None]


def _new_profile(args, input_path, output_path, rows_selected):
    return {
        "input": str(input_path),
        "output": str(output_path),
        "model": str(args.model),
        "shard": args.shard,
        "shards": args.shards,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "rows_selected": rows_selected,
        "items": 0,
        "valid_chains": 0,
        "invalid_chains": 0,
        "zero_valid": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "gen_seconds": 0.0,
        "items_per_s": 0.0,
        "completion_tokens_per_s": 0.0,
        "total_tokens_per_s": 0.0,
        "gpu_name": None,
        "status": "empty",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def generate_one_input(llm, sampling, tokenizer, args, input_path, output_path):
    rows = read_jsonl(input_path)
    for row in rows:
        validate_row(row)
    if args.limit:
        rows = rows[: args.limit]

    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.shards == args.shard]
    output_path = Path(output_path)
    _guard_output(output_path)
    invalid_path = output_path.with_suffix(".invalid.jsonl")
    # part-00003.jsonl -> profile-00003.json (the summary globs profile-*.json)
    profile_path = output_path.with_name(
        output_path.stem.replace("part-", "profile-") + ".json")

    if args.resume:
        done = load_done_ids(output_path)
        if done:
            selected = [(index, row) for index, row in selected
                        if row["id"] not in done]
            log(f"[{input_path.name} shard {args.shard}] resume: {len(done)} ids done, "
                f"{len(selected)} left")

    profile = _new_profile(
        args, input_path, output_path,
        len(rows) // args.shards + (1 if args.shard < len(rows) % args.shards else 0))
    if not selected:
        atomic_write_json(profile_path, profile)
        log(f"[{input_path.name} shard {args.shard}] nothing left to do")
        return profile

    tokenize_begin = time.monotonic()
    prompts = [make_prompt(tokenizer, row) for _, row in selected]
    lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
               for prompt in prompts]
    tokenize_seconds = time.monotonic() - tokenize_begin

    budget = args.max_model_len - args.max_tokens
    over_budget = [(index, row, length) for (index, row), length in zip(selected, lengths)
                   if length > budget]
    if over_budget:
        log(f"[{input_path.name} shard {args.shard}] {len(over_budget)} prompts exceed "
            f"max_model_len-max_tokens={budget}; writing q=null for them")
    keep = [(index, row, prompt, length) for (index, row), prompt, length
            in zip(selected, prompts, lengths) if length <= budget]

    begin = time.monotonic()
    processed = 0
    with JsonlAppender(output_path, resume=args.resume and output_path.exists()) as out, \
            JsonlAppender(invalid_path,
                          resume=args.resume and invalid_path.exists()) as invalid_out:
        # Every input id must produce one output line.  Prompts that cannot fit the
        # context budget are written with q=null and logged instead of being dropped.
        for _index, row, length in over_budget:
            out.append([{
                "id": row["id"], "source": row["source"], "lang": row.get("lang"),
                "n_options": len(row["options"]), "n_chains": args.samples,
                "valid_chains": 0, "sets": [], "q": None,
                "finish_reasons": ["prompt_too_long"] * args.samples,
                "prompt_tokens": length,
                "completion_tokens": [0] * args.samples,
            }])
            invalid_out.append([{
                "id": row["id"], "chain": None, "reason": "prompt_too_long",
                "prompt_tokens": length,
                "max_model_len": args.max_model_len, "max_tokens": args.max_tokens,
            }])
            processed += 1
            profile["invalid_chains"] += args.samples
            profile["zero_valid"] += 1
            profile["prompt_tokens"] += length

        for offset in range(0, len(keep), args.batch_size):
            batch = keep[offset: offset + args.batch_size]
            outputs = _generate_with_retry(
                llm, [item[2] for item in batch], sampling,
                {"add_special_tokens": False})
            records, invalid_records = [], []
            for (_index, row, _prompt, prompt_length), request_output in zip(
                    batch, outputs):
                if request_output is None:
                    records.append({
                        "id": row["id"], "source": row["source"], "lang": row.get("lang"),
                        "n_options": len(row["options"]), "n_chains": args.samples,
                        "valid_chains": 0, "sets": [], "q": None,
                        "finish_reasons": ["error"], "prompt_tokens": prompt_length,
                        "completion_tokens": [0] * args.samples,
                    })
                    invalid_records.append({
                        "id": row["id"], "chain": None, "reason": "generation_error",
                        "n_chains": args.samples,
                    })
                    continue
                record, invalid = build_record(row, request_output, args.samples)
                if record["prompt_tokens"] is None:
                    # Some vLLM builds omit prompt_token_ids; use our tokenizer count.
                    record["prompt_tokens"] = prompt_length
                records.append(record)
                invalid_records.extend(invalid)
            out.append(records)
            if invalid_records:
                invalid_out.append(invalid_records)

            processed += len(records)
            for record in records:
                profile["valid_chains"] += record["valid_chains"]
                profile["invalid_chains"] += record["n_chains"] - record["valid_chains"]
                profile["zero_valid"] += 1 if record["valid_chains"] == 0 else 0
                profile["prompt_tokens"] += record["prompt_tokens"] or 0
                profile["completion_tokens"] += sum(
                    token for token in record["completion_tokens"] if token)
            if args.progress_every and processed % args.progress_every < len(records):
                elapsed = time.monotonic() - begin
                log(f"[{input_path.name} shard {args.shard}] {processed}/"
                    f"{len(keep) + len(over_budget)} rows, "
                    f"{profile['valid_chains']} valid chains, "
                    f"{profile['completion_tokens']/max(elapsed,1e-9):.0f} tok/s")

    gen_seconds = time.monotonic() - begin
    profile.update({
        "items": processed,
        "gen_seconds": round(gen_seconds, 3),
        "tokenize_seconds": round(tokenize_seconds, 3),
        "items_per_s": round(processed / gen_seconds, 4) if gen_seconds else 0.0,
        "completion_tokens_per_s": (round(profile["completion_tokens"] / gen_seconds, 2)
                                    if gen_seconds else 0.0),
        "total_tokens_per_s": (round((profile["completion_tokens"]
                                      + profile["prompt_tokens"]) / gen_seconds, 2)
                               if gen_seconds else 0.0),
        "status": "ok" if keep else "all_over_budget",
    })
    atomic_write_json(profile_path, profile)
    log(f"[{input_path.name} shard {args.shard}] done: {processed} rows, "
        f"{profile['valid_chains']} valid chains, "
        f"{profile['invalid_chains']} invalid, {profile['zero_valid']} zero-valid, "
        f"{profile['completion_tokens_per_s']:.0f} tok/s")
    return profile


def generate_one_input_two_stage(llm, stage1_sampling, stage2_sampling, tokenizer,
                                 args, input_path, output_path):
    """Two-stage generation for one split/shard (thinking, then forced answer)."""
    rows = read_jsonl(input_path)
    for row in rows:
        validate_row(row)
    if args.limit:
        rows = rows[: args.limit]

    selected = [(index, row) for index, row in enumerate(rows)
                if index % args.shards == args.shard]
    output_path = Path(output_path)
    _guard_output(output_path)
    invalid_path = output_path.with_suffix(".invalid.jsonl")
    profile_path = output_path.with_name(
        output_path.stem.replace("part-", "profile-") + ".json")

    if args.resume:
        done = load_done_ids(output_path)
        if done:
            selected = [(index, row) for index, row in selected
                        if row["id"] not in done]
            log(f"[{input_path.name} shard {args.shard}] resume: {len(done)} ids done, "
                f"{len(selected)} left")

    profile = _new_profile(
        args, input_path, output_path,
        len(rows) // args.shards + (1 if args.shard < len(rows) % args.shards else 0))
    stage1_budget = args.max_model_len - args.thinking_max_tokens
    stage2_budget = args.max_model_len - args.answer_max_tokens
    profile.update({
        "two_stage": True,
        "thinking_max_tokens": args.thinking_max_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "stage1_prompt_budget": stage1_budget,
        "stage2_prompt_budget": stage2_budget,
        "stage1_finish_reasons": {},
        "stage2_finish_reasons": {},
        "stage2_prompt_tokens": 0,
    })
    if not selected:
        atomic_write_json(profile_path, profile)
        log(f"[{input_path.name} shard {args.shard}] nothing left to do")
        return profile

    tokenize_begin = time.monotonic()
    prompts = [make_prompt(tokenizer, row) for _, row in selected]
    lengths = [prompt_length(tokenizer, prompt) for prompt in prompts]
    tokenize_seconds = time.monotonic() - tokenize_begin

    over_budget = [(index, row, length) for (index, row), length in zip(selected, lengths)
                   if length > stage1_budget]
    if over_budget:
        log(f"[{input_path.name} shard {args.shard}] {len(over_budget)} prompts exceed "
            f"max_model_len-thinking_max_tokens={stage1_budget}; writing q=null for them")
    keep = [(index, row, prompt, length) for (index, row), prompt, length
            in zip(selected, prompts, lengths) if length <= stage1_budget]

    def add_reason(record):
        for reason in record.get("stage1_finish_reasons") or []:
            counter = profile["stage1_finish_reasons"]
            counter[str(reason)] = counter.get(str(reason), 0) + 1
        for reason in record.get("stage2_finish_reasons") or []:
            counter = profile["stage2_finish_reasons"]
            counter[str(reason)] = counter.get(str(reason), 0) + 1

    def add_record_counts(record):
        profile["valid_chains"] += record["valid_chains"]
        profile["invalid_chains"] += record["n_chains"] - record["valid_chains"]
        profile["zero_valid"] += 1 if record["valid_chains"] == 0 else 0
        profile["prompt_tokens"] += record["prompt_tokens"] or 0
        profile["completion_tokens"] += sum(
            token for token in record["completion_tokens"] if token)
        add_reason(record)

    begin = time.monotonic()
    processed = 0
    with JsonlAppender(output_path, resume=args.resume and output_path.exists()) as out, \
            JsonlAppender(invalid_path,
                          resume=args.resume and invalid_path.exists()) as invalid_out:
        for _index, row, length in over_budget:
            record = {
                "id": row["id"], "source": row["source"], "lang": row.get("lang"),
                "n_options": len(row["options"]), "n_chains": args.samples,
                "valid_chains": 0, "sets": [], "q": None,
                "finish_reasons": ["prompt_too_long"] * args.samples,
                "prompt_tokens": length,
                "completion_tokens": [0] * args.samples,
                "stage1_finish_reasons": ["prompt_too_long"] * args.samples,
                "stage2_finish_reasons": ["not_run"] * args.samples,
            }
            out.append([record])
            invalid_out.append([{
                "id": row["id"], "chain": None, "reason": "prompt_too_long",
                "prompt_tokens": length,
                "max_model_len": args.max_model_len,
                "thinking_max_tokens": args.thinking_max_tokens,
            }])
            processed += 1
            add_record_counts(record)

        for offset in range(0, len(keep), args.batch_size):
            batch = keep[offset: offset + args.batch_size]
            stage1_outputs = _generate_with_retry(
                llm, [item[2] for item in batch], stage1_sampling,
                {"add_special_tokens": False})

            rows_entries = []
            pending = []
            for (_index, row, _prompt, stage1_prompt_tokens), stage1_output in zip(
                    batch, stage1_outputs):
                if stage1_output is None:
                    rows_entries.append((row, stage1_prompt_tokens, None, None))
                    continue
                outputs = list(getattr(stage1_output, "outputs", None) or [])
                entries = []
                for chain in range(args.samples):
                    if chain < len(outputs):
                        completion = outputs[chain]
                        stage1_finish = completion.finish_reason or "unknown"
                        token_ids = getattr(completion, "token_ids", None)
                        stage1_tokens = len(token_ids) if token_ids is not None else 0
                        reasoning = extract_stage1_reasoning(completion.text or "")
                    else:
                        stage1_finish = "missing"
                        stage1_tokens = 0
                        reasoning = ""
                    entry = {
                        "stage1_finish_reason": stage1_finish,
                        "stage1_completion_tokens": stage1_tokens,
                        "stage1_prompt_tokens": stage1_prompt_tokens,
                        "reasoning": reasoning,
                        "stage2_prompt_tokens": 0,
                        "stage2_output": None,
                        "stage2_error": None,
                    }
                    entries.append(entry)
                    fitted = fit_stage2_prompt(tokenizer, row, reasoning, stage2_budget)
                    if fitted is None:
                        entry["stage2_error"] = "prompt_too_long"
                    else:
                        stage2_prompt, _used, stage2_prompt_tokens = fitted
                        entry["stage2_prompt_tokens"] = stage2_prompt_tokens
                        pending.append((entry, stage2_prompt))
                rows_entries.append((row, stage1_prompt_tokens, stage1_output, entries))

            if pending:
                stage2_outputs = _generate_with_retry(
                    llm, [prompt for _entry, prompt in pending], stage2_sampling,
                    {"add_special_tokens": False})
                for (entry, _prompt), stage2_output in zip(pending, stage2_outputs):
                    entry["stage2_output"] = stage2_output

            records, invalid_records = [], []
            for row, stage1_prompt_tokens, stage1_output, entries in rows_entries:
                if stage1_output is None:
                    record = {
                        "id": row["id"], "source": row["source"],
                        "lang": row.get("lang"), "n_options": len(row["options"]),
                        "n_chains": args.samples, "valid_chains": 0, "sets": [],
                        "q": None, "finish_reasons": ["error"] * args.samples,
                        "prompt_tokens": stage1_prompt_tokens,
                        "completion_tokens": [0] * args.samples,
                        "stage1_finish_reasons": ["error"] * args.samples,
                        "stage2_finish_reasons": ["not_run"] * args.samples,
                    }
                    records.append(record)
                    invalid_records.append({
                        "id": row["id"], "chain": None,
                        "reason": "generation_error", "n_chains": args.samples,
                    })
                    continue
                record, invalid = build_two_stage_record(
                    row, entries, args.samples)
                records.append(record)
                invalid_records.extend(invalid)

            out.append(records)
            if invalid_records:
                invalid_out.append(invalid_records)
            processed += len(records)
            for record in records:
                add_record_counts(record)
            for _row, _plen, _s1, entries in rows_entries:
                profile["stage2_prompt_tokens"] += sum(
                    entry.get("stage2_prompt_tokens") or 0
                    for entry in (entries or []))
            if args.progress_every and processed % args.progress_every < len(records):
                elapsed = time.monotonic() - begin
                log(f"[{input_path.name} shard {args.shard}] {processed}/"
                    f"{len(keep) + len(over_budget)} rows, "
                    f"{profile['valid_chains']} valid chains, "
                    f"{profile['completion_tokens']/max(elapsed,1e-9):.0f} tok/s")

    gen_seconds = time.monotonic() - begin
    profile.update({
        "items": processed,
        "gen_seconds": round(gen_seconds, 3),
        "tokenize_seconds": round(tokenize_seconds, 3),
        "items_per_s": round(processed / gen_seconds, 4) if gen_seconds else 0.0,
        "completion_tokens_per_s": (round(profile["completion_tokens"] / gen_seconds, 2)
                                    if gen_seconds else 0.0),
        "total_tokens_per_s": (round((profile["completion_tokens"]
                                      + profile["prompt_tokens"]) / gen_seconds, 2)
                               if gen_seconds else 0.0),
        "status": "ok" if keep else "all_over_budget",
    })
    atomic_write_json(profile_path, profile)
    log(f"[{input_path.name} shard {args.shard}] done: {processed} rows, "
        f"{profile['valid_chains']} valid chains, "
        f"{profile['invalid_chains']} invalid, {profile['zero_valid']} zero-valid, "
        f"{profile['completion_tokens_per_s']:.0f} tok/s")
    return profile


def run_generation(args):
    check_model_complete(args.model)
    inputs = [Path(path) for path in args.input]
    outputs = [Path(path) for path in args.output]
    if not inputs:
        raise SystemExit("generation requires at least one --input")
    if len(outputs) != len(inputs):
        raise SystemExit("--output must be given once per --input")
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"input not found: {path}")
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))

    # Avoid loading a 4B model when every shard is already complete.
    pending = []
    for input_path, output_path in zip(inputs, outputs):
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[: args.limit]
        shard_rows = [(i, row) for i, row in enumerate(rows)
                      if i % args.shards == args.shard]
        done = load_done_ids(output_path) if args.resume else set()
        pending.append(any(row["id"] not in done for _, row in shard_rows))
    if not any(pending):
        log(f"[shard {args.shard}] all inputs complete; skipping model load")
        return 0

    load_begin = time.monotonic()
    llm, sampling, gpu_name = build_engine(args)
    load_seconds = time.monotonic() - load_begin
    log(f"[shard {args.shard}] model loaded in {load_seconds:.1f}s on {gpu_name}")

    stage1_sampling = stage2_sampling = None
    if args.two_stage:
        stage1_sampling, stage2_sampling = build_stage_samplers(args, sampling)
        log(f"[shard {args.shard}] two-stage: thinking<={args.thinking_max_tokens} "
            f"tokens (stop='</think>'), answer<={args.answer_max_tokens} tokens greedy")

    status = 0
    for position, (input_path, output_path) in enumerate(zip(inputs, outputs)):
        try:
            if args.two_stage:
                profile = generate_one_input_two_stage(
                    llm, stage1_sampling, stage2_sampling, tokenizer, args,
                    input_path, output_path)
            else:
                profile = generate_one_input(llm, sampling, tokenizer, args,
                                             input_path, output_path)
            profile["gpu_name"] = gpu_name
            profile["load_seconds"] = round(load_seconds, 3) if position == 0 else 0.0
            profile_path = output_path.with_name(
                output_path.stem.replace("part-", "profile-") + ".json")
            atomic_write_json(profile_path, profile)
        except Exception as error:  # noqa: BLE001 - keep going to the next split
            log(f"[error] {input_path.name}: {error}")
            status = 1
    return status


# ---------------------------------------------------------------------------
# merge / summary
# ---------------------------------------------------------------------------
def merge_parts(parts_dir, output_path, invalid_log=None, input_path=None):
    parts_dir = Path(parts_dir)
    parts = sorted(path for path in parts_dir.glob("part-*.jsonl")
                   if not path.name.endswith(".invalid.jsonl"))
    if not parts:
        log(f"[merge] no part files under {parts_dir}")
        return 0

    merged = {}
    for part in parts:
        sanitize_jsonl(part)
        for line in part.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            merged[record["id"]] = record

    order = []
    if input_path and Path(input_path).exists():
        order = [row["id"] for row in read_jsonl(input_path)]
    written = set()
    lines = []
    for ident in order:
        if ident in merged:
            lines.append(json.dumps(merged[ident], ensure_ascii=False,
                                    allow_nan=False))
            written.add(ident)
    for ident in sorted(set(merged) - written):
        lines.append(json.dumps(merged[ident], ensure_ascii=False, allow_nan=False))
    atomic_write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))

    if invalid_log:
        invalid_records = []
        invalid_parts = sorted(parts_dir.glob("part-*.invalid.jsonl"))
        for part in invalid_parts:
            sanitize_jsonl(part)
            for line in part.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    invalid_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        for record in merged.values():
            if record.get("valid_chains") == 0:
                invalid_records.append({
                    "id": record["id"], "chain": None, "reason": "no_valid_chains",
                    "n_chains": record.get("n_chains"),
                    "finish_reasons": record.get("finish_reasons"),
                })
        invalid_lines = [json.dumps(record, ensure_ascii=False, allow_nan=False)
                         for record in invalid_records]
        atomic_write_text(invalid_log, "\n".join(invalid_lines)
                          + ("\n" if invalid_lines else ""))

    log(f"[merge] {len(parts)} parts -> {output_path} ({len(merged)} unique ids)")
    return len(merged)


def _token_stats(values):
    if not values:
        return {"n": 0, "sum": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": round(statistics.mean(values), 1),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


def summarize(all_records, input_rows, profiles, args):
    total = {
        "rows": len(all_records),
        "rows_with_valid": 0,
        "rows_zero_valid": 0,
        "chains": 0,
        "valid_chains": 0,
        "invalid_chains": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    q_histogram = Counter()
    finish_reasons = Counter()
    stage1_finish_reasons = Counter()
    stage2_finish_reasons = Counter()
    two_stage_rows = 0
    by_source = defaultdict(lambda: {
        "rows": 0, "rows_with_valid": 0, "rows_zero_valid": 0, "chains": 0,
        "valid_chains": 0, "invalid_chains": 0, "q_histogram": Counter(),
    })
    zero_valid_ids = []
    set_size_histogram = Counter()
    completion_tokens = []
    gold_pos_q, gold_neg_q = [], []
    exact_matches = 0
    jaccard_sum = 0.0
    gold_rows = 0

    input_by_id = {row["id"]: row for row in input_rows}
    for record in all_records:
        n_chains = int(record.get("n_chains") or 0)
        valid = int(record.get("valid_chains") or 0)
        total["chains"] += n_chains
        total["valid_chains"] += valid
        total["invalid_chains"] += n_chains - valid
        total["prompt_tokens"] += record.get("prompt_tokens") or 0
        total["completion_tokens"] += sum(
            value for value in (record.get("completion_tokens") or []) if value)
        if valid:
            total["rows_with_valid"] += 1
        else:
            total["rows_zero_valid"] += 1
            zero_valid_ids.append(record["id"])
        if record.get("q"):
            for value in record["q"]:
                q_histogram[format(value, ".6g")] += 1
        for reason in record.get("finish_reasons") or []:
            finish_reasons[str(reason)] += 1
        if "stage1_finish_reasons" in record:
            two_stage_rows += 1
            for reason in record.get("stage1_finish_reasons") or []:
                stage1_finish_reasons[str(reason)] += 1
            for reason in record.get("stage2_finish_reasons") or []:
                stage2_finish_reasons[str(reason)] += 1
        set_size_histogram[str(valid)] += 1
        completion_tokens.extend(
            value for value in (record.get("completion_tokens") or []) if value)

        source = record.get("source") or "unknown"
        bucket = by_source[source]
        bucket["rows"] += 1
        bucket["chains"] += n_chains
        bucket["valid_chains"] += valid
        bucket["invalid_chains"] += n_chains - valid
        if valid:
            bucket["rows_with_valid"] += 1
        else:
            bucket["rows_zero_valid"] += 1
        if record.get("q"):
            for value in record["q"]:
                bucket["q_histogram"][format(value, ".6g")] += 1

        gold = input_by_id.get(record["id"], {}).get("target")
        if gold and record.get("q"):
            gold_rows += 1
            keys = [option["key"] for option in input_by_id[record["id"]]["options"]]
            for key, q_value, flag in zip(keys, record["q"], gold):
                (gold_pos_q if flag else gold_neg_q).append(q_value)
            predicted = {key for key, q_value in zip(keys, record["q"]) if q_value >= 0.5}
            gold_set = {key for key, flag in zip(keys, gold) if flag}
            exact_matches += 1 if predicted == gold_set else 0
            union = predicted | gold_set
            jaccard_sum += len(predicted & gold_set) / len(union) if union else 1.0

    for bucket in by_source.values():
        bucket["q_histogram"] = dict(sorted(bucket["q_histogram"].items()))
        bucket["valid_rate"] = (round(bucket["valid_chains"] / bucket["chains"], 6)
                                if bucket["chains"] else None)

    total["valid_rate"] = (round(total["valid_chains"] / total["chains"], 6)
                           if total["chains"] else None)
    total_tokens = total["prompt_tokens"] + total["completion_tokens"]

    estimate = {
        "total_tokens": total_tokens,
        "assumed_tokens_per_second_per_gpu": args.assume_tokens_per_second,
        "assumed_gpus": args.assume_gpus,
        "assumed_price_usd_per_gpu_hour": args.price_per_gpu_hour,
        "gpu_hours": round(total_tokens / args.assume_tokens_per_second / 3600, 3)
        if args.assume_tokens_per_second else None,
        "wall_hours": round(total_tokens / args.assume_tokens_per_second
                            / args.assume_gpus / 3600, 3)
        if args.assume_tokens_per_second else None,
        "cost_usd": round(total_tokens / args.assume_tokens_per_second / 3600
                          * args.price_per_gpu_hour, 2)
        if args.assume_tokens_per_second else None,
        "note": "cost = gpu_hours * price; gpu_hours = total_tokens / assumed_rate "
                "/ 3600. Replace with observed numbers from the per-shard profiles.",
    }

    observed = None
    if profiles:
        # Each shard processes every split sequentially, so group profiles by shard and
        # sum their load+generation time before taking the slowest shard.
        seconds_by_shard = defaultdict(float)
        for profile in profiles:
            shard = int(profile.get("shard", 0))
            seconds_by_shard[shard] += (profile.get("load_seconds", 0.0)
                                        + profile.get("gen_seconds", 0.0))
        wall_seconds = max(seconds_by_shard.values())
        gpu_seconds = sum(seconds_by_shard.values())
        observed_tokens = sum(profile.get("prompt_tokens", 0)
                              + profile.get("completion_tokens", 0)
                              for profile in profiles)
        observed = {
            "shards": len(seconds_by_shard),
            "wall_seconds": round(wall_seconds, 1),
            "gpu_seconds": round(gpu_seconds, 1),
            "gpu_hours": round(gpu_seconds / 3600, 4),
            "tokens": observed_tokens,
            "tokens_per_second_per_gpu": round(
                observed_tokens / gpu_seconds, 1) if gpu_seconds else None,
            "cost_usd": round(gpu_seconds / 3600 * args.price_per_gpu_hour, 2),
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(args.model),
        "samples_per_question": args.samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "totals": total,
        "by_source": dict(sorted(by_source.items())),
        "q_histogram": dict(sorted(q_histogram.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "two_stage": bool(two_stage_rows),
        "two_stage_rows": two_stage_rows,
        "stage1_finish_reasons": dict(sorted(stage1_finish_reasons.items())),
        "stage2_finish_reasons": dict(sorted(stage2_finish_reasons.items())),
        "set_size_histogram": dict(sorted(set_size_histogram.items())),
        "completion_tokens": _token_stats(completion_tokens),
        "zero_valid": {
            "count": len(zero_valid_ids),
            "ids": zero_valid_ids[:100],
        },
        "gold_diagnostics": {
            "rows_with_gold": gold_rows,
            "mean_q_gold_positive": (round(statistics.mean(gold_pos_q), 6)
                                     if gold_pos_q else None),
            "mean_q_gold_negative": (round(statistics.mean(gold_neg_q), 6)
                                     if gold_neg_q else None),
            "exact_set_agreement_at_q_0.5": (round(exact_matches / gold_rows, 6)
                                             if gold_rows else None),
            "mean_jaccard_at_q_0.5": (round(jaccard_sum / gold_rows, 6)
                                      if gold_rows else None),
        },
        "estimated_cost": estimate,
        "observed": observed,
        "notes": [
            "q is null for rows with zero valid chains; the training harness drops them.",
            "Truncated chains (finish_reason=length) are treated as invalid; raise "
            "max_tokens or lower thinking length if the invalid rate is high.",
            "sets contains only valid chains; finish_reasons/completion_tokens cover "
            "all sampled chains.",
            "observed reflects the most recent invocation of each shard, so after a "
            "resume it covers only newly generated rows; estimated_cost uses the full "
            "merged token counts.",
        ],
    }

    if two_stage_rows:
        summary["notes"].append(
            "Two-stage mode: finish_reasons mirrors stage 2 (the answer pass that "
            "decides validity); stage1_finish_reasons records thinking-cap "
            "truncation, which no longer invalidates a chain.")

    if args.assume_avg_completion_tokens:
        assumed_completion = (total["rows"] * args.samples
                              * args.assume_avg_completion_tokens)
        summary["estimate_preflight"] = {
            "assumed_avg_completion_tokens": args.assume_avg_completion_tokens,
            "assumed_completion_tokens": assumed_completion,
            "assumed_total_tokens": total["prompt_tokens"] + assumed_completion,
            "gpu_hours": round((total["prompt_tokens"] + assumed_completion)
                               / args.assume_tokens_per_second / 3600, 3)
            if args.assume_tokens_per_second else None,
            "wall_hours": round((total["prompt_tokens"] + assumed_completion)
                                / args.assume_tokens_per_second
                                / args.assume_gpus / 3600, 3)
            if args.assume_tokens_per_second else None,
            "cost_usd": round((total["prompt_tokens"] + assumed_completion)
                              / args.assume_tokens_per_second / 3600
                              * args.price_per_gpu_hour, 2)
            if args.assume_tokens_per_second else None,
        }
    return summary


def summarize_datasets(input_paths, data_dir, parts_root, args):
    all_records = []
    all_input_rows = []
    profiles = []
    coverage = {}
    for input_path in input_paths:
        input_path = Path(input_path)
        output_path = Path(data_dir) / f"self_teacher_{input_path.stem}.jsonl"
        if not output_path.exists():
            log(f"[summary] missing {output_path}; skipping")
            coverage[input_path.stem] = {
                "input_rows": len(read_jsonl(input_path)),
                "output_rows": 0,
                "missing": len(read_jsonl(input_path)),
                "missing_ids": [row["id"] for row in read_jsonl(input_path)][:100],
                "extra": 0,
            }
            continue
        records = read_jsonl(output_path)
        input_rows = read_jsonl(input_path)
        input_ids = [row["id"] for row in input_rows]
        output_ids = {record["id"] for record in records}
        missing = [ident for ident in input_ids if ident not in output_ids]
        extra = sorted(output_ids - set(input_ids))
        coverage[input_path.stem] = {
            "input_rows": len(input_rows),
            "output_rows": len(records),
            "missing": len(missing),
            "missing_ids": missing[:100],
            "extra": len(extra),
            "extra_ids": extra[:100],
        }
        all_records.extend(records)
        all_input_rows.extend(input_rows)
        split_parts = Path(parts_root) / input_path.stem
        if split_parts.exists():
            profile_paths = sorted(split_parts.glob("profile-*.json"))
            profile_paths += sorted(split_parts.glob("part-*.profile.json"))
            for profile_path in profile_paths:
                try:
                    profiles.append(json.loads(profile_path.read_text()))
                except json.JSONDecodeError:
                    log(f"[summary] ignoring corrupt profile {profile_path}")
    summary = summarize(all_records, all_input_rows, profiles, args)
    summary["coverage"] = coverage
    return summary


def merge_all(args):
    parts_root = Path(args.parts_root)
    data_dir = Path(args.data_dir)
    invalid_dir = Path(args.invalid_dir)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    input_paths = []
    for split_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        split = split_dir.name
        input_path = data_dir / f"{split}.jsonl"
        output_path = data_dir / f"self_teacher_{split}.jsonl"
        invalid_log = invalid_dir / f"ms_self_teacher_invalid_{split}.jsonl"
        merge_parts(split_dir, output_path, invalid_log=invalid_log,
                    input_path=input_path if input_path.exists() else None)
        if input_path.exists():
            input_paths.append(input_path)
    if not input_paths:
        raise SystemExit(f"no split inputs found under {data_dir}")
    summary = summarize_datasets(input_paths, data_dir, parts_root, args)
    atomic_write_json(args.summary, summary)
    log(f"[summary] wrote {args.summary}")
    return 0


# ---------------------------------------------------------------------------
# CPU modes
# ---------------------------------------------------------------------------
def validate_only(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    report = {"model": str(args.model), "inputs": {}, "checked_at":
              time.strftime("%Y-%m-%dT%H:%M:%S"), "two_stage": bool(args.two_stage)}
    if args.two_stage:
        report["thinking_max_tokens"] = args.thinking_max_tokens
        report["answer_max_tokens"] = args.answer_max_tokens
    generation_budget = args.thinking_max_tokens if args.two_stage else args.max_tokens
    stage2_budget = args.max_model_len - args.answer_max_tokens
    failed = False
    for path in args.input:
        path = Path(path)
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[: args.limit]
        lengths = []
        stage2_lengths = []
        by_source = defaultdict(list)
        errors = []
        samples = []
        for position, row in enumerate(rows):
            try:
                validate_row(row)
            except ValueError as error:
                errors.append({"row": position, "error": str(error)})
                failed = True
                continue
            prompt = make_prompt(tokenizer, row)
            length = prompt_length(tokenizer, prompt)
            lengths.append(length)
            by_source[row["source"]].append(length)
            if args.two_stage:
                # Overhead of the stage-2 prompt with empty reasoning; the real
                # prompt adds at most --thinking-max-tokens of reasoning and is
                # truncated by fit_stage2_prompt when it does not fit.
                stage2_lengths.append(prompt_length(
                    tokenizer, make_stage2_prompt(tokenizer, row, "")))
            if len(samples) < 3:
                sample = {"id": row["id"], "source": row["source"],
                          "lang": row.get("lang"), "tokens": length,
                          "prompt": prompt}
                if args.two_stage:
                    sample["stage2_prompt_empty_reasoning"] = make_stage2_prompt(
                        tokenizer, row, "")
                samples.append(sample)
        entry = {
            "rows": len(rows),
            "invalid_rows": errors[:20],
            "invalid_row_count": len(errors),
            "by_source": {
                source: {"n": len(values), **_token_stats(values)}
                for source, values in sorted(by_source.items())
            },
            "token_lengths": _token_stats(lengths),
            "over_budget": sum(1 for length in lengths
                               if length > args.max_model_len - generation_budget),
            "prompt_samples": samples,
        }
        if args.two_stage:
            entry["stage2_token_lengths_empty_reasoning"] = _token_stats(stage2_lengths)
            entry["stage2_over_budget_empty_reasoning"] = sum(
                1 for length in stage2_lengths if length > stage2_budget)
        report["inputs"][str(path)] = entry
    atomic_write_json(args.report, report)
    log(f"wrote {args.report}")
    return 1 if failed else 0


def estimate(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    report = {
        "model": str(args.model),
        "samples_per_question": args.samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "assumed_avg_completion_tokens": args.assume_avg_completion_tokens,
        "assumed_tokens_per_second_per_gpu": args.assume_tokens_per_second,
        "assumed_gpus": args.assume_gpus,
        "assumed_price_usd_per_gpu_hour": args.price_per_gpu_hour,
        "inputs": {},
    }
    total_rows = 0
    total_prompt_tokens = 0
    for path in args.input:
        path = Path(path)
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[: args.limit]
        lengths = []
        by_source = defaultdict(list)
        for row in rows:
            validate_row(row)
            length = len(tokenizer(make_prompt(tokenizer, row),
                                   add_special_tokens=False)["input_ids"])
            lengths.append(length)
            by_source[row["source"]].append(length)
        prompt_tokens = sum(lengths)
        completion_tokens = len(rows) * args.samples * args.assume_avg_completion_tokens
        total_tokens = prompt_tokens + completion_tokens
        gpu_hours = total_tokens / args.assume_tokens_per_second / 3600
        total_rows += len(rows)
        total_prompt_tokens += prompt_tokens
        report["inputs"][str(path)] = {
            "rows": len(rows),
            "by_source": {
                source: {"n": len(values), **_token_stats(values)}
                for source, values in sorted(by_source.items())
            },
            "token_lengths": _token_stats(lengths),
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "estimated_total_tokens": total_tokens,
            "estimated_gpu_hours": round(gpu_hours, 3),
            "estimated_wall_hours": round(gpu_hours / args.assume_gpus, 3),
            "estimated_cost_usd": round(gpu_hours * args.price_per_gpu_hour, 2),
        }
    total_completion = total_rows * args.samples * args.assume_avg_completion_tokens
    total_tokens = total_prompt_tokens + total_completion
    report["totals"] = {
        "rows": total_rows,
        "estimated_prompt_tokens": total_prompt_tokens,
        "estimated_completion_tokens": total_completion,
        "estimated_total_tokens": total_tokens,
        "estimated_gpu_hours": round(
            total_tokens / args.assume_tokens_per_second / 3600, 3),
        "estimated_wall_hours": round(
            total_tokens / args.assume_tokens_per_second / args.assume_gpus / 3600, 3),
        "estimated_cost_usd": round(
            total_tokens / args.assume_tokens_per_second / 3600
            * args.price_per_gpu_hour, 2),
    }
    report["notes"] = [
        "No GPU generation was run; this is a projection from prompt token counts.",
        f"Assumes {args.assume_avg_completion_tokens} completion tokens per chain and "
        f"{args.assume_tokens_per_second} tokens/s per GPU.",
        "The merge step overwrites these estimates with observed profile numbers.",
    ]
    atomic_write_json(args.report, report)
    log(f"wrote {args.report}")
    return 0


def write_parser_selftest(args):
    records, failures = parser_selftest()
    atomic_write_json(args.report, {
        "cases": records,
        "passed": len(records) - len(failures),
        "failed": len(failures),
    })
    print(f"parser selftest: {len(records) - len(failures)}/{len(records)} passed")
    for expected, got, raw, meta in failures:
        print(f"FAIL expected={expected} got={got} raw={raw!r} meta={meta}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Self-teacher soft-target generator for multi-select decisions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", action="append", default=[],
                        help="input JSONL; repeat for train+dev")
    parser.add_argument("--output", action="append", default=[],
                        help="output JSONL; one per --input (default: "
                             "self_teacher_<stem>.jsonl next to the input)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2,
                        help="independent thinking chains per question")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="prompts per vLLM generate call")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking-token-budget", type=int, default=0,
                        help="legacy vLLM thinking budget (0 = disabled); broken on "
                             "vLLM 0.19.1 without --reasoning-config, use --two-stage")
    parser.add_argument("--two-stage", action="store_true",
                        help="sample thinking (stop at </think>), then force a short "
                             "greedy answer pass; supersedes --thinking-token-budget")
    parser.add_argument("--thinking-max-tokens", type=int, default=1024,
                        help="stage-1 thinking cap in two-stage mode")
    parser.add_argument("--answer-max-tokens", type=int, default=64,
                        help="stage-2 answer cap in two-stage mode")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true",
                        help="skip ids already present in the output part file")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N rows of each input (smoke tests)")
    parser.add_argument("--progress-every", type=int, default=200)

    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--parser-selftest", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--merge-all", action="store_true")

    parser.add_argument("--report", type=Path, default=None,
                        help="JSON report for --validate-only/--parser-selftest/--estimate")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--parts-root", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--invalid-dir", type=Path, default=DEFAULT_INVALID_DIR)
    parser.add_argument("--assume-tokens-per-second", type=float, default=2500.0,
                        help="per-GPU throughput used only for cost estimates")
    parser.add_argument("--assume-gpus", type=int, default=8,
                        help="GPU count used only for wall-clock estimates")
    parser.add_argument("--assume-price-per-gpu-hour", type=float, default=2.5,
                        dest="price_per_gpu_hour")
    parser.add_argument("--assume-avg-completion-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    if args.shards < 1:
        parser.error("--shards must be >= 1")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.max_tokens < 1 or args.max_model_len < 1:
        parser.error("token limits must be positive")
    if args.max_tokens >= args.max_model_len:
        parser.error("--max-tokens must be smaller than --max-model-len")
    if args.thinking_token_budget and args.thinking_token_budget >= args.max_tokens:
        parser.error("--thinking-token-budget must be smaller than --max-tokens")
    if args.two_stage:
        if args.thinking_token_budget:
            parser.error("--thinking-token-budget is broken on vLLM 0.19.1 "
                         "(reasoning_config required); use --two-stage instead")
        if args.thinking_max_tokens < 1 or args.answer_max_tokens < 1:
            parser.error("--thinking-max-tokens/--answer-max-tokens must be positive")
        if args.thinking_max_tokens >= args.max_model_len:
            parser.error("--thinking-max-tokens must be smaller than --max-model-len")
        if args.answer_max_tokens >= args.max_model_len:
            parser.error("--answer-max-tokens must be smaller than --max-model-len")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")

    if not args.output and args.input:
        args.output = [str(Path(path).parent / f"self_teacher_{Path(path).stem}.jsonl")
                       for path in args.input]
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.check_model:
        shards = check_model_complete(args.model)
        print(f"model complete: {args.model} ({len(shards)} safetensors shards)")
        return 0
    if args.parser_selftest:
        if args.report is None:
            args.report = ROOT / "logs/cpu_checks/ms_teacher_parser_cases.json"
        return write_parser_selftest(args)
    if args.validate_only:
        if args.report is None:
            args.report = ROOT / "logs/cpu_checks/ms_teacher_validate.json"
        if not args.input:
            raise SystemExit("--validate-only requires --input")
        return validate_only(args)
    if args.estimate:
        if args.report is None:
            args.report = ROOT / "logs/cpu_checks/ms_teacher_estimate.json"
        if not args.input:
            raise SystemExit("--estimate requires --input")
        return estimate(args)
    if args.merge_all:
        return merge_all(args)
    return run_generation(args)


if __name__ == "__main__":
    sys.exit(main())
