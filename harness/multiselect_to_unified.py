"""Build a unified multi-select dataset from CMB-Exam (Chinese, multiple-answer)
and MultiRC (English, multi-sentence candidate labels).

Output rows (JSONL):
  {"id","source","lang","split","state","question",
   "options":[{"key","desc"}],"target":[0/1,...],"n_options","gold_letters"}

The target is a multi-hot vector; it is NOT normalized to sum to one.
Splits:
  train: CMB train 多项选择题 + MultiRC train, minus a stratified dev slice
  dev:   ~2,000 questions from the train pools (plus CMB val 多选)
  test:  CMB test 多项选择题 + MultiRC validation (MultiRC test labels are hidden)

Usage:
  python3 -I multiselect_to_unified.py --raw-dir ../data/multiselect/raw \
      --out-dir ../data/multiselect --dev-size 2000
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def row_key(row):
    return norm(row["question"]) + " || " + norm(row["state"])[:200]


def parse_letters(answer, keys):
    """Return sorted unique letters of answer if it is a pure letter string over keys."""
    s = str(answer).strip().upper().replace(" ", "").replace(",", "").replace("，", "")
    if not s or not re.fullmatch(r"[A-Z]+", s):
        return None
    letters = sorted(set(s), key=lambda c: keys.index(c) if c in keys else 99)
    if any(c not in keys for c in letters):
        return None
    return letters


def make_row(ident, source, lang, split, state, question, options, target, gold_letters=None):
    return {
        "id": ident,
        "source": source,
        "lang": lang,
        "split": split,
        "state": state,
        "question": question,
        "options": options,
        "target": target,
        "n_options": len(options),
        "gold_letters": gold_letters,
    }


def convert_cmb(rows, source, split, prefix, dropped):
    out = []
    for index, row in enumerate(rows):
        if row.get("question_type") != "多项选择题":
            continue
        options = row.get("option") or {}
        keys = sorted(options.keys())
        if not 2 <= len(keys) <= 26:
            dropped["bad_option_count"] += 1
            continue
        letters = parse_letters(row.get("answer"), keys)
        if not letters:
            dropped["unparseable_answer"] += 1
            continue
        target = [1 if k in letters else 0 for k in keys]
        if sum(target) == 0:
            dropped["empty_target"] += 1
            continue
        opt_rows = [{"key": k, "desc": norm(options[k])} for k in keys]
        state = "；".join(str(row.get(k, "")) for k in ("exam_type", "exam_class", "exam_subject")
                          if row.get(k))
        ident = f"{prefix}:{row.get('id', index)}"
        out.append(make_row(ident, source, "zh", split, state, norm(row.get("question", "")),
                            opt_rows, target, "".join(letters)))
    return out


def convert_cmb_test(questions, answers, dropped):
    answer_by_id = {a.get("id"): a.get("answer") for a in answers}
    rows = []
    for q in questions:
        if q.get("question_type") != "多项选择题":
            continue
        q = dict(q)
        q["answer"] = answer_by_id.get(q.get("id"))
        rows.append(q)
    return convert_cmb(rows, "cmb_exam_multi", "test", "cmb_test", dropped)


def convert_multirc(parquet_path, split, dropped):
    table = pq.read_table(parquet_path).to_pylist()
    groups = {}
    for row in table:
        idx = row["idx"]
        if isinstance(idx, dict):
            key = (idx.get("paragraph"), idx.get("question"))
        else:
            key = (str(idx), str(row.get("question")))
        groups.setdefault(key, []).append(row)
    out = []
    for qi, (key, candidates) in enumerate(groups.items()):
        if len(candidates) < 2:
            dropped["multirc_too_few_candidates"] += 1
            continue
        if len(candidates) > 255:
            dropped["multirc_too_many_candidates"] += 1
            continue
        labels = [int(c.get("label", 0)) for c in candidates]
        if sum(labels) == 0:
            dropped["multirc_all_negative"] += 1
            continue
        options = [{"key": LETTERS[i], "desc": norm(c.get("answer", ""))}
                   for i, c in enumerate(candidates)]
        first = candidates[0]
        out.append(make_row(f"multirc:{split}:{qi}", "multirc", "en", split,
                            norm(first.get("paragraph", "")), norm(first.get("question", "")),
                            options, labels))
    return out


def stratified_dev(rows, dev_size, seed):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    total = len(rows)
    if dev_size >= total:
        raise SystemExit("dev_size must be smaller than the training pool")
    alloc = {}
    for source, values in by_source.items():
        alloc[source] = max(1, round(dev_size * len(values) / total))
    # adjust to exactly dev_size
    while sum(alloc.values()) > dev_size:
        source = max(alloc, key=lambda s: alloc[s] - 1)
        alloc[source] -= 1
    while sum(alloc.values()) < dev_size:
        source = max(by_source, key=lambda s: len(by_source[s]) - alloc[s])
        alloc[source] += 1
    rng = random.Random(seed)
    dev, train = [], []
    for source, values in by_source.items():
        values = list(values)
        rng.shuffle(values)
        n = alloc[source]
        dev.extend(values[:n])
        train.extend(values[n:])
    return train, dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sii-train", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dropped = Counter()

    cmb_train = json.loads((args.raw_dir / "cmb_train.json").read_text())
    cmb_val = json.loads((args.raw_dir / "cmb_val.json").read_text())
    cmb_test = json.loads((args.raw_dir / "cmb_test.json").read_text())
    cmb_test_answers = json.loads((args.raw_dir / "cmb_test_answers.json").read_text())

    train_pool = []
    train_pool += convert_cmb(cmb_train, "cmb_exam_multi", "train", "cmb_train", dropped)
    train_pool += convert_multirc(args.raw_dir / "multirc_train.parquet", "train", dropped)
    dev_extra = convert_cmb(cmb_val, "cmb_exam_multi", "dev", "cmb_val", dropped)
    test = convert_cmb_test(cmb_test, cmb_test_answers, dropped)
    test += convert_multirc(args.raw_dir / "multirc_validation.parquet", "test", dropped)

    seen = set()
    deduped = []
    for row in train_pool:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    train_pool = deduped
    train, dev = stratified_dev(train_pool, args.dev_size, args.seed)
    dev += dev_extra
    dev_keys = {row_key(row) for row in dev}
    train = [row for row in train if row_key(row) not in dev_keys]
    for row in train:
        row["split"] = "train"
    for row in dev:
        row["split"] = "dev"

    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(rows):
        return {
            "n": len(rows),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "by_lang": dict(Counter(r["lang"] for r in rows)),
            "option_counts": dict(sorted(Counter(r["n_options"] for r in rows).items())),
            "positive_counts": dict(sorted(Counter(sum(r["target"]) for r in rows).items())),
        }

    summary = {
        "train": stats(train), "dev": stats(dev), "test": stats(test),
        "dropped": dict(dropped),
        "max_options": max(r["n_options"] for r in train + dev + test),
    }

    # contamination / split overlap checks
    key = row_key

    train_keys = {key(r) for r in train}
    dev_keys = {key(r) for r in dev}
    summary["overlap"] = {
        "train_dev": len(train_keys & dev_keys),
        "train_test": len(train_keys & {key(r) for r in test}),
        "dev_test": len(dev_keys & {key(r) for r in test}),
    }
    if args.sii_train and args.sii_train.exists():
        sii = [json.loads(l) for l in args.sii_train.open() if l.strip()]
        cmb_sii = [r for r in sii if r.get("source") in ("cmb", "cmexam")]
        sii_keys = {norm(r.get("state", "")) for r in cmb_sii}
        sii_keys |= {norm(r.get("question", "")) for r in cmb_sii}
        cmb_test = [r for r in test if r["source"] == "cmb_exam_multi"]
        hits = sum(1 for r in cmb_test
                   if norm(r["state"]) in sii_keys or norm(r["question"]) in sii_keys)
        summary["sii_overlap"] = {"cmb_sii_rows": len(cmb_sii),
                                  "cmb_test_rows": len(cmb_test),
                                  "exact_question_or_state_hits": hits}

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
