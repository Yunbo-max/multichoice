"""Build a new, passage-grouped CMB/MultiRC dataset without replacing old splits.

The legacy row schema is retained, with explicit ``selection_mode`` and
``allow_empty`` protocol fields. MultiRC additionally retains paragraph and
question identity. Its dev split keeps whole paragraphs together, matching by
either original group_id or full normalized passage text within the source.
CMB uses per-question groups, not shared exam metadata. Group sizes make the
requested dev row count approximate; summary.json records actual allocations.

Default CMB conversion remains 多项选择题. ``--cmb-question-mode mixed`` also
includes native 单项选择题, explicitly marked single. A one-positive multi
question remains multi. Targets are multi-hot and never normalized.

Usage (pyarrow is needed only to read the raw MultiRC parquet files)::

  python -I harness/multiselect_to_unified.py --raw-dir data/multiselect/raw \
      --out-dir data/multiselect_grouped --dev-size 2000
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPTION_KEYS = list(LETTERS) + [a + b for a in LETTERS for b in LETTERS]


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def row_key(row):
    """Source-scoped full normalized question and state; no prefix truncation."""
    return (str(row["source"]), norm(row["question"]), norm(row["state"]))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def group_tokens(row):
    """Aliases whose transitive union defines a leakage group.

    Passage fallback supports published rows that have no group_id. Source
    scoping avoids treating unrelated source identifiers as the same group.
    All full-question duplicates also belong to the same group.
    """
    source = str(row["source"])
    tokens = {_json([source, "question", row_key(row)])}
    if row.get("group_id") is not None:
        tokens.add(_json([source, "group_id", row["group_id"]]))
    # Exact-question deduplication must retain aliases from discarded rows:
    # another question can share one of those paragraph IDs.
    for alias in row.get("group_id_aliases", []):
        tokens.add(_json([source, "group_id", alias]))
    if source == "multirc" and norm(row.get("state", "")):
        tokens.add(_json([source, "passage", norm(row["state"])]))
    return tokens


def _group_indices(rows):
    """Union groups by all aliases, including transitive ID/text matches."""
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        for token in group_tokens(row):
            if token in owners:
                a, b = root(index), root(owners[token])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owners[token] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[root(index)].append(index)
    return list(groups.values())


def overlap_audit(splits):
    """Report full-question intersections and shared connected groups separately."""
    names = list(splits)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    questions = {name: {row_key(row) for row in rows} for name, rows in splits.items()}
    full = {f"{a}_{b}": len(questions[a] & questions[b]) for a, b in pairs}
    rows, membership = [], []
    for name, values in splits.items():
        rows.extend(values)
        membership.extend([name] * len(values))
    groups = {f"{a}_{b}": 0 for a, b in pairs}
    by_source = defaultdict(lambda: {key: 0 for key in groups})
    for indices in _group_indices(rows):
        present = {membership[index] for index in indices}
        for a, b in pairs:
            if a in present and b in present:
                groups[f"{a}_{b}"] += 1
                by_source[rows[indices[0]]["source"]][f"{a}_{b}"] += 1
    return {
        **full,  # retain the old flat overlap keys, now using full text
        "full_question": full,
        "group": groups,
        "group_by_source": dict(sorted(by_source.items())),
        "definitions": {
            "full_question": "source + full whitespace-normalized question + full state",
            "group": "connected shared group_id or full MultiRC passage within source; "
                     "CMB per-question groups",
            "scope": "exact normalized matches only; no semantic near-duplicate guarantee",
        },
    }


def parse_letters(answer, keys):
    """Return sorted unique letters of answer if it is a pure letter string over keys."""
    s = str(answer).strip().upper().replace(" ", "").replace(",", "").replace("，", "")
    if not s or not re.fullmatch(r"[A-Z]+", s):
        return None
    letters = sorted(set(s), key=lambda c: keys.index(c) if c in keys else 99)
    if any(c not in keys for c in letters):
        return None
    return letters


def make_row(ident, source, lang, split, state, question, options, target,
             gold_letters=None, selection_mode="multi", **metadata):
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
        "selection_mode": selection_mode,
        "allow_empty": False,
        **metadata,
    }


def convert_cmb(rows, source, split, prefix, dropped, question_mode="multi"):
    if question_mode not in ("multi", "mixed"):
        raise ValueError("question_mode must be multi or mixed")
    out = []
    for index, row in enumerate(rows):
        native_type = row.get("question_type")
        selection_mode = {"多项选择题": "multi", "单项选择题": "single"}.get(native_type)
        if selection_mode is None or (question_mode == "multi" and selection_mode != "multi"):
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
        if selection_mode == "single" and len(letters) != 1:
            dropped["single_with_multiple_answers"] += 1
            continue
        target = [1 if k in letters else 0 for k in keys]
        if sum(target) == 0:
            dropped["empty_target"] += 1
            continue
        opt_rows = [{"key": k, "desc": norm(options[k])} for k in keys]
        state = "；".join(str(row.get(k, "")) for k in ("exam_type", "exam_class", "exam_subject")
                          if row.get(k))
        ident = f"{prefix}:{row.get('id', index)}"
        converted = make_row(ident, source, "zh", split, state, norm(row.get("question", "")),
                             opt_rows, target, "".join(letters), selection_mode=selection_mode,
                             question_id=row.get("id", index), original_question_type=native_type)
        converted["group_id"] = "question:" + _digest(row_key(converted))
        out.append(converted)
    return out


def convert_cmb_test(questions, answers, dropped, question_mode="multi"):
    answer_by_id = {a.get("id"): a.get("answer") for a in answers}
    rows = []
    for q in questions:
        q = dict(q)
        q["answer"] = answer_by_id.get(q.get("id"))
        rows.append(q)
    return convert_cmb(rows, "cmb_exam_multi", "test", "cmb_test", dropped, question_mode)


def convert_multirc_rows(table, split, dropped, original_split=None):
    """Convert raw candidate rows; retain original question and paragraph IDs.

    The existing question IDs (multirc:<split>:<ordinal>) remain compatible with
    earlier conversion. Original upstream identity is recorded independently.
    Original split namespaces paragraph IDs because official partitions can
    reuse integers; matching full passages still detects cross-partition reuse.
    """
    original_split = original_split or ("validation" if split == "test" else split)
    groups = {}
    for row in table:
        idx = row.get("idx")
        if isinstance(idx, dict) and idx.get("paragraph") is not None and idx.get("question") is not None:
            key = ("idx", _json([idx["paragraph"], idx["question"]]))
        else:
            key = ("text", norm(row.get("paragraph", "")), norm(row.get("question", "")))
        groups.setdefault(key, []).append(row)
    out = []
    for qi, (key, candidates) in enumerate(groups.items()):
        if len(candidates) < 2:
            dropped["multirc_too_few_candidates"] += 1
            continue
        if len(candidates) > 255:
            dropped["multirc_too_many_candidates"] += 1
            continue
        raw_labels = [c.get("label") for c in candidates]
        if any(value not in (0, 1) for value in raw_labels):
            dropped["multirc_invalid_label"] += 1
            continue
        labels = [int(value) for value in raw_labels]
        if sum(labels) == 0:
            dropped["multirc_all_negative"] += 1
            continue
        options = [{"key": OPTION_KEYS[i], "desc": norm(c.get("answer", ""))}
                   for i, c in enumerate(candidates)]
        first = candidates[0]
        if any(norm(c.get("paragraph", "")) != norm(first.get("paragraph", ""))
               or norm(c.get("question", "")) != norm(first.get("question", ""))
               for c in candidates):
            dropped["multirc_inconsistent_question"] += 1
            continue
        idx = first.get("idx")
        idx = idx if isinstance(idx, dict) else {}
        paragraph_id, question_id = idx.get("paragraph"), idx.get("question")
        group_id = (f"multirc:{original_split}:paragraph:{_json(paragraph_id)}"
                    if paragraph_id is not None else
                    "multirc:passage:" + _digest(norm(first.get("paragraph", ""))))
        out.append(make_row(f"multirc:{split}:{qi}", "multirc", "en", split,
                            norm(first.get("paragraph", "")), norm(first.get("question", "")),
                            options, labels, group_id=group_id, question_id=question_id,
                            original_split=original_split,
                            original_idx={"paragraph": paragraph_id, "question": question_id}))
    return out


def convert_multirc(parquet_path, split, dropped):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("reading raw MultiRC parquet requires pyarrow") from error
    return convert_multirc_rows(pq.read_table(parquet_path).to_pylist(), split, dropped)


def stratified_dev(rows, dev_size, seed, return_report=False):
    """Deterministic source-stratified allocation of indivisible leakage groups.

    Largest-remainder row targets sum to dev_size. Seeded hash order and greedy
    absolute-error improvement choose groups near each target; one group per
    source is retained for training. Counts are approximate, not promised exact.
    No gold label or target cardinality participates in the allocation.
    """
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    total = len(rows)
    if not 0 <= dev_size < total:
        raise ValueError("dev_size must be nonnegative and smaller than the training pool")
    alloc = {source: dev_size * len(values) // total for source, values in by_source.items()}
    remainders = sorted(by_source, key=lambda source:
                        (-(dev_size * len(by_source[source]) % total), source))
    for source in remainders[:dev_size - sum(alloc.values())]:
        alloc[source] += 1
    dev, train = [], []
    source_report = {}
    for source, values in sorted(by_source.items()):
        values = sorted(values, key=lambda row: (row_key(row), str(row["id"])))
        groups = _group_indices(values)

        def order(indices):
            aliases = sorted({token for index in indices for token in group_tokens(values[index])})
            return _digest([seed, source, aliases])

        groups.sort(key=order)
        selected, count = set(), 0
        for index, group in enumerate(groups):
            if len(selected) >= len(groups) - 1:
                break
            if abs(count + len(group) - alloc[source]) < abs(count - alloc[source]):
                selected.add(index)
                count += len(group)
        for index, group in enumerate(groups):
            (dev if index in selected else train).extend(values[i] for i in group)
        source_report[source] = {
            "requested_dev_rows": alloc[source], "actual_dev_rows": count,
            "actual_train_rows": len(values) - count, "groups": len(groups),
            "dev_groups": len(selected), "train_groups": len(groups) - len(selected),
            "max_group_rows": max(map(len, groups)),
        }
    report = {"requested_dev_rows": dev_size, "actual_dev_rows": len(dev),
              "actual_train_rows": len(train), "by_source": source_report,
              "method": "source row quotas; deterministic whole-group approximate allocation",
              "seed": seed}
    return (train, dev, report) if return_report else (train, dev)


def _exclude_overlaps(rows, heldout):
    """Exclude entire connected groups intersecting a fixed held-out partition."""
    combined = list(rows) + list(heldout)
    excluded = set()
    for group in _group_indices(combined):
        if any(index >= len(rows) for index in group):
            excluded.update(index for index in group if index < len(rows))
    return [row for i, row in enumerate(rows) if i not in excluded], len(excluded)


def build_splits(train_pool, dev_extra, test, dev_size, seed):
    """Preserve official test, remove held-out overlap, then carve grouped dev."""
    representatives, deduped = {}, []
    for row in sorted(train_pool, key=lambda r: (row_key(r), str(r["id"]))):
        key = row_key(row)
        if key not in representatives:
            representative = dict(row)
            representatives[key] = representative
            deduped.append(representative)
        else:
            representative = representatives[key]
            aliases = list(representative.get("group_id_aliases", [])) + list(row.get("group_id_aliases", []))
            aliases.extend(value for value in (representative.get("group_id"), row.get("group_id"))
                           if value is not None)
            unique = {_json(value): value for value in aliases}
            representative["group_id_aliases"] = [unique[name] for name in sorted(unique)]
    report = {"deduplicated_train_rows": len(train_pool) - len(deduped)}
    dev_extra, report["excluded_dev_extra_test_rows"] = _exclude_overlaps(dev_extra, test)
    pool, report["excluded_train_holdout_rows"] = _exclude_overlaps(deduped, dev_extra + test)
    train, dev, allocation = stratified_dev(pool, dev_size, seed, return_report=True)
    train = [dict(row, split="train") for row in train]
    dev = [dict(row, split="dev") for row in dev + dev_extra]
    test = [dict(row, split="test") for row in test]
    report.update({"allocation": allocation, "additional_dev_rows": len(dev_extra),
                   "actual_dev_rows_with_extra": len(dev)})
    report["overlap"] = overlap_audit({"train": train, "dev": dev, "test": test})
    if any(report["overlap"]["group"].values()) or any(report["overlap"]["full_question"].values()):
        raise RuntimeError("split construction left a group or full-question overlap")
    return train, dev, test, report


def ensure_new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("--out-dir must be a new or empty directory; existing published data is never overwritten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sii-train", type=Path, default=None)
    parser.add_argument("--cmb-question-mode", choices=("multi", "mixed"), default="multi")
    args = parser.parse_args()
    ensure_new_output(args.out_dir)
    if args.out_dir.resolve() == args.raw_dir.resolve():
        parser.error("--out-dir must differ from --raw-dir")
    dropped = Counter()

    cmb_train = json.loads((args.raw_dir / "cmb_train.json").read_text())
    cmb_val = json.loads((args.raw_dir / "cmb_val.json").read_text())
    cmb_test = json.loads((args.raw_dir / "cmb_test.json").read_text())
    cmb_test_answers = json.loads((args.raw_dir / "cmb_test_answers.json").read_text())

    train_pool = []
    train_pool += convert_cmb(cmb_train, "cmb_exam_multi", "train", "cmb_train", dropped,
                              args.cmb_question_mode)
    train_pool += convert_multirc(args.raw_dir / "multirc_train.parquet", "train", dropped)
    dev_extra = convert_cmb(cmb_val, "cmb_exam_multi", "dev", "cmb_val", dropped,
                            args.cmb_question_mode)
    test = convert_cmb_test(cmb_test, cmb_test_answers, dropped, args.cmb_question_mode)
    test += convert_multirc(args.raw_dir / "multirc_validation.parquet", "test", dropped)

    train, dev, test, split_report = build_splits(
        train_pool, dev_extra, test, args.dev_size, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
            "selection_modes": dict(Counter(r.get("selection_mode", "multi") for r in rows)),
            "groups": len(_group_indices(rows)),
        }

    summary = {
        "train": stats(train), "dev": stats(dev), "test": stats(test),
        "dropped": dict(dropped),
        "max_options": max(r["n_options"] for r in train + dev + test),
        "split_policy": split_report,
        "cmb_question_mode": args.cmb_question_mode,
        "overlap": split_report["overlap"],
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

