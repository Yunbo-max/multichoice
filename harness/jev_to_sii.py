"""Convert JEVal parquet into the JSONL shape SII-Decision expects.

Does not execute any downloaded repository code. Option order is preserved;
gold labels are converted to the index of the correct option in that order.

Outputs:
  sii_input.jsonl   one row per JEVal item: id/type/state/question/options + gold_index
  gold.json         id -> {type, gold_index, n_options, source, source_item_id}
  summary.json      conversion statistics and any label anomalies

Usage:
  python3 -I jev_to_sii.py --parquet ../data/jeval/data/test.parquet \
      --out-dir ../data/jeval
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def convert(row):
    questions = json.loads(row["questions_json"])
    if len(questions) != 1:
        raise ValueError(f"{row['id']}: expected exactly one question, got {len(questions)}")
    _, spec = next(iter(questions.items()))
    qtype = spec.get("type")
    instruction = spec.get("instructions", spec.get("question", ""))
    criteria = spec.get("criteria", None)
    label = json.loads(row["label_json"])

    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"{row['id']}: choice criteria must be a dict")
        items = list(criteria.items())
        keys = [str(k) for k, _ in items]
        if isinstance(label, str) and label in criteria:
            gold = keys.index(label)
        elif isinstance(label, int) and 0 <= label < len(items):
            gold = label
        else:
            raise ValueError(f"{row['id']}: cannot map choice label {label!r} to keys {keys}")
        options = [{"key": str(k), "desc": text(v)} for k, v in items]

    elif qtype == "noul":
        if isinstance(criteria, dict):
            items = list(criteria.items())
        else:
            # boolq/decider-style items carry no criteria. Their native label is
            # 0 = false, 1 = true, so present options in that order.
            items = [("false", ""), ("true", "")]
        options = [{"key": str(k), "desc": text(v)} for k, v in items]
        if not (isinstance(label, int) and 0 <= label < len(options)):
            raise ValueError(f"{row['id']}: noul label must index an option, got {label!r}")
        gold = label

    elif qtype == "score":
        if isinstance(criteria, dict):
            items = list(criteria.items())
            keys = [str(k) for k, _ in items]
            gold = keys.index(str(label)) if isinstance(label, str) else int(label)
        else:
            items = list(enumerate(criteria))
            gold = int(label)
        options = [{"key": str(k), "desc": text(v)} for k, v in items]

    else:
        raise ValueError(f"{row['id']}: unknown type {qtype!r}")

    if not 2 <= len(options) <= 255:
        raise ValueError(f"{row['id']}: {len(options)} options is outside 2..255")
    if len({o['key'] for o in options}) != len(options):
        raise ValueError(f"{row['id']}: duplicate option keys")
    if not 0 <= gold < len(options):
        raise ValueError(f"{row['id']}: gold index {gold} outside 0..{len(options)-1}")

    return {
        "id": row["id"],
        "type": qtype,
        "state": row["state"],
        "question": instruction,
        "options": options,
        "gold_index": gold,
        "source": row["source"],
        "source_item_id": row["source_item_id"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(args.parquet).to_pylist()
    converted, errors = [], []
    for row in rows:
        try:
            converted.append(convert(row))
        except Exception as error:
            errors.append({"id": row["id"], "error": str(error)})

    sii_path = args.out_dir / "sii_input.jsonl"
    with sii_path.open("w", encoding="utf-8") as stream:
        for row in converted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    gold = {
        row["id"]: {
            "type": row["type"],
            "gold_index": row["gold_index"],
            "n_options": len(row["options"]),
            "source": row["source"],
            "source_item_id": row["source_item_id"],
        }
        for row in converted
    }
    (args.out_dir / "gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    summary = {
        "rows_in": len(rows),
        "rows_out": len(converted),
        "errors": errors,
        "types": dict(Counter(r["type"] for r in converted)),
        "sources": len({r["source"] for r in converted}),
        "option_counts": dict(sorted(Counter(len(r["options"]) for r in converted).items())),
        "max_options": max((len(r["options"]) for r in converted), default=0),
        "gold_histogram": {
            t: dict(sorted(Counter(r["gold_index"] for r in converted if r["type"] == t).items()))
            for t in ("choice", "noul", "score")
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
