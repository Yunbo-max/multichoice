"""Score JEVal predictions (SII or StartLux format) against the converted gold labels.

Metrics follow the JEVal data card: pooled accuracy over choice+noul, ECE with 15
equal-width bins, and raw multiclass Brier. Score questions are reported
separately as NMAE because the paper excludes them from the 10,938-item table.

Prediction line formats accepted:
  SII:      {"id","type","probs":[...]} or {"id","type","probabilities":{...}}
  StartLux: {"id","type","probabilities":{...}} or {"id","type","noul":p_true}

Usage:
  python3 -I score_jeval.py --input ../data/jeval/sii_input.jsonl \
      --gold ../data/jeval/gold.json --pred preds.jsonl --out summary.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = {}
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def is_true_key(key):
    return key.strip().lower() in ("true", "yes")


def probs_for(pred, item, n):
    if isinstance(pred.get("probs"), list) and len(pred["probs"]) == n:
        return [float(x) for x in pred["probs"]]
    if isinstance(pred.get("probabilities"), dict):
        options = item["options"]
        return [float(pred["probabilities"][o["key"]]) for o in options]
    if item["type"] == "noul":
        p_true = pred.get("noul", pred.get("p_true"))
        if p_true is None:
            raise KeyError("noul prediction has neither probs, probabilities nor noul")
        options = item["options"]
        return [float(p_true) if is_true_key(o["key"]) else 1.0 - float(p_true) for o in options]
    raise KeyError("prediction has no probability vector")


def aggregate(rows):
    """rows: valid records with type/probs/gold_index."""
    cn = [r for r in rows if r["type"] in ("choice", "noul")]
    out = {"n": len(rows), "n_choice_noul": len(cn)}
    if cn:
        out["accuracy"] = sum(r["correct"] for r in cn) / len(cn)
        bins = [[] for _ in range(15)]
        for r in cn:
            bins[min(14, int(r["confidence"] * 15))].append(r)
        out["ece_15bin"] = sum(
            len(b) / len(cn) * abs(sum(r["confidence"] for r in b) / len(b)
                                   - sum(r["correct"] for r in b) / len(b))
            for b in bins if b)
        out["brier_raw"] = sum(r["brier"] for r in cn) / len(cn)
    scores = [r for r in rows if r["type"] == "score"]
    if scores:
        out["score_nmae"] = sum(r["nmae"] for r in scores) / len(scores)
        out["score_exact"] = sum(r["exact"] for r in scores) / len(scores)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    items = load_jsonl(args.input)
    gold = json.loads(args.gold.read_text())
    preds = load_jsonl(args.pred)
    missing = sorted(set(gold) - set(preds))

    records, errors = [], []
    for ident, meta in gold.items():
        if ident not in preds:
            continue
        item = items[ident]
        n = len(item["options"])
        try:
            probs = probs_for(preds[ident], item, n)
        except Exception as error:
            errors.append({"id": ident, "error": str(error)})
            continue
        if len(probs) != n or any(not math.isfinite(p) or p < 0 for p in probs):
            errors.append({"id": ident, "error": f"bad probability vector (n={n})"})
            continue
        total = sum(probs)
        if total <= 0:
            errors.append({"id": ident, "error": "probability sum is zero"})
            continue
        probs = [p / total for p in probs]
        g = meta["gold_index"]
        pred_i = max(range(n), key=probs.__getitem__)
        record = {
            "id": ident, "type": item["type"], "source": meta["source"],
            "n_options": n, "gold_index": g, "pred_index": pred_i,
            "correct": int(pred_i == g),
            "confidence": probs[pred_i],
            "brier": sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(probs)),
        }
        if item["type"] == "score":
            expected = sum(i * p for i, p in enumerate(probs))
            record["expected"] = expected
            record["nmae"] = abs(expected - g) / (n - 1) if n > 1 else 0.0
            record["exact"] = int(pred_i == g)
        records.append(record)

    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    summary = {
        "overall": aggregate(records),
        "by_type": {t: aggregate([r for r in records if r["type"] == t])
                    for t in ("choice", "noul", "score")},
        "by_source": {s: aggregate(rows) for s, rows in sorted(by_source.items())},
        "missing_predictions": len(missing),
        "invalid_predictions": len(errors),
        "errors": errors[:20],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")

    o = summary["overall"]
    print(f"choice+noul n={o['n_choice_noul']}  acc={100*o['accuracy']:.2f}%  "
          f"ECE15={o['ece_15bin']:.4f}  Brier={o['brier_raw']:.4f}")
    if "score_nmae" in o:
        print(f"score n={summary['by_type']['score']['n']}  NMAE={o['score_nmae']:.4f}  "
              f"exact={100*o['score_exact']:.2f}%")
    print(f"missing={len(missing)}  invalid={len(errors)}")
    if args.out:
        print(f"full summary: {args.out}")


if __name__ == "__main__":
    main()
