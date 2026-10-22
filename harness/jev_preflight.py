"""CPU-only preflight for reproducing SII-Decision on JEVal.

Builds the documented SII prompt (chat template, thinking off, answer cue) with the
Qwen3.5-4B tokenizer and reports exact token lengths / option-label coverage.
This script does NOT execute any code from the downloaded repositories.

Usage:
  python3 -I jev_preflight.py --parquet PATH --model Qwen/Qwen3.5-4B
"""
import argparse
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from transformers import AutoTokenizer

CUE = "The correct answer is ("


def body(row, labels):
    q = row["question"]
    opts = q.get("criteria", q.get("options"))
    if opts is None:
        # JEVal noul items without explicit criteria: implicit true/false options
        items = [("true", ""), ("false", "")]
    elif isinstance(opts, dict):
        items = list(opts.items())
    elif isinstance(opts, str):
        items = [("true", ""), ("false", "")]
    else:
        items = [(str(i), v) for i, v in enumerate(opts)]
    lines = []
    for i, (k, v) in enumerate(items):
        desc = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        lines.append(f"({labels[i]}) {k}: {desc}")
    return (f"State:\n{row['state']}\nQuestion: {q.get('instructions', q.get('question', ''))}"
            f"\nOptions:\n" + "\n".join(lines))


def option_labels(tok):
    """Replicates the released code: A-Z then AA..ZZ, keep single-token labels, first 255."""
    import string
    candidates = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                                 for b in string.ascii_uppercase]
    labels = [l for l in candidates if len(tok.encode(l, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise SystemExit(f"tokenizer provides only {len(labels)} single-token labels")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    labels = option_labels(tok)
    print(f"tokenizer={args.model} vocab={tok.vocab_size} labels={labels[:5]}...{labels[-3:]} n={len(labels)}")

    df = pq.read_table(args.parquet).to_pandas()
    if args.limit:
        df = df.head(args.limit)
    lengths = []
    over = Counter()
    per_source = defaultdict(list)
    max_item = (0, None)
    for _, r in df.iterrows():
        q = json.loads(r["questions_json"])
        if len(q) != 1:
            raise SystemExit(f"expected one question per JEVal item, got {len(q)}")
        name, spec = next(iter(q.items()))
        row = {"state": r["state"], "question": {"type": spec.get("type"),
                                                  "instructions": spec.get("instructions", spec.get("question", "")),
                                                  "criteria": spec.get("criteria", spec.get("options"))}}
        text = tok.apply_chat_template([{"role": "user", "content": body(row, labels)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + CUE
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        per_source[r["source"]].append(n)
        if n > 6000:
            over[r["source"]] += 1
        if n > max_item[0]:
            max_item = (n, r["id"])
    lengths.sort()
    print(f"n={len(lengths)} mean={sum(lengths)/len(lengths):.0f} p50={lengths[len(lengths)//2]} "
          f"p90={lengths[int(.9*len(lengths))]} p95={lengths[int(.95*len(lengths))]} "
          f"p99={lengths[int(.99*len(lengths))]} max={lengths[-1]} ({max_item[1]})")
    n_over = sum(over.values())
    print(f">6000 tokens: {n_over} ({100*n_over/len(lengths):.2f}%)")
    print("top sources over 6000:", over.most_common(12))
    print("longest sources (mean tokens):",
          sorted(((k, sum(v)/len(v)) for k, v in per_source.items()), key=lambda x: -x[1])[:10])


if __name__ == "__main__":
    main()
