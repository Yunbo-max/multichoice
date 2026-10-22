"""CPU smoke test for SII-Decision inference, reimplemented from the documented recipe.

This does NOT import or execute code from the downloaded repository. It loads the
released standalone weights, renders the documented prompt, reads the option-label
logits at the last prompt position, and prints probabilities.

Usage:
  ../.venv-sii/bin/python -I sii_smoke.py --model ../models/SII-4B-LoRA \
      --input ../data/jeval/sii_input.jsonl --items 3
"""
import argparse
import json
import string

import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

CUE = "The correct answer is ("


def option_labels(tokenizer):
    candidates = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                                 for b in string.ascii_uppercase]
    labels = [label for label in candidates
              if len(tokenizer.encode(label, add_special_tokens=False)) == 1][:255]
    if len(labels) != 255:
        raise ValueError(f"tokenizer provides only {len(labels)} single-token labels")
    return labels


def render(tokenizer, row, labels):
    lines = [f"({labels[i]}) {o['key']}: {o['desc']}" for i, o in enumerate(row["options"])]
    body = (f"State:\n{row['state']}\nQuestion: {row['question']}\nOptions:\n"
            + "\n".join(lines))
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return prefix + CUE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--items", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    labels = option_labels(tokenizer)
    label_ids = torch.tensor([tokenizer.encode(l, add_special_tokens=False)[0] for l in labels])
    config = AutoConfig.from_pretrained(args.model)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else Qwen3_5ForCausalLM
    print(f"loading {args.model} ({config.model_type}) on CPU ...", flush=True)
    model = cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu",
                                attn_implementation="sdpa")
    model.eval()
    backbone = getattr(model.model, "language_model", model.model)
    head = model.get_output_embeddings()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    # shortest item per type keeps the smoke test quick
    chosen = []
    for qtype in ("choice", "noul", "score"):
        pool = [r for r in rows if r["type"] == qtype]
        if pool:
            chosen.append(min(pool, key=lambda r: len(r["state"]) + sum(len(o["desc"]) for o in r["options"])))
    chosen = chosen[:args.items]

    for row in chosen:
        text = render(tokenizer, row, labels)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            hidden = backbone(**enc, use_cache=False, return_dict=True).last_hidden_state
            slot = hidden[0, int(enc["attention_mask"].sum()) - 1]
            logits = head(slot).float().index_select(-1, label_ids)
        n = len(row["options"])
        probs = logits[:n].softmax(-1).tolist()
        pred = max(range(n), key=probs.__getitem__)
        print(json.dumps({
            "id": row["id"], "type": row["type"], "n_options": n,
            "pred": pred, "gold": row["gold_index"], "correct": pred == row["gold_index"],
            "probs": probs,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
