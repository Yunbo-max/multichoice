#!/usr/bin/env python3
"""Pre-cache Qwen3.5 multimodal processor features for the Valen Sokoban rows.

The per-image CPU preprocessing (resize/normalize/patch) is the throughput
bottleneck of ``train_visual_action.py``. This script runs that preprocessing
once and saves the resulting tensors, so training can skip it entirely.

Run once on a single-GPU job (CPU-bound, so the GPU is only used to keep the
image available), e.g.::

    python -m treejev.cache_valen_features \
        --jsonl runs/tree/valen_split/train_visual.jsonl \
        --model /path/to/Qwen3.5-4B \
        --output runs/tree/valen_cache_train \
        --chunk-size 1000
"""
import argparse
import json
from pathlib import Path

import torch

from treejev.data import read_jsonl

PROMPT = (
    "You are playing Sokoban. Walls block movement. Push boxes onto targets. "
    "You cannot pull boxes or push two boxes at once. "
    "Choose one best next move for the pictured board.\n"
    "Options: (A) up; (B) right; (C) down; (D) left.\n"
    "Output only the option letter."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=1000)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"output exists: {args.output}")
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    cue = processor.tokenizer("The correct answer is (",
                              add_special_tokens=False, return_tensors="pt")["input_ids"]
    rows = list(read_jsonl(args.jsonl))
    args.output.mkdir(parents=True)
    manifest = []
    for chunk_start in range(0, len(rows), args.chunk_size):
        chunk_rows = rows[chunk_start:chunk_start + args.chunk_size]
        pixel_values = []
        grid_thw = []
        input_ids = []
        attention_mask = []
        mm_token_type = []
        valid_ids = []
        for row in chunk_rows:
            with Image.open(row["image_path"]) as raw:
                image = raw.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}]}]
            features = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
                padding=True)
            features["input_ids"] = torch.cat((features["input_ids"], cue), dim=1)
            features["attention_mask"] = torch.cat(
                (features["attention_mask"], torch.ones_like(cue)), dim=1)
            if "mm_token_type_ids" in features:
                features["mm_token_type_ids"] = torch.cat(
                    (features["mm_token_type_ids"], torch.zeros_like(cue)), dim=1)
            pixel_values.append(features["pixel_values"].contiguous())
            grid_thw.append(features["image_grid_thw"][0])
            input_ids.append(features["input_ids"][0])
            attention_mask.append(features["attention_mask"][0])
            mm_token_type.append(features.get("mm_token_type_ids",
                                torch.zeros_like(features["input_ids"]))[0])
            valid_ids.append(row["id"])
        chunk_path = args.output / f"chunk_{chunk_start // args.chunk_size:04d}.pt"
        max_len = max(t.shape[0] for t in input_ids)
        pad = lambda t: torch.nn.functional.pad(
            t, (0, max_len - t.shape[0]), value=0)
        torch.save({
            "ids": valid_ids,
            "pixel_values": pixel_values,          # list of [patches_i, 1536]
            "image_grid_thw": torch.stack(grid_thw),  # [n, 3]
            "input_ids": torch.stack([pad(t) for t in input_ids]),
            "attention_mask": torch.stack([pad(t) for t in attention_mask]),
            "mm_token_type_ids": torch.stack([pad(t) for t in mm_token_type]),
        }, chunk_path)
        manifest.append({"chunk": chunk_path.name, "n": len(chunk_rows),
                         "start": chunk_start})
        print(f"wrote {chunk_path.name} ({len(chunk_rows)} rows)", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"chunks": len(manifest), "rows": len(rows),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
