#!/usr/bin/env bash
# Recurrent SetJev pilot: ONLY gold-supervised head training on frozen Qwen features.
# No SII cloud submission, no external API and no implicit multi-GPU launch.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-runs/qz/grouped_bce_4gpu_s44/v1/final}"
MODEL_ID="${MODEL_ID:-grouped_bce_s44_final}"
CACHE="${CACHE:-runs/recurrent_cache}"
OUT="${OUT:-runs/recurrent}"
STAGE="${1:-smoke}"

case "$STAGE" in
  smoke)
    "$PYTHON_BIN" -m unittest discover -s tests -p test_recurrent_set_decision.py -v
    "$PYTHON_BIN" harness/run_recurrent_set.py smoke
    "$PYTHON_BIN" harness/toy_sequential_set.py --train-episodes 512 --eval-episodes 256 \
      --epochs 10 --horizon 3 --steps 4
    ;;
  cache)
    mkdir -p "$CACHE"
    "$PYTHON_BIN" harness/run_recurrent_set.py cache \
      --dataset data/multiselect_grouped_multi_5k/train.jsonl \
      --model "$MODEL" --model-identity "$MODEL_ID" \
      --output "$CACHE/train5k.pt" --device cuda --dtype bf16 --batch-size 1
    "$PYTHON_BIN" harness/run_recurrent_set.py cache \
      --dataset data/multiselect_grouped_multi/dev.jsonl \
      --model "$MODEL" --model-identity "$MODEL_ID" \
      --output "$CACHE/dev2042.pt" --device cuda --dtype bf16 --batch-size 1
    ;;
  train)
    for seed in 43 44 45; do
      for mode in set latent_only static; do
        "$PYTHON_BIN" harness/run_recurrent_set.py train \
          --train-cache "$CACHE/train5k.pt" --dev-cache "$CACHE/dev2042.pt" \
          --output "$OUT/${mode}_s${seed}" --feedback "$mode" \
          --steps 4 --epochs 5 --latent-dim 128 --seed "$seed" --device cuda
      done
    done
    ;;
  *)
    echo "usage: bash qz/run_recurrent_set_pilot.sh {smoke|cache|train}" >&2
    exit 2
    ;;
esac
