#!/usr/bin/env bash
# Sharded SII-Decision evaluation launcher.
#
# One process per replica.  Each replica owns GPUS_PER_REPLICA visible GPUs and
# one data shard; shard i handles rows i, i+SHARDS, i+2*SHARDS, ...
# Part files and profiles are written under OUTDIR/parts, logs under OUTDIR/logs.
#
# Required environment:
#   MODEL    local model folder (e.g. .../models/SII-4B-LoRA)
#   OUTDIR   output folder (created if missing)
#
# Optional environment (defaults shown):
#   INPUT=.../data/jeval/sii_input.jsonl
#   GOLD=.../data/jeval/gold.json
#   SHARDS=8              number of replicas / data shards
#   GPUS_PER_REPLICA=1    >1 spreads one model across N GPUs (device_map=balanced)
#   BATCH_SIZE=8
#   MAX_BATCH_TOKENS=32768
#   MAX_INPUT_TOKENS=100000
#   RESUME=1              1: keep existing part files and skip done ids
#   LIMIT=0               0: full input; N>0: benchmark subset
#   LIMIT_MODE=head       head|stride
#   SCORE=1               1: run score_jeval.py after merging
#   PROGRESS_EVERY=200
#
# Example (8 replicas on 8 H100):
#   MODEL=$PWD/models/SII-27B-LoRA OUTDIR=$PWD/runs/sii_27b SHARDS=8 \
#   BATCH_SIZE=2 MAX_BATCH_TOKENS=16384 bash harness/run_sii_eval.sh
#
# Example (4 replicas x 2 GPUs, throughput comparison):
#   MODEL=$PWD/models/SII-4B-LoRA OUTDIR=$PWD/runs/sii_4b_tp2 SHARDS=4 \
#   GPUS_PER_REPLICA=2 BATCH_SIZE=8 MAX_BATCH_TOKENS=32768 bash harness/run_sii_eval.sh
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-sii/bin/python}"
EVAL="$ROOT/harness/sii_gpu_eval.py"
SCORER="$ROOT/harness/score_jeval.py"

: "${MODEL:?set MODEL to a local model folder}"
: "${OUTDIR:?set OUTDIR to an output folder}"
INPUT="${INPUT:-$ROOT/data/jeval/sii_input.jsonl}"
GOLD="${GOLD:-$ROOT/data/jeval/gold.json}"
SHARDS="${SHARDS:-8}"
GPUS_PER_REPLICA="${GPUS_PER_REPLICA:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-32768}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-100000}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-0}"
LIMIT_MODE="${LIMIT_MODE:-head}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
DEVICE="${DEVICE:-cuda}"
if [ -z "${SCORE:-}" ]; then
  if [ "$LIMIT" = "0" ]; then SCORE=1; else SCORE=0; fi
fi

PARTS="$OUTDIR/parts"
LOGS="$OUTDIR/logs"
mkdir -p "$PARTS" "$LOGS"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== SII eval ==="
echo "model=$MODEL"
echo "input=$INPUT"
echo "outdir=$OUTDIR"
echo "replicas=$SHARDS gpus_per_replica=$GPUS_PER_REPLICA batch=$BATCH_SIZE"
echo "max_batch_tokens=$MAX_BATCH_TOKENS max_input_tokens=$MAX_INPUT_TOKENS limit=$LIMIT"

"$PYTHON" -I "$EVAL" --check-model --model "$MODEL"

if [ "$SHARDS" -lt 1 ]; then
  echo "SHARDS must be >= 1" >&2
  exit 2
fi
if [ "$DEVICE" = "cpu" ] && { [ "$SHARDS" -ne 1 ] || [ "$GPUS_PER_REPLICA" -ne 1 ]; }; then
  echo "DEVICE=cpu supports SHARDS=1 GPUS_PER_REPLICA=1 only" >&2
  exit 2
fi
if [ "$DEVICE" = "cuda" ] && command -v nvidia-smi >/dev/null 2>&1; then
  available="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
  needed=$((SHARDS * GPUS_PER_REPLICA))
  if [ "${available:-0}" -gt 0 ] && [ "$needed" -gt "$available" ]; then
    echo "requested $needed GPUs but only $available visible" >&2
    exit 2
  fi
fi
if [ "$RESUME" != "1" ]; then
  rm -f "$PARTS"/part-*.jsonl "$PARTS"/part-*.jsonl.errors.jsonl "$PARTS"/profile-*.json
fi

common=(
  "$PYTHON" -I "$EVAL"
  --model "$MODEL"
  --input "$INPUT"
  --shards "$SHARDS"
  --batch-size "$BATCH_SIZE"
  --max-batch-tokens "$MAX_BATCH_TOKENS"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --prompt-style "${PROMPT_STYLE:-chat}"
  --model-parallel "$GPUS_PER_REPLICA"
  --device "$DEVICE"
  --gpu-index 0
  --limit "$LIMIT"
  --limit-mode "$LIMIT_MODE"
  --progress-every "$PROGRESS_EVERY"
  --profile-interval 0.5
)
if [ "$RESUME" = "1" ]; then
  common+=(--resume)
fi

pids=()
for replica in $(seq 0 $((SHARDS - 1))); do
  devices=""
  for slot in $(seq 0 $((GPUS_PER_REPLICA - 1))); do
    index=$((replica * GPUS_PER_REPLICA + slot))
    if [ -n "$devices" ]; then devices="$devices,$index"; else devices="$index"; fi
  done
  part="$(printf "%s/part-%05d.jsonl" "$PARTS" "$replica")"
  profile="$(printf "%s/profile-%05d.json" "$PARTS" "$replica")"
  log="$(printf "%s/shard-%05d.log" "$LOGS" "$replica")"
  echo "[launch] replica $replica gpus=$devices -> $part"
  CUDA_VISIBLE_DEVICES="$devices" \
  SII_PHYSICAL_GPU="$devices" \
    "${common[@]}" --shard "$replica" --output "$part" --profile "$profile" \
    --physical-gpu "$devices" > "$log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
echo "[launch] all replicas finished (fail=$fail)"

"$PYTHON" -I "$EVAL" --merge-parts "$PARTS" --output "$OUTDIR/predictions.jsonl"
"$PYTHON" -I "$EVAL" --merge-profiles "$PARTS" --output "$OUTDIR/profile.json" || true

if [ "$SCORE" = "1" ]; then
  "$PYTHON" -I "$SCORER" --input "$INPUT" --gold "$GOLD" \
    --pred "$OUTDIR/predictions.jsonl" --out "$OUTDIR/summary.json" \
    | tee "$OUTDIR/score.txt"
fi

exit "$fail"
