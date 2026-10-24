#!/usr/bin/env bash
# SII-style self-teacher: thinking-chain generation (vLLM) + SII answer-slot
# readout (transformers).  Two independent phases, each sharded one process per
# GPU and resumable:
#
#   phase A  .venv-startlux vLLM: N thinking chains per row, stop at </think>.
#            Writes runs/.../reasoning/parts/<split>/part-<shard>.jsonl
#   phase B  .venv-sii transformers: read prompt at the end of each chain,
#            last-position hidden state -> 255 label logits -> sigmoid(q) for
#            the first K options, averaged over chains.  No answer parsing.
#            Writes runs/.../q/parts/<split>/part-<shard>.jsonl
#
# Everything is configurable through environment variables:
#   REASONING=1 READOUT=1 SHARDS=8 SAMPLES=2 MAX_TOKENS=1024 MAX_MODEL_LEN=8192
#   REASONING_BATCH=32 READOUT_BATCH=4 READOUT_MAX_INPUT_TOKENS=8192 Q_TRANSFORM=sigmoid
#   MAX_NUM_SEQS=128 GPU_MEM_UTIL=0.9 SPLITS="train dev"
#   MODEL=... VLLM_PYTHON=... READOUT_PYTHON=... DATA_DIR=... RUN_ROOT=...
#   SUMMARY=... LOG_DIR=...
#
# Set REASONING=0 for a readout-only job and READOUT=0 for a generation-only
# job (the full-data payloads are split this way).  The free-form "Answer: AC"
# generator in generate_self_teacher.py is not used here.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$R"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

HARNESS=$R/harness/sii_style_teacher.py
VLLM_PYTHON=${VLLM_PYTHON:-$R/.venv-startlux/bin/python}
READOUT_PYTHON=${READOUT_PYTHON:-$R/.venv-sii/bin/python}
MODEL=${MODEL:-$R/models/Qwen3.5-4B}
REASONING=${REASONING:-1}
READOUT=${READOUT:-1}
SHARDS=${SHARDS:-8}
SAMPLES=${SAMPLES:-2}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
REASONING_BATCH=${REASONING_BATCH:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
READOUT_BATCH=${READOUT_BATCH:-4}
READOUT_MAX_INPUT_TOKENS=${READOUT_MAX_INPUT_TOKENS:-8192}
Q_TRANSFORM=${Q_TRANSFORM:-sigmoid}
SPLITS=${SPLITS:-"train dev"}
DATA_DIR=${DATA_DIR:-$R/data/multiselect}
RUN_ROOT=${RUN_ROOT:-$R/runs/qz/sii_style_teacher}
SUMMARY=${SUMMARY:-$DATA_DIR/sii_teacher_summary.json}
LOG_DIR=${LOG_DIR:-$R/logs/qz}
REASONING_PARTS=$RUN_ROOT/reasoning/parts
REASONING_DIR=$RUN_ROOT/reasoning
Q_PARTS=$RUN_ROOT/q/parts
STATUS=$LOG_DIR/sii_style_teacher_status.json

STARTED_AT=$(date -Is)
mkdir -p "$REASONING_PARTS" "$Q_PARTS" "$LOG_DIR"

echo "=== SII-style self-teacher ==="
echo "started_at=$STARTED_AT"
echo "root=$R"
echo "model=$MODEL"
echo "reasoning=$REASONING readout=$READOUT shards=$SHARDS samples=$SAMPLES"
echo "max_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN"
echo "reasoning_batch=$REASONING_BATCH readout_batch=$READOUT_BATCH q_transform=$Q_TRANSFORM"
echo "splits=$SPLITS"

declare -a pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

phase_a=0
phase_b=0
reasoning_merge=0
readout_merge=0

if [ "$REASONING" = "1" ]; then
    "$VLLM_PYTHON" -I "$HARNESS" --check-model --model "$MODEL" || {
        echo "model check failed" >&2
        exit 3
    }
    for split in $SPLITS; do
        input=$DATA_DIR/$split.jsonl
        if [ ! -f "$input" ]; then
            echo "missing input: $input" >&2
            exit 2
        fi
        mkdir -p "$REASONING_PARTS/$split"
    done
    pids=()
    for index in $(seq 0 $((SHARDS - 1))); do
        tag=$(printf '%02d' "$index")
        log=$LOG_DIR/sii_style_teacher_reasoning_shard-$tag.log
        command=("$VLLM_PYTHON" -I "$HARNESS" generate
            --model "$MODEL" --shard "$index" --shards "$SHARDS"
            --samples "$SAMPLES" --max-tokens "$MAX_TOKENS"
            --max-model-len "$MAX_MODEL_LEN"
            --batch-size "$REASONING_BATCH"
            --max-num-seqs "$MAX_NUM_SEQS"
            --gpu-memory-utilization "$GPU_MEM_UTIL" --resume)
        for split in $SPLITS; do
            command+=(--input "$DATA_DIR/$split.jsonl"
                      --output "$REASONING_PARTS/$split/part-$(printf '%05d' "$index").jsonl")
        done
        echo "launch reasoning shard $index -> $log"
        CUDA_VISIBLE_DEVICES=$index "${command[@]}" > "$log" 2>&1 &
        pids+=("$!")
    done
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then code=0; else code=$?; phase_a=1; fi
        echo "reasoning shard $index exit=$code"
    done
    "$VLLM_PYTHON" -I "$HARNESS" --merge-reasoning \
        --parts-root "$REASONING_PARTS" --output-dir "$REASONING_DIR" \
        --data-dir "$DATA_DIR" --samples "$SAMPLES" || reasoning_merge=1
fi

if [ "$READOUT" = "1" ]; then
    for split in $SPLITS; do
        if [ ! -f "$REASONING_DIR/$split.jsonl" ]; then
            echo "missing reasoning file: $REASONING_DIR/$split.jsonl (run REASONING=1 first)" >&2
            exit 4
        fi
    done
    pids=()
    for index in $(seq 0 $((SHARDS - 1))); do
        tag=$(printf '%02d' "$index")
        log=$LOG_DIR/sii_style_teacher_readout_shard-$tag.log
        command=("$READOUT_PYTHON" -I "$HARNESS" readout
            --model "$MODEL" --shard "$index" --shards "$SHARDS"
            --samples "$SAMPLES" --readout-batch-size "$READOUT_BATCH"
            --max-input-tokens "$READOUT_MAX_INPUT_TOKENS"
            --q-transform "$Q_TRANSFORM" --resume)
        for split in $SPLITS; do
            command+=(--input "$DATA_DIR/$split.jsonl"
                      --reasoning "$REASONING_DIR/$split.jsonl"
                      --output "$Q_PARTS/$split/part-$(printf '%05d' "$index").jsonl")
        done
        echo "launch readout shard $index -> $log"
        CUDA_VISIBLE_DEVICES=$index "${command[@]}" > "$log" 2>&1 &
        pids+=("$!")
    done
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then code=0; else code=$?; phase_b=1; fi
        echo "readout shard $index exit=$code"
    done
    "$READOUT_PYTHON" -I "$HARNESS" --merge-readout \
        --parts-root "$Q_PARTS" --data-dir "$DATA_DIR" \
        --summary "$SUMMARY" --samples "$SAMPLES" || readout_merge=1
fi

overall=$(( phase_a || phase_b || reasoning_merge || readout_merge ))
FINISHED_AT=$(date -Is)
"$READOUT_PYTHON" -I - "$STATUS" "$SHARDS" "$SAMPLES" "$MAX_TOKENS" \
    "$MAX_MODEL_LEN" "$REASONING" "$READOUT" "$phase_a" "$phase_b" \
    "$reasoning_merge" "$readout_merge" "$STARTED_AT" "$FINISHED_AT" \
    "$SPLITS" "$SUMMARY" <<'PY'
import json
import os
import sys

(path, shards, samples, max_tokens, max_model_len, reasoning, readout,
 phase_a, phase_b, reasoning_merge, readout_merge, started, finished,
 splits, summary) = sys.argv[1:16]
payload = {
    "shards": int(shards),
    "samples": int(samples),
    "max_tokens": int(max_tokens),
    "max_model_len": int(max_model_len),
    "mode": {"reasoning": reasoning == "1", "readout": readout == "1"},
    "phase_a_exit": int(phase_a),
    "phase_b_exit": int(phase_b),
    "reasoning_merge_exit": int(reasoning_merge),
    "readout_merge_exit": int(readout_merge),
    "exit_code": int(phase_a or phase_b or reasoning_merge or readout_merge),
    "splits": splits.split(),
    "started_at": started,
    "finished_at": finished,
    "summary": summary,
}
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY
echo "finished_at=$FINISHED_AT"
echo "JOB EXIT=$overall"
exit "$overall"
