#!/usr/bin/env bash
# Launch the self-teacher generator on 8 GPUs, one vLLM process per GPU.
#
# Each worker loads Qwen3.5-4B once and processes its shard of every split
# (train, then dev) sequentially.  After all workers exit, per-shard part files
# are merged into data/multiselect/self_teacher_{train,dev}.jsonl and
# data/multiselect/self_teacher_summary.json is written.
#
# Everything is configurable through environment variables:
#   SHARDS=8 SAMPLES=2 MAX_TOKENS=2048 MAX_MODEL_LEN=8192 BATCH_SIZE=32
#   MAX_NUM_SEQS=128 GPU_MEM_UTIL=0.9 SPLITS="train dev"
#   TWO_STAGE=1 THINKING_MAX_TOKENS=1024 ANSWER_MAX_TOKENS=64
#   MODEL=... VENV_PYTHON=... PARTS_ROOT=... LOG_DIR=... DATA_DIR=... SUMMARY=...
#
# TWO_STAGE=1 runs stage 1 (thinking, stop at </think>) then stage 2 (forced short
# answer).  THINKING_BUDGET is the legacy vLLM budget path; it is broken on vLLM
# 0.19.1 without a reasoning config and is never passed when TWO_STAGE=1.
#
# The script is resumable: rerun it and each worker skips ids already present in
# its part file.  A failed shard is reported in the exit status and in
# logs/qz/ms_self_teacher_status.json; the merge still runs on partial parts.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$R"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

V=${VENV_PYTHON:-$R/.venv-startlux/bin/python}
HARNESS=$R/harness/generate_self_teacher.py
MODEL=${MODEL:-$R/models/Qwen3.5-4B}
SHARDS=${SHARDS:-8}
SAMPLES=${SAMPLES:-2}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
SPLITS=${SPLITS:-"train dev"}
THINKING_BUDGET=${THINKING_BUDGET:-0}
TWO_STAGE=${TWO_STAGE:-0}
THINKING_MAX_TOKENS=${THINKING_MAX_TOKENS:-1024}
ANSWER_MAX_TOKENS=${ANSWER_MAX_TOKENS:-64}
DATA_DIR=${DATA_DIR:-$R/data/multiselect}
PARTS_ROOT=${PARTS_ROOT:-$R/runs/qz/ms_self_teacher/parts}
LOG_DIR=${LOG_DIR:-$R/logs/qz}
SUMMARY=${SUMMARY:-$DATA_DIR/self_teacher_summary.json}
STATUS=$LOG_DIR/ms_self_teacher_status.json

STARTED_AT=$(date -Is)
mkdir -p "$PARTS_ROOT" "$LOG_DIR"

echo "=== ms self-teacher ==="
echo "started_at=$STARTED_AT"
echo "root=$R"
echo "python=$V"
echo "model=$MODEL"
echo "shards=$SHARDS samples=$SAMPLES max_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN"
echo "batch_size=$BATCH_SIZE max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEM_UTIL"
echo "splits=$SPLITS"
if [ "$TWO_STAGE" = "1" ]; then
    echo "mode=two_stage thinking_max_tokens=$THINKING_MAX_TOKENS answer_max_tokens=$ANSWER_MAX_TOKENS"
else
    echo "mode=one_stage thinking_budget=$THINKING_BUDGET"
fi

"$V" -I "$HARNESS" --check-model --model "$MODEL" || {
    echo "model check failed" >&2
    exit 3
}

declare -a pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

for index in $(seq 0 $((SHARDS - 1))); do
    tag=$(printf '%02d' "$index")
    log=$LOG_DIR/ms_self_teacher_shard-$tag.log
    command=("$V" -I "$HARNESS"
        --model "$MODEL"
        --shard "$index" --shards "$SHARDS"
        --samples "$SAMPLES"
        --max-tokens "$MAX_TOKENS"
        --max-model-len "$MAX_MODEL_LEN"
        --batch-size "$BATCH_SIZE"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --resume)
    if [ "$TWO_STAGE" = "1" ]; then
        command+=(--two-stage
                  --thinking-max-tokens "$THINKING_MAX_TOKENS"
                  --answer-max-tokens "$ANSWER_MAX_TOKENS")
    elif [ "$THINKING_BUDGET" -gt 0 ]; then
        command+=(--thinking-token-budget "$THINKING_BUDGET")
    fi
    for split in $SPLITS; do
        input=$DATA_DIR/$split.jsonl
        if [ ! -f "$input" ]; then
            echo "missing input: $input" >&2
            exit 2
        fi
        mkdir -p "$PARTS_ROOT/$split"
        command+=(--input "$input"
                  --output "$PARTS_ROOT/$split/part-$(printf '%05d' "$index").jsonl")
    done
    echo "launch shard $index -> $log"
    CUDA_VISIBLE_DEVICES=$index "${command[@]}" > "$log" 2>&1 &
    pids+=("$!")
done

overall=0
declare -a shard_codes=()
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        code=0
    else
        code=$?
        overall=1
    fi
    shard_codes+=("$code")
    echo "shard $index exit=$code"
done

echo "=== merge ==="
"$V" -I "$HARNESS" --merge-all \
    --parts-root "$PARTS_ROOT" \
    --data-dir "$DATA_DIR" \
    --summary "$SUMMARY" \
    --invalid-dir "$LOG_DIR" \
    --model "$MODEL" \
    --samples "$SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN"
merge_code=$?
if [ "$merge_code" -ne 0 ]; then
    overall=1
fi
echo "merge exit=$merge_code"

FINISHED_AT=$(date -Is)
"$V" -I - "$STATUS" "$SHARDS" "$SAMPLES" "$MAX_TOKENS" "$MAX_MODEL_LEN" \
    "$overall" "$merge_code" "$STARTED_AT" "$FINISHED_AT" \
    "${shard_codes[*]}" "$SPLITS" "$SUMMARY" "$TWO_STAGE" <<'PY'
import json
import os
import sys

(path, shards, samples, max_tokens, max_model_len, overall, merge_code,
 started, finished, codes, splits, summary, two_stage) = sys.argv[1:14]
payload = {
    "shards": int(shards),
    "samples": int(samples),
    "two_stage": two_stage == "1",
    "max_tokens": int(max_tokens),
    "max_model_len": int(max_model_len),
    "splits": splits.split(),
    "shard_exit_codes": [int(code) for code in codes.split()],
    "merge_exit_code": int(merge_code),
    "exit_code": int(overall),
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
