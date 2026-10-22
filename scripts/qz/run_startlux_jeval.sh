#!/usr/bin/env bash
# Run the StartLux-Decision JEVal evaluation on one QZ node.
#
#   bash scripts/qz/run_startlux_jeval.sh <4b|27b> [gpus] [model]
#
#   bash scripts/qz/run_startlux_jeval.sh 4b 0,1,2,3
#   bash scripts/qz/run_startlux_jeval.sh 27b 0,1,2,3,4,5,6,7
#
# Everything is resumable: rerunning continues from logs/qz/startlux_<tag>/predictions.jsonl.
# Compute nodes have no network: weights and the tokenizer must already be on shared storage, and
# HF_HUB_OFFLINE=1 is set here.  The official serve path (startlux-decide serve --launch-vllm) is
# used, with one vLLM server per GPU on vLLM 0.19.1.
#
# Environment overrides:
#   MODEL, GPUS, CONCURRENCY (128), MAX_MODEL_LEN (131072), GPU_MEM_UTIL (0.95),
#   MAX_THINK_BUDGET (4096), SHARD/SHARDS (0/1), THINK_MODE/GATE/MIX (request controls),
#   EXTRA_FLAGS (extra startlux_eval.py flags), RECONVERT=1, SCORE=0 to skip scoring.
set -euo pipefail

TAG=${1:?usage: run_startlux_jeval.sh <4b|27b> [gpus] [model]}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv-startlux/bin/python"
if [ "$TAG" = "27b" ]; then
  DEFAULT_MODEL="$ROOT/models/StartLux-27B-AutoThink"
  DEFAULT_GPUS="0,1,2,3,4,5,6,7"
  NAME="StartLux-27B-AutoThink"
elif [ "$TAG" = "4b" ]; then
  DEFAULT_MODEL="$ROOT/models/StartLux-4B-AutoThink"
  DEFAULT_GPUS="0,1,2,3"
  NAME="StartLux-4B-AutoThink"
else
  echo "unknown tag '$TAG' (use 4b or 27b)" >&2
  exit 2
fi
MODEL=${MODEL:-${3:-$DEFAULT_MODEL}}
GPUS=${GPUS:-${2:-$DEFAULT_GPUS}}
CONCURRENCY=${CONCURRENCY:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
MAX_THINK_BUDGET=${MAX_THINK_BUDGET:-4096}
SHARD=${SHARD:-0}
SHARDS=${SHARDS:-1}
SCORE=${SCORE:-1}
RUN_DIR="$ROOT/logs/qz/startlux_${TAG}"
REQUESTS="$ROOT/data/jeval/startlux/startlux_requests.jsonl"
SIDECAR="$ROOT/data/jeval/startlux/startlux_sidecar.json"

mkdir -p "$RUN_DIR" "$RUN_DIR/vllm"
ulimit -n 65536 2>/dev/null || echo "warning: could not raise the open-file limit (now $(ulimit -n))"

echo "HOST $(hostname) START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "model=$MODEL gpus=$GPUS concurrency=$CONCURRENCY max_model_len=$MAX_MODEL_LEN " \
     "gpu_mem=$GPU_MEM_UTIL shard=$SHARD/$SHARDS"

# ---------------------------------------------------------------- preflight (fail fast, no GPU)
if [ ! -f "$MODEL/config.json" ] || [ ! -f "$MODEL/decision_config.json" ] \
   || [ ! -f "$MODEL/tokenizer.json" ] || [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "ABORT: model folder $MODEL is incomplete (config/tokenizer/index missing)." >&2
  echo "       download it on a network-enabled node first; see scripts/qz/download_startlux_27b.sh" >&2
  exit 2
fi
"$PY" -I - "$MODEL" <<'PY'
import json, os, sys
model = sys.argv[1]
index = json.load(open(os.path.join(model, "model.safetensors.index.json")))
files = sorted(set(index["weight_map"].values()))
missing = [f for f in files if not os.path.isfile(os.path.join(model, f))
           or os.path.getsize(os.path.join(model, f)) == 0]
total = sum(os.path.getsize(os.path.join(model, f)) for f in files if f not in missing)
print(f"weights: {len(files)} shards, {total / 1e9:.1f} GB, missing={len(missing)}")
if missing:
    print("missing shards:", missing[:5], file=sys.stderr)
    raise SystemExit(2)
PY

"$PY" -I - "$MODEL" <<'PY'
import importlib.metadata as m, json, sys, torch
from startlux_decide.modelcfg import load_model_folder, load_tokenizer
from startlux_decide.prompt import check_tokenizer
model = sys.argv[1]
folder = load_model_folder(model)
tokenizer = load_tokenizer(model)
letters = check_tokenizer(tokenizer)
print(json.dumps({"vllm": m.version("vllm"), "torch": m.version("torch"),
                  "transformers": m.version("transformers"),
                  "startlux-decide": m.version("startlux-decide"),
                  "cuda_devices": torch.cuda.device_count(),
                  "folder_kind": folder.kind, "letter_ids": letters[:6] + ["..."]}, indent=2))
PY

if command -v nvidia-smi > /dev/null; then nvidia-smi -L; fi
GPUS_VISIBLE=$(printf '%s' "$GPUS" | awk -F, '{print NF}')
CUDA_COUNT=$("$PY" -I -c 'import torch; print(torch.cuda.device_count())')
if [ "$CUDA_COUNT" -lt "$GPUS_VISIBLE" ]; then
  echo "ABORT: $GPUS_VISIBLE GPUs requested but torch sees $CUDA_COUNT" >&2
  exit 3
fi

# ---------------------------------------------------------------- conversion (CPU, idempotent)
if [ "$RECONVERT" = "1" ] || [ ! -f "$REQUESTS" ] || [ ! -f "$SIDECAR" ]; then
  echo "converting JEVal to /v1/systemone requests"
  "$PY" -I "$ROOT/harness/jev_to_startlux.py" --out-dir "$ROOT/data/jeval/startlux" \
    > "$RUN_DIR/conversion.json"
fi
echo "requests: $(wc -l < "$REQUESTS") rows"

# ---------------------------------------------------------------- evaluation
EVAL_FLAGS=(
  --model "$MODEL" --name "$NAME"
  --gpus "$GPUS" --launch-mode official
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-think-budget "$MAX_THINK_BUDGET" --concurrency "$CONCURRENCY"
  --retries 5 --timeout 3600 --startup-timeout 1800
  --out "$RUN_DIR/predictions.jsonl" --answers "$RUN_DIR/answers.jsonl"
  --metrics "$RUN_DIR/metrics.json" --monitor-gpu --gpu-log "$RUN_DIR/gpu_util.csv"
  --server-log "$RUN_DIR/server.log" --vllm-log-dir "$RUN_DIR/vllm"
  --shard "$SHARD" --shards "$SHARDS"
)
[ -n "${THINK_MODE:-}" ] && EVAL_FLAGS+=(--think-mode "$THINK_MODE")
[ -n "${GATE:-}" ] && EVAL_FLAGS+=(--gate "$GATE")
[ -n "${MIX:-}" ] && EVAL_FLAGS+=(--mix "$MIX")
if [ "$SCORE" = "1" ]; then
  EVAL_FLAGS+=(--score --score-out "$RUN_DIR/score_summary.json")
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_FLAGS:-}" ] && EVAL_FLAGS+=(${EXTRA_FLAGS})

echo "--- evaluation ---"
set +e
"$PY" -I "$ROOT/harness/startlux_eval.py" "${EVAL_FLAGS[@]}"
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "--- evaluation failed (status $STATUS); server log tail ---" >&2
  tail -60 "$RUN_DIR/server.log" 2>/dev/null >&2 || true
  for log in "$RUN_DIR"/vllm/*.log; do
    [ -f "$log" ] || continue
    echo "--- $log ---" >&2
    tail -40 "$log" >&2 || true
  done
fi

echo "--- metrics ---"
[ -f "$RUN_DIR/metrics.json" ] && "$PY" -I -c "
import json,sys
m=json.load(open('$RUN_DIR/metrics.json'))
keys=('ok','failed','retries','elapsed_s','items_per_s','input_tokens_per_s','output_tokens_per_s','latency_p50_s','latency_p95_s','gpu')
print(json.dumps({k:m.get(k) for k in keys},indent=2))
"
echo "--- score ---"
[ -f "$RUN_DIR/score_summary.json" ] && "$PY" -I -c "
import json
s=json.load(open('$RUN_DIR/score_summary.json'))
print(json.dumps(s['overall'],indent=2))
print('missing',s['missing_predictions'],'invalid',s['invalid_predictions'])
"
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$STATUS"
exit "$STATUS"
