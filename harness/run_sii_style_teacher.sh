#!/usr/bin/env bash
# Finite local/sharded teacher run. DRY_RUN=1 prints native CLI commands only.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEACHER_SCRIPT="$ROOT_DIR/harness/sii_style_teacher.py"
MODEL=${MODEL:-$ROOT_DIR/models/Qwen3.5-4B}
TEACHER_VIEW=${TEACHER_VIEW:-direct}
TEACHER_ID=${TEACHER_ID:-$(basename "$MODEL"):$TEACHER_VIEW}
CHECKPOINT_ID=${CHECKPOINT_ID:-}
DATA_DIR=${DATA_DIR:-$ROOT_DIR/data/multiselect_pilot}
RUN_ROOT=${RUN_ROOT:-$ROOT_DIR/runs/qz/sii_style_teacher}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
REASONING=${REASONING:-1}
READOUT=${READOUT:-1}
SHARDS=${SHARDS:-1}
SAMPLES=${SAMPLES:-2}
Q_TRANSFORM=${Q_TRANSFORM:-membership}
TRUNCATED_POLICY=${TRUNCATED_POLICY:-exclude}
YES_TOKEN=${YES_TOKEN:-Yes}
NO_TOKEN=${NO_TOKEN:-No}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
REASONING_BATCH=${REASONING_BATCH:-32}
READOUT_BATCH=${READOUT_BATCH:-4}
READOUT_MAX_INPUT_TOKENS=${READOUT_MAX_INPUT_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
SEED=${SEED:-0}
LIMIT=${LIMIT:-0}
RESUME=${RESUME:-1}
DRY_RUN=${DRY_RUN:-0}
STORE_LOGITS=${STORE_LOGITS:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
DEVICE=${DEVICE:-cuda}
MAX_PHASE_SECONDS=${MAX_PHASE_SECONDS:-21600}
PYTHON_GENERATE=${PYTHON_GENERATE:-python}
if [[ -x "$ROOT_DIR/.venv-sii/bin/python" ]]; then
  PYTHON_READOUT=${PYTHON_READOUT:-$ROOT_DIR/.venv-sii/bin/python}
else
  PYTHON_READOUT=${PYTHON_READOUT:-python}
fi

die() { printf 'teacher runner: %s\n' "$*" >&2; exit 2; }
for variable in REASONING READOUT RESUME DRY_RUN STORE_LOGITS ENFORCE_EAGER; do
  [[ ${!variable} == 0 || ${!variable} == 1 ]] || die "$variable must be 0 or 1"
done
(( REASONING + READOUT > 0 )) || die "enable REASONING and/or READOUT"
for variable in SHARDS SAMPLES MAX_TOKENS MAX_MODEL_LEN MAX_NUM_SEQS REASONING_BATCH READOUT_BATCH READOUT_MAX_INPUT_TOKENS MAX_PHASE_SECONDS; do
  [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || die "$variable must be a positive integer"
done
[[ $LIMIT =~ ^[0-9]+$ ]] || die "LIMIT must be a nonnegative integer"
[[ $TEACHER_VIEW == direct || $TEACHER_VIEW == verify ]] || die "TEACHER_VIEW must be direct or verify"
[[ $TRUNCATED_POLICY == include || $TRUNCATED_POLICY == exclude ]] || die "TRUNCATED_POLICY must be include or exclude"
[[ $DEVICE == cuda || $DEVICE == cpu ]] || die "DEVICE must be cuda or cpu"
case "$Q_TRANSFORM" in membership|sigmoid|centered_sigmoid|softmax) ;; *) die "unknown Q_TRANSFORM" ;; esac
if [[ $Q_TRANSFORM != membership ]]; then
  printf 'Diagnostic transform %s: these q values are not membership soft targets.\n' "$Q_TRANSFORM" >&2
fi

read -r -a SPLIT_NAMES <<< "${SPLITS:-train dev}"
(( ${#SPLIT_NAMES[@]} > 0 )) || die "SPLITS must not be empty"
for split in "${SPLIT_NAMES[@]}"; do
  [[ $split =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid split name: $split"
  [[ -f "$DATA_DIR/$split.jsonl" ]] || die "missing input: $DATA_DIR/$split.jsonl"
done
GPU_LIST=()
if [[ -n ${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}} ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
  (( ${#GPU_LIST[@]} >= SHARDS )) || die "fewer visible GPU IDs than SHARDS"
else
  for ((shard=0; shard<SHARDS; shard++)); do GPU_LIST+=("$shard"); done
fi

VIEW_ROOT="$RUN_ROOT/$TEACHER_VIEW"
REASONING_ROOT="$VIEW_ROOT/reasoning"
READOUT_ROOT="$VIEW_ROOT/q/$Q_TRANSFORM/$TRUNCATED_POLICY"
MERGED_DIR="$READOUT_ROOT/merged"
SUMMARY=${SUMMARY:-$MERGED_DIR/sii_teacher_summary.json}
COMMON=(--model "$MODEL" --teacher-id "$TEACHER_ID" --teacher-view "$TEACHER_VIEW"
        --samples "$SAMPLES" --q-transform "$Q_TRANSFORM" --truncated-policy "$TRUNCATED_POLICY"
        --yes-token "$YES_TOKEN" --no-token "$NO_TOKEN" --limit "$LIMIT"
        --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN"
        --max-input-tokens "$READOUT_MAX_INPUT_TOKENS" --seed "$SEED"
        --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
[[ -z $CHECKPOINT_ID ]] || COMMON+=(--checkpoint-id "$CHECKPOINT_ID")
OPTIONAL=()
(( RESUME == 0 )) || OPTIONAL+=(--resume)
(( STORE_LOGITS == 0 )) || OPTIONAL+=(--store-logits)
(( ENFORCE_EAGER == 0 )) || OPTIONAL+=(--enforce-eager)

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

print_command() { printf 'DRY_RUN'; printf ' %q' "$@"; printf '\n'; }
PIDS=()
stop_children() {
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  exit 130
}
trap stop_children INT TERM

run_stage() {
  local mode=$1 interpreter=$2 part_root=$3
  local shard split part log_path status=0
  local -a command
  PIDS=()
  for ((shard=0; shard<SHARDS; shard++)); do
    printf -v part 'part-%05d.jsonl' "$shard"
    command=("$interpreter" -I "$TEACHER_SCRIPT" "$mode" "${COMMON[@]}" "${OPTIONAL[@]}"
             --shard "$shard" --shards "$SHARDS")
    if [[ $mode == generate ]]; then
      command+=(--batch-size "$REASONING_BATCH" --max-num-seqs "$MAX_NUM_SEQS"
                --gpu-memory-utilization "$GPU_MEM_UTIL")
    else
      command+=(--readout-batch-size "$READOUT_BATCH" --device "$DEVICE")
    fi
    for split in "${SPLIT_NAMES[@]}"; do
      command+=(--input "$DATA_DIR/$split.jsonl" --output "$part_root/$split/$part")
      [[ $mode != readout ]] || command+=(--reasoning "$REASONING_ROOT/$split.jsonl")
    done
    if (( DRY_RUN )); then
      print_command env "CUDA_VISIBLE_DEVICES=${GPU_LIST[$shard]}" timeout "$MAX_PHASE_SECONDS" "${command[@]}"
    else
      mkdir -p "$LOG_DIR/$TEACHER_VIEW/$mode"
      for split in "${SPLIT_NAMES[@]}"; do mkdir -p "$part_root/$split"; done
      printf -v log_path '%s/%s/%s/shard-%05d.log' "$LOG_DIR" "$TEACHER_VIEW" "$mode" "$shard"
      env "CUDA_VISIBLE_DEVICES=${GPU_LIST[$shard]}" timeout "$MAX_PHASE_SECONDS" \
        "${command[@]}" > "$log_path" 2>&1 &
      PIDS+=("$!")
    fi
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
  PIDS=()
  return "$status"
}

run_merge() {
  local mode=$1 interpreter=$2 part_root=$3 output_dir=$4
  local -a command=("$interpreter" -I "$TEACHER_SCRIPT" "$mode" "${COMMON[@]}"
                    --parts-root "$part_root" --data-dir "$DATA_DIR" --output-dir "$output_dir")
  [[ $mode != merge-readout ]] || command+=(--summary "$SUMMARY")
  if (( DRY_RUN )); then print_command "${command[@]}"; else "${command[@]}"; fi
}

if (( REASONING )); then
  phase_status=0
  run_stage generate "$PYTHON_GENERATE" "$REASONING_ROOT/parts" || phase_status=1
  run_merge merge-reasoning "$PYTHON_GENERATE" "$REASONING_ROOT/parts" "$REASONING_ROOT" || phase_status=1
  (( phase_status == 0 )) || exit "$phase_status"
fi
if (( READOUT )); then
  phase_status=0
  run_stage readout "$PYTHON_READOUT" "$READOUT_ROOT/parts" || phase_status=1
  # Produce an explicit coverage report even if a worker produced null q rows.
  run_merge merge-readout "$PYTHON_READOUT" "$READOUT_ROOT/parts" "$MERGED_DIR" || phase_status=1
  (( phase_status == 0 )) || exit "$phase_status"
fi
if (( READOUT )); then
  printf 'Teacher view %s complete; %s outputs: %s\n' "$TEACHER_VIEW" "$Q_TRANSFORM" "$MERGED_DIR"
else
  printf 'Teacher view %s complete; reasoning outputs: %s\n' "$TEACHER_VIEW" "$REASONING_ROOT"
fi
