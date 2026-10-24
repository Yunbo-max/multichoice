#!/usr/bin/env bash
# CPU-only acceptance checks for the StartLux JEVal reproduction.
#
# Runs: environment imports, tokenizer/letter check, JEVal -> /v1/systemone conversion with schema
# validation, a mock-server end-to-end client test (retries, prediction format, scoring, resume),
# and the official vLLM launch dry-run.  No GPU is required.
#
#   bash harness/run_startlux_cpu_checks.sh
#
# Exit 0 only when every check passes.  Logs land in logs/cpu_checks/.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv-startlux/bin/python"
OUT="$ROOT/logs/cpu_checks"
DATA="$OUT/startlux_data"
MODEL="$ROOT/models/StartLux-4B-AutoThink"
PORT=$((18000 + RANDOM % 1000))
MOCK_PID=""
mkdir -p "$OUT" "$DATA"

cleanup() {
  if [ -n "$MOCK_PID" ] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

step() { printf '\n===== %s =====\n' "$*"; }

step "1. environment imports and versions"
"$PY" -I - <<'PY'
import importlib.metadata as m, json
import vllm, torch, transformers, startlux_decide
from startlux_decide.modelcfg import load_model_folder, load_tokenizer
from startlux_decide.prompt import check_tokenizer
root = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare"
folder = load_model_folder(root + "/models/StartLux-4B-AutoThink")
tok = load_tokenizer(root + "/models/StartLux-4B-AutoThink")
ids = check_tokenizer(tok)
print(json.dumps({"vllm": m.version("vllm"), "torch": m.version("torch"),
                  "transformers": m.version("transformers"),
                  "startlux-decide": m.version("startlux-decide"),
                  "folder_kind": folder.kind, "bidirectional": folder.bidirectional,
                  "letter_ids": ids[:6] + ["..."], "tokenizer": type(tok).__name__}, indent=2))
PY

step "2. JEVal -> /v1/systemone conversion + official schema validation"
"$PY" -I "$ROOT/harness/jev_to_startlux.py" --out-dir "$DATA" > "$OUT/conversion.json"
"$PY" -I - <<PY
import json
s = json.load(open("$OUT/conversion.json"))
assert s["schema_validation"] == "ok", s
assert s["rows_out"] == s["rows_in"] == 11257, s
print(json.dumps({k: s[k] for k in ("rows_in", "rows_out", "types", "wide_lists",
                                     "passes_histogram", "none_check_off_items")}, indent=2))
PY

step "3. official launch dry-run (vLLM 0.19.1 flags, no GPU touched)"
"$ROOT/.venv-startlux/bin/startlux-decide" launch-vllm \
  --model "$MODEL" --gpus 0,1,2,3 --base-port 8001 \
  --python "$PY" --max-model-len 131072 --gpu-memory-utilization 0.95 \
  --no-images --dry-run > "$OUT/launch_vllm_dry_run.txt" 2>&1
head -3 "$OUT/launch_vllm_dry_run.txt"

step "4. mock /v1/systemone end-to-end (retries, 255-option list, score, resume)"
python3 -I "$ROOT/harness/mock_startlux_server.py" --port "$PORT" \
  --fail-first 2 --fail-every 7 --delay 0.005 > "$OUT/mock_server.log" 2>&1 &
MOCK_PID=$!
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  sleep 0.2
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "mock server did not start"; exit 1; }

IDS="jeval:aida-testc:000000,jeval:ai2-arc-challenge:000000,jeval:agentharm:000002,jeval:agentharm:000000,jeval:decider:000050"
rm -f "$OUT/mock_predictions.jsonl" "$OUT/mock_answers.jsonl" \
      "$OUT/mock_predictions.answers.jsonl" "$OUT/mock_metrics.json" "$OUT/resume_metrics.json"
"$PY" -I "$ROOT/harness/startlux_eval.py" \
  --requests "$DATA/startlux_requests.jsonl" --sidecar "$DATA/startlux_sidecar.json" \
  --url "http://127.0.0.1:$PORT" --ids "$IDS" --concurrency 4 \
  --out "$OUT/mock_predictions.jsonl" --answers "$OUT/mock_answers.jsonl" \
  --metrics "$OUT/mock_metrics.json" --score --score-out "$OUT/mock_score.json" \
  --sii-input "$ROOT/data/jeval/sii_input.jsonl" --gold "$ROOT/data/jeval/gold.json" \
  --allow-failures > "$OUT/mock_eval.log" 2>&1
cat "$OUT/mock_eval.log" | tail -14

"$PY" -I - <<PY
import json
preds = [json.loads(l) for l in open("$OUT/mock_predictions.jsonl")]
metrics = json.load(open("$OUT/mock_metrics.json"))
score = json.load(open("$OUT/mock_score.json"))
assert len(preds) == 5, len(preds)
assert metrics["ok"] == 5 and metrics["failed"] == 0, metrics
assert metrics["retries"] >= 2, metrics          # the injected 502/429 responses
assert score["invalid_predictions"] == 0, score
widths = {p["type"]: len(p["probabilities"]) for p in preds}
choice_widths = {len(p["probabilities"]) for p in preds if p["type"] == "choice"}
assert widths["noul"] == 2 and widths["score"] == 5, widths
assert choice_widths == {4, 255}, choice_widths
print(json.dumps({"predictions": len(preds), "retries": metrics["retries"],
                  "invalid": score["invalid_predictions"]}, indent=2))
PY

BEFORE=$(wc -l < "$OUT/mock_predictions.jsonl")
"$PY" -I "$ROOT/harness/startlux_eval.py" \
  --requests "$DATA/startlux_requests.jsonl" --sidecar "$DATA/startlux_sidecar.json" \
  --url "http://127.0.0.1:$PORT" --ids "$IDS" --concurrency 4 \
  --out "$OUT/mock_predictions.jsonl" --answers "$OUT/mock_answers.jsonl" \
  --metrics "$OUT/resume_metrics.json" --allow-failures > "$OUT/resume.log" 2>&1
AFTER=$(wc -l < "$OUT/mock_predictions.jsonl")
[ "$BEFORE" = "$AFTER" ] || { echo "resume appended duplicate lines: $BEFORE -> $AFTER"; exit 1; }
echo "resume check: $AFTER prediction lines unchanged"

step "5. mock sweep used by the tuning harness"
"$PY" -I "$ROOT/harness/startlux_tune.py" --url "http://127.0.0.1:$PORT" \
  --request-limit 8 --concurrency-levels 2,4 --rounds 1 \
  --out-dir "$OUT/tune" > "$OUT/tune.log" 2>&1
tail -4 "$OUT/tune.log"
"$PY" -I - <<PY
import json
s = json.load(open("$OUT/tune/tune_summary.json"))
assert len(s["results"]) == 2 and s["recommendation"], s
print("tune check: 2 rounds, recommendation", s["recommendation"]["concurrency"])
PY

step "6. 27B weights (informational)"
if [ -f "$ROOT/models/StartLux-27B-AutoThink/model.safetensors.index.json" ]; then
  COUNT=$(ls "$ROOT/models/StartLux-27B-AutoThink"/model-*.safetensors 2>/dev/null | wc -l)
  echo "27B weight shards present: $COUNT (expected 14)"
else
  echo "27B weights not present yet: download before submitting the 27B job"
fi

printf '\nALL CPU CHECKS PASSED\n'
