#!/usr/bin/env bash
# Stage-gated QZ submission; never submit Gate 1 and Gate 3 together.
#
#   bash qz/submit_setjev_v2.sh validate all
#   bash qz/submit_setjev_v2.sh --dry-run gate1
#   CONFIRM_SUBMIT=1 bash qz/submit_setjev_v2.sh --submit gate1
#   GATE1_REVIEWED=1 GATE1_AUDIT_JSON=/path/to/audit.json CONFIRM_SUBMIT=1 \
#     bash qz/submit_setjev_v2.sh --submit gate3
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="$DIR/../qz_setjev_v2"
MODE="${1:-validate}"
GATE="${2:-gate1}"
if [ "$#" -gt 2 ]; then
  echo "usage: $0 [validate|--dry-run|--submit] [gate1|gate3|all]" >&2
  exit 2
fi
case "$MODE" in
  validate|--dry-run|--submit) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
case "$GATE" in
  gate1|gate3|all) ;;
  *) echo "unknown gate: $GATE" >&2; exit 2 ;;
esac
if [ "$MODE" = "--submit" ] && [ "$GATE" = "all" ]; then
  echo "Refusing combined GPU submission; run Gate 1, review, then Gate 3." >&2
  exit 3
fi
if [ ! -d "$PAYLOAD_DIR" ]; then
  echo "no payloads; run python3 qz/make_setjev_v2_payloads.py" >&2
  exit 2
fi

shopt -s nullglob
case "$GATE" in
  gate1) FILES=("$PAYLOAD_DIR"/readout-audit-gate1-*.json) ;;
  gate3) FILES=("$PAYLOAD_DIR"/gate3-*.json) ;;
  all) FILES=("$PAYLOAD_DIR"/*.json) ;;
esac
shopt -u nullglob
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no $GATE payloads found in $PAYLOAD_DIR" >&2
  exit 2
fi
printf 'Stage: %s; mode: %s; jobs: %s\n' "$GATE" "$MODE" "${#FILES[@]}"

python3 - "${FILES[@]}" <<'PY'
import json, sys
required = ("workspace_id", "project_id", "logic_compute_group_id", "name",
            "description", "framework", "command", "max_running_time_ms",
            "task_priority", "framework_config")
ok = True
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        job = json.load(stream)
    errors = [key for key in required if key not in job]
    cfg = job.get("framework_config", [])
    if not isinstance(cfg, list) or len(cfg) != 1:
        errors.append("framework_config must contain one element")
    else:
        errors += [f"framework_config[0].{key}" for key in
                   ("image", "image_type", "instance_count", "spec_id", "shm_gi")
                   if key not in cfg[0]]
    if not str(job.get("command", "")).strip():
        errors.append("empty command")
    print(f"{'OK  ' if not errors else 'FAIL'} {path}" +
          (f" {errors}" if errors else ""))
    ok = ok and not errors
sys.exit(0 if ok else 1)
PY

if [ "$MODE" = "validate" ]; then
  echo "validation only; no GPU jobs submitted"
  exit 0
fi

if [ "$MODE" = "--submit" ]; then
  if [ "${CONFIRM_SUBMIT:-0}" != "1" ]; then
    echo "Refusing live submission without CONFIRM_SUBMIT=1" >&2
    exit 3
  fi
  if [ "$GATE" = "gate3" ]; then
    if [ "${GATE1_REVIEWED:-0}" != "1" ] || [ ! -r "${GATE1_AUDIT_JSON:-}" ]; then
      echo "Gate 3 requires GATE1_REVIEWED=1 and readable GATE1_AUDIT_JSON" >&2
      exit 3
    fi
    python3 - "$GATE1_AUDIT_JSON" <<'PY'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if (report.get("schema") != "readout_audit_v1"
        or report.get("mode") != "checkpoint-audit"
        or report.get("limit_mode") != "stratified"
        or report.get("n_questions", 0) < 100
        or not report.get("selected_ids_sha256")):
    raise SystemExit("Invalid or incomplete Gate 1 report (need real stratified model audit)")
comparison = report.get("comparison", {})
for metric in ("question_em", "bce", "membership_brier", "count_mae"):
    row = comparison.get(metric, {})
    if any(not isinstance(row.get(key), (float, int))
           or not math.isfinite(row[key])
           for key in ("native", "yesno", "delta_yesno_minus_native")):
        raise SystemExit(f"Gate 1 lacks usable {metric} comparison")
print("Gate 1 report validated. GATE1_REVIEWED=1 records explicit review.")
PY
  fi
fi

if ! command -v qz >/dev/null 2>&1; then
  echo "qz CLI unavailable; nothing submitted" >&2
  exit 127
fi
for file in "${FILES[@]}"; do
  echo "=== dry-run ${file##*/} ==="
  qz train CreateJob --dry-run --data "$(cat "$file")"
done
if [ "$MODE" = "--submit" ]; then
  for file in "${FILES[@]}"; do
    echo "=== submit $GATE ${file##*/} ==="
    qz train CreateJob --data "$(cat "$file")"
  done
fi
