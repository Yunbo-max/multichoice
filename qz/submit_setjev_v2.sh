#!/usr/bin/env bash
# Submit SetJev v2 Gate 1 + Gate 3 payloads in a loop.
#
#   bash qz/submit_setjev_v2.sh validate          # validate JSON only
#   bash qz/submit_setjev_v2.sh --dry-run         # validate + dry-run each payload
#   CONFIRM_SUBMIT=1 bash qz/submit_setjev_v2.sh --submit   # live submit
#
# Gate 1 (readout audit) must be reviewed before spending GPU on Gate 3.
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="$DIR/../qz_setjev_v2"

if [ ! -d "$PAYLOAD_DIR" ]; then
  echo "no payloads; run: python3 qz/make_setjev_v2_payloads.py" >&2
  exit 2
fi

mapfile -t FILES < <(ls "$PAYLOAD_DIR"/*.json 2>/dev/null | sort)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "no payloads found in $PAYLOAD_DIR" >&2
  exit 2
fi

MODE="${1:-validate}"
case "$MODE" in
  validate|--dry-run|--submit) ;;
  *) echo "usage: $0 [validate|--dry-run|--submit]" >&2; exit 2 ;;
esac

# Validate every payload regardless of mode.
python3 - "${FILES[@]}" <<'PY'
import json, sys
required = ["workspace_id", "project_id", "logic_compute_group_id", "name",
            "description", "framework", "command", "max_running_time_ms",
            "task_priority", "framework_config"]
ok = True
for path in sys.argv[1:]:
    job = json.load(open(path, encoding="utf-8"))
    problems = [k for k in required if k not in job]
    cfg = job.get("framework_config", [])
    if len(cfg) != 1:
        problems.append("framework_config must have one element")
    else:
        for k in ("image", "image_type", "instance_count", "spec_id", "shm_gi"):
            if k not in cfg[0]:
                problems.append(f"framework_config[0].{k}")
    if not str(job.get("command", "")).strip():
        problems.append("empty command")
    print(f"{'OK  ' if not problems else 'FAIL'} {path}" + (f"  {problems}" if problems else ""))
    ok = ok and not problems
sys.exit(0 if ok else 1)
PY

if [ "$MODE" = "validate" ]; then
  echo "validation only; nothing submitted"
  exit 0
fi

if ! command -v qz >/dev/null 2>&1; then
  echo "qz CLI not found; no API call made." >&2
  exit 127
fi

for file in "${FILES[@]}"; do
  echo "=== dry-run ${file##*/} ==="
  qz train CreateJob --dry-run --data "$(cat "$file")"
done

if [ "$MODE" = "--submit" ]; then
  if [ "${CONFIRM_SUBMIT:-0}" != "1" ]; then
    echo "Refusing live submit without CONFIRM_SUBMIT=1" >&2
    exit 3
  fi
  for file in "${FILES[@]}"; do
    echo "=== submitting ${file##*/} ==="
    qz train CreateJob --data "$(cat "$file")"
  done
fi
