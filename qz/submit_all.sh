#!/usr/bin/env bash
# Validate and (optionally) dry-run / submit the SII-Decision QZ CreateJob files.
#
#   bash qz/submit_all.sh                 # validate JSON + print per-file status
#   bash qz/submit_all.sh --dry-run       # also call: qz train CreateJob --dry-run ...
#   CONFIRM_SUBMIT=1 bash qz/submit_all.sh --submit
#
# qz is NOT installed in the current login container. When it is absent this
# script still validates every payload and prints the exact commands to run from
# a machine that has the qz CLI. Nothing is submitted without CONFIRM_SUBMIT=1.
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FILES=(
  "$DIR/sii_4b_eval.json"
  "$DIR/sii_27b_eval.json"
  "$DIR/sii_4b_lora_train.json"
  "$DIR/sii_27b_full_train.json"
  "$DIR/sii_4b_benchmark.json"
  "$DIR/sii_27b_benchmark.json"
)

echo "=== validating CreateJob payloads ==="
python3 - "${FILES[@]}" <<'PY'
import json
import sys

required = ["workspace_id", "project_id", "logic_compute_group_id", "name",
            "description", "framework", "command", "max_running_time_ms",
            "task_priority", "framework_config"]
ok = True
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        job = json.load(stream)
    missing = [key for key in required if key not in job]
    config = job.get("framework_config", [])
    problems = list(missing)
    if len(config) != 1:
        problems.append("framework_config must have one element")
    else:
        spec = config[0]
        for key in ("image", "image_type", "instance_count", "spec_id", "shm_gi"):
            if key not in spec:
                problems.append(f"framework_config[0].{key}")
    if not str(job.get("command", "")).strip():
        problems.append("empty command")
    print(f"{'OK  ' if not problems else 'FAIL'} {path}"
          + ("" if not problems else f"  problems={problems}"))
    ok = ok and not problems
sys.exit(0 if ok else 1)
PY

mode="${1:-validate}"
if [ "$mode" = "validate" ] || [ -z "$mode" ]; then
  exit 0
fi
if [ "$mode" != "--dry-run" ] && [ "$mode" != "--submit" ]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

if ! command -v qz >/dev/null 2>&1; then
  echo
  echo "qz CLI is not installed in this container; no API call made."
  echo "Run these from a machine with qz (dry-run first, then live after review):"
  for file in "${FILES[@]}"; do
    echo "  qz train CreateJob --dry-run --data \"\$(cat $file)\""
  done
  exit 127
fi

for file in "${FILES[@]}"; do
  echo "=== ${file##*/} ==="
  qz train CreateJob --dry-run --data "$(cat "$file")"
done

if [ "$mode" = "--submit" ]; then
  if [ "${CONFIRM_SUBMIT:-0}" != "1" ]; then
    echo "Refusing live submit without CONFIRM_SUBMIT=1" >&2
    exit 3
  fi
  for file in "${FILES[@]}"; do
    echo "=== submitting ${file##*/} ==="
    qz train CreateJob --data "$(cat "$file")"
  done
fi
