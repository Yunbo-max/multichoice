#!/usr/bin/env bash
# Submit one QZ CreateJob payload: dry-run first, then live (unless --dry-run-only).
#
# Run this from a terminal that has the qz CLI:
#   cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/decision_compare
#   bash qz/submit_one.sh qz/sii_4b_benchmark.json
#   bash qz/submit_one.sh qz/sii_4b_eval.json
#   bash qz/submit_one.sh scripts/qz/startlux_4b_jeval_createjob.json
#   ...
set -euo pipefail

PAYLOAD=${1:?usage: submit_one.sh <payload.json> [--dry-run-only]}
MODE=${2:-}

if ! command -v qz >/dev/null 2>&1; then
  echo "qz CLI not found in PATH; run this on the machine where 'qz login' succeeded." >&2
  exit 127
fi

[ -f "$PAYLOAD" ] || { echo "no such payload: $PAYLOAD" >&2; exit 2; }
python3 -m json.tool "$PAYLOAD" >/dev/null || { echo "invalid JSON: $PAYLOAD" >&2; exit 2; }

echo "=== dry-run: $PAYLOAD ==="
qz train CreateJob --dry-run --data "$(cat "$PAYLOAD")"

if [ "$MODE" = "--dry-run-only" ]; then
  exit 0
fi

echo "=== live submit: $PAYLOAD ==="
qz train CreateJob --data "$(cat "$PAYLOAD")"
