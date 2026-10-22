#!/usr/bin/env bash
# Pre-download the StartLux-Decision weights on a network-enabled node (login node).
# Compute nodes have no network; run this before submitting the QZ jobs.
#
#   bash scripts/qz/download_startlux_27b.sh            # 27B AutoThink (~54.7 GB)
#   bash scripts/qz/download_startlux_27b.sh 4b         # 4B AutoThink (~9.1 GB)
#
# The download is resumable: rerunning continues from the partial files.  hf-mirror is used
# because huggingface.co is not reachable from this cluster.
set -euo pipefail

TAG=${1:-27b}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HF="$ROOT/.venv-startlux/bin/hf"
[ -x "$HF" ] || HF=$(command -v hf)
if [ -z "$HF" ]; then echo "hf CLI not found (install huggingface-hub)" >&2; exit 2; fi

if [ "$TAG" = "27b" ]; then
  REPO="startlux-models/StartLux-Decision-v1.1-27B-AutoThink"
  DIR="$ROOT/models/StartLux-27B-AutoThink"
  EXPECTED_GB=54.7
elif [ "$TAG" = "4b" ]; then
  REPO="startlux-models/StartLux-Decision-v1.1-4B-AutoThink"
  DIR="$ROOT/models/StartLux-4B-AutoThink"
  EXPECTED_GB=9.1
else
  echo "usage: download_startlux_27b.sh [4b|27b]" >&2
  exit 2
fi

mkdir -p "$DIR"
echo "downloading $REPO -> $DIR (~${EXPECTED_GB} GB, resumable)"
for attempt in $(seq 1 20); do
  echo "--- attempt $attempt ---"
  if env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
      "$HF" download "$REPO" --local-dir "$DIR" --exclude 'figures/*'; then
    break
  fi
  echo "download interrupted; retrying in 15 s (partial files are kept)"
  sleep 15
done

# ---------------------------------------------------------------- verification
"$ROOT/.venv-startlux/bin/python" -I - "$DIR" "$EXPECTED_GB" <<'PY'
import json, os, sys
model, expected_gb = sys.argv[1], float(sys.argv[2])
required = ["config.json", "decision_config.json", "tokenizer.json",
            "tokenizer_config.json", "model.safetensors.index.json"]
missing = [f for f in required if not os.path.isfile(os.path.join(model, f))]
index_path = os.path.join(model, "model.safetensors.index.json")
shards, missing_shards, total = 0, [], 0
if os.path.isfile(index_path):
    index = json.load(open(index_path))
    files = sorted(set(index["weight_map"].values()))
    shards = len(files)
    for name in files:
        path = os.path.join(model, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing_shards.append(name)
        else:
            total += os.path.getsize(path)
declared = json.load(open(index_path)).get("metadata", {}).get("total_size", 0) if os.path.isfile(index_path) else 0
print(json.dumps({"missing_files": missing, "shards": shards,
                  "missing_shards": missing_shards,
                  "downloaded_gb": round(total / 1e9, 2),
                  "declared_gb": round(declared / 1e9, 2),
                  "expected_gb": expected_gb}, indent=2))
if missing or missing_shards:
    raise SystemExit("download incomplete: rerun this script")
if declared and abs(total - declared) > 1e8:
    raise SystemExit("weight shards do not add up to the declared total: rerun this script")
print("download verified")
PY
