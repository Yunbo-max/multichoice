#!/usr/bin/env bash
# Explicit, opt-in two-state memory tests; never submits cloud jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON_BIN:-python3}"
STAGE="${1:-smoke}"
case "$STAGE" in
  smoke)
    "$PY" -m unittest discover -s tests -p 'test_episodic_set_decision.py' -v
    "$PY" -m unittest discover -s tests -p 'test_episodic_features.py' -v
    "$PY" -m unittest discover -s tests -p 'test_episodic_builder.py' -v
    "$PY" -m unittest discover -s tests -p 'test_episodic_generator_cache.py' -v
    "$PY" -m unittest discover -s tests -p 'test_episodic_online.py' -v
    "$PY" -m unittest discover -s tests -p 'test_episodic_identifiability.py' -v
    ;;
  toy)
    SEED="${SEED:-43}"
    OUT="${OUT:-runs/episodic/toy_${SEED}}"
    "$PY" harness/train_episodic_set.py train \
      --output "$OUT" --seed "$SEED" --device "${DEVICE:-cpu}" \
      --epochs "${EPOCHS:-12}" --train-batches "${TRAIN_BATCHES:-20}" \
      --eval-batches "${EVAL_BATCHES:-8}" --batch-size "${BATCH_SIZE:-16}" \
      --inner-steps "${INNER_STEPS:-4}" --horizon "${HORIZON:-6}" \
      --rule-switch-prob "${RULE_SWITCH_PROB:-0.25}"
    ;;
  logged)
    : "${TRAIN_CACHE:?Set TRAIN_CACHE to chronological training episode cache}"
    : "${DEV_CACHE:?Set DEV_CACHE to separate held-out episode cache}"
    : "${OUT:?Set OUT to fresh experiment output directory}"
    "$PY" harness/train_episodic_features.py \
      --train-cache "$TRAIN_CACHE" --dev-cache "$DEV_CACHE" --output "$OUT" \
      --device "${DEVICE:-cuda}" --seed "${SEED:-43}" \
      --epochs "${EPOCHS:-5}" --inner-steps "${INNER_STEPS:-4}" \
      --latent-dim "${LATENT_DIM:-128}"
    ;;
  *)
    echo "usage: bash qz/run_episodic_set.sh {smoke|toy|logged}" >&2
    exit 2
    ;;
esac
