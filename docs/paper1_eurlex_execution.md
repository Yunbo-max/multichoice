# Paper 1 — EUR-LEX executable runbook (2026-11-01)

**Scope:** one NEW real benchmark only: LexGLUE EUR-LEX, 100 EuroVoc labels. Defer Pairwise learning, ECtHR and SATA. Old CMB/MultiRC results remain historical negatives. **No EUR-LEX GPU jobs or full benchmark data have been executed by these GitHub commits.**

## Code now implemented

- paper_static/eurlex_data.py: official 55k train / 5k validation / optional 5k test, pinned HF revision and SHA256, exactly 100 official EuroVoc ClassLabel names, verified English label descriptors from a fixed Git commit, deterministic prompt-length audit with original Qwen3.5 chat template and 255-label single-token readout, document-only head/tail truncation, train/dev document overlap rejection, pilot subsamples and immutable hashes.
- harness/set_distillation.py and harness/train_multiselect.py: new --gold-count-weight for the strong BCE+gold Count Head; default 1 preserves earlier runs. For K=100 set weight 100 to compare full Gold Count CE with per-candidate BCE. Only available with --objective set --count-target gold.
- qz/make_eurlex_payloads.py: SAFE QZ one-H100 training payload generation. No test, auto-submit or rm -rf; model and dataset checksums checked before launch.
- qz/make_eurlex_eval_payloads.py: DEV-only model inference plus proper-score plans; never Test.
- paper_static/eurlex_metrics.py: official micro/macro F1, Exact-Set EM, Count NLL/Brier, Count MAE, Gold-Set NLL under the same legal support, and over/under-selection.
- paper_static/eurlex_tune_threshold.py: DEV-only BCE threshold selection using micro/macro F1 and exact EM.
- tests/test_paper1_eurlex.py and tests/test_paper1_eurlex_metrics.py: offline schema, no-Test and F1/NLL regressions. GitHub Actions CPU workflow checks syntax and pure-Python tests. Torch gradient test requires installed PyTorch.

Dataset: https://huggingface.co/datasets/coastalcph/lex_glue (configuration eurlex, cc-by-4.0); descriptors pinned to nlpaueb/multi-eurlex Git SHA af2da31b54ca4272d880a46aee8dbbf346712e90. Every document receives ALL 100 EuroVoc concept candidates in their exact official order; gold labels are never used to select candidates.

## 1. Create official EUR-LEX train and validation ONLY

Commands on the QZ checkout; adapt R and M if the cluster location has changed:

~~~bash
R=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev
M=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B
cd "$R"

"$R/.venv-sii/bin/python" -m paper_static.eurlex_data \
  --output "$R/data/eurlex_decision_v1" \
  --tokenizer "$M" \
  --download-descriptors \
  --pilot-train 5000 --pilot-dev 1000 --max-input-tokens 8192
~~~

This first step needs HF dataset and pinned GitHub descriptor network access. On an offline node supply --descriptors /path/to/pinned/eurovoc_descriptors.json, downloaded on an accessible machine, instead of --download-descriptors. The file is fingerprinted in manifest.json. Existing output directories are never overwritten.

The script loads only train and validation by default. There is NO test.jsonl unless --include-test is deliberately used to materialize a distinct final package. It verifies the 100 official ID-to-description mappings and full prompt-token budget. Inspect the manifest before any GPU job.

## 2. Mandatory Qwen input and data preflight

~~~bash
python -m unittest discover -s tests -p test_paper1_eurlex.py -v
python -m unittest discover -s tests -p test_paper1_eurlex_metrics.py -v

"$R/.venv-sii/bin/python" -I harness/train_multiselect.py --validate-only \
  --train "$R/data/eurlex_decision_v1/pilot/train.jsonl" \
  --dev "$R/data/eurlex_decision_v1/pilot/dev.jsonl" \
  --model "$M" --modules "$R/innerjev/configs/4B/modules.json" \
  --target gold --objective bce --lora --lora-r 16 --lora-alpha 32 \
  --global-batch 8 --epochs 1 --world-size 1 \
  --max-input-tokens 8192 --max-batch-tokens 8192 \
  --expect-trainable 0 \
  --validate-output "$R/logs/eurlex_pilot_validation.json"
~~~

Abort on token overflow, missing labels, wrong tokenizer answer-slot IDs,
invalid train/DEV split, truncated options or nonfinite gradients.
Document-only truncation is permitted, but must be identical across arms.

## 3. Generate the first QZ Pilot (does NOT submit jobs)

~~~bash
"$R/.venv-sii/bin/python" -m qz.make_eurlex_payloads \
  --template qz_c64/c64-warm-bce.json \
  --manifest "$R/data/eurlex_decision_v1/manifest.json" \
  --repo-root "$R" --data-root "$R/data/eurlex_decision_v1" \
  --run-root "$R/runs/paper1_eurlex_pilot" --model-root "$M" \
  --output-dir qz_eurlex_pilot --seeds 43

# FIRST warm common BCE, and check the exported final checkpoint exists:
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-warm-bce-s44.json)"

# Only AFTER the common checkpoint passes verification, run matched arms:
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-bce_continue-s43.json)"
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-bce_goldcount_w100-s43.json)"
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-set_ce_independent-s43.json)"
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-set_ce_anchor-s43.json)"
qz train CreateJob --data "$(cat qz_eurlex_pilot/paper1-eurlex-pilot-set_ce_anchor_detach-s43.json)"
~~~

These are staged single-H100 Qwen3.5-4B LoRA r16/alpha32 jobs:
batch 8, 120 BCE warm updates, 80 identical extra updates per arm,
lr 2e-5 as an initial Pilot choice, 8192-token max input.
Count CE baseline has --count-target gold --gold-count-weight 100;
this avoids a K=100 weak-count-head strawman. Tune LR and this count
weight on DEV with equal search budget before a final superiority claim.
The warm BCE result remains in the paper separately from BCE Continue.

## 4. DEV inference and objective, comparable metrics

~~~bash
"$R/.venv-sii/bin/python" -m qz.make_eurlex_eval_payloads \
  --template qz_c64/c64-warm-bce.json \
  --repo-root "$R" --data-root "$R/data/eurlex_decision_v1" \
  --run-root "$R/runs/paper1_eurlex_pilot" \
  --eval-root "$R/runs/paper1_eurlex_pilot_dev_eval" \
  --output-dir qz_eurlex_pilot_eval --seeds 43

# Submit each generated DEV evaluation only after its model exists.
# It runs harness.eval_multiselect and paper_static.eurlex_metrics.

# Independently calibrate BCE threshold on DEV (not Test):
"$R/.venv-sii/bin/python" -m paper_static.eurlex_tune_threshold \
  --gold "$R/data/eurlex_decision_v1/pilot/dev.jsonl" \
  --predictions "$R/runs/paper1_eurlex_pilot_dev_eval/bce_continue/seed_43/predictions.jsonl" \
  --output "$R/runs/paper1_eurlex_pilot_dev_eval/bce_continue/seed_43/dev_thresholds.json"
~~~

Report official EUR-LEX micro/macro F1, Exact-Set EM, Count NLL,
Gold-Set NLL, Count MAE and time/VRAM; use the same publicly valid
count support for all methods. Don't use Test threshold tuning.
Use no gold count at inference. For count mechanism, compare A4 Anchor
against BCE, BCE Continue, strong BCE+gold Count Head, Independent Count
Set CE, and Anchor Detach. If Learned Count matches Anchor, attribute
any benefit to cardinality modeling, NOT a new probability family.

## 5. Full 55k train / 5k validation and final locked 5k Test

Reuse this audited package; generate a NEW, uniquely named training and
evaluation payload directory with --full --seeds 43 44 45 and matched budgets.
For all arms, fix the same candidate list, optimizer updates, model, prompt,
seed design and validation selection rules. Train only on 55k official
train; choose checkpoints, LR and thresholds on 5k official validation.

After freezing final configurations, use a NEW package with --include-test,
the same dataset revision and descriptor SHA. For actual held-out scoring,
paper_static.eurlex_metrics requires --split test --unlock-test and a
frozen-selection file with keys locked=true, dataset=coastalcph/lex_glue,
gold_sha256, checkpoint_id, threshold. This is a deliberate one-off Test
operation; existing QZ generators cannot silently launch Test.

## Publication boundary

EUR-LEX has K=100 but may still be well predicted by independent BCE.
Do NOT assume a real-dataset advantage; the earlier synthetic K=64
trend has known generator and scorer audit issues and is not a substitute
for verified real-data results. First demonstrate proper Count/Joint-NLL
gain over the learned-count baseline without sacrificing official F1.
