# Paper 1 EUR-LEX: end-to-end P0, P1, P2 code and acceptance gates
**Implementation date: 2026-11-02.** Branch: research/static-setjev-paper-20261027.

Scope: ONE new real benchmark (EUR-LEX with 100 fixed EuroVoc labels).
Retain the previous 5k Train / 1k DEV / 120-step warm + 200-step continuation
pilot for historical diagnosis only. **No new H100 runs or full 55k/5k/5k
results are claimed by this implementation.** Existing EUR-LEX DEV F1 scores
and corrected Independent Set-MAP results are in paper1_eurlex_pilot_results.md.
Do not re-open an official Test set as a development split.

## Scientific objectives and FAIR comparisons

P0: On EXACTLY the same 1k DEV rows and each saved checkpoint:
- fixed threshold 0.5 on raw BCE option sigmoid probabilities;
- threshold sweep 0.05..0.95 on BCE sigmoid OR structured, count-corrected
  **true marginals** P(i in Y|x), chosen on DEV (optimistic DEV diagnostic);
- exact set-MAP (highest-probability *individual set*, not count mode);
- Count Mode + top-m by candidate scores;
- non-deployable GOLD count + top-m ranking ceiling;
- official EUR-LEX micro/macro F1, Exact-Set EM, over/under selection;
- proper count NLL, full Gold-Set NLL and Count Brier under same public
  all-count support 0..K, not score-clipping or per-sample gold-based priors;
- paired question/cluster bootstrap for delta micro-F1 and
  paired count-NLL delta against BCE Continue.

P1a: Full 5,000-example Pilot BCE warm and SAME-budget continuation under
the current 4B LoRA trainer (rather than the earlier 120+200-step smoke).
With batch=8, an actual 5000-row pass requires **625 optimizer steps**
per arm. All five arm continuations start from the same warm BCE
checkpoint and have the exact same examples, LR search budget, seed,
backbone, candidate descriptions and prompts. Original BCE warm checkpoint
is reported separately.

P1b: **Causal frozen-score control**, the cleanest count-head experiment:
1. Take a single, fully trained BCE checkpoint selected on TRAIN/DEV.
2. On Qwen3.5-4B, export original answer-slot hidden h(x), original
   s_i^BCE(x) and exact log Z_m(s_BCE). Save these once per Train/DEV row.
3. FREEZE the backbone, original s_i^BCE(x), and all candidate ordering.
   Learn only one zero-initialized linear head a_m(x)=Wh+b using gold
   cardinality CE (no /K).
4. In an otherwise identical run learn an independent c_m(x)=Wh+b
   with the exact same hidden inputs, dimensions, optimizer steps, LR
   sweep, seed, per-batch examples and Count CE. Add a separate
   train-prior bias initialization for the independent head (stronger
   control, not a different parameterized family).
5. Decode both with the SAME normalized joint set probability. Train
   only on TRAIN, select count-head step and LR only on DEV. Report
   Count NLL, proper Gold-Set NLL, calibrated macro/micro F1, count
   MAE, and candidate-oracle rank ceiling. No gradient enters Qwen4B.

The learned count head c and anchor a share expressive power:
    c_m(x)=a_m(x)+log Z_m(s_BCE(x)).
A positive result is optimization/calibration, NOT representational novelty.

P2: A 55k full official training plan with 3 matched seeds may be
generated **only if** P0/P1 strong BCE + learned-count controls show
paired count- and full-set NLL gains and F1 non-inferiority. The plan
requires observed real QZ seconds/step from training.jsonl and
refuses to emit potentially killed 3-hour 6875-step LoRA jobs.
The current `harness/train_multiselect.py` does NOT support multi-job
LoRA resume with faithful optimizer and shuffle state. Do not fake
this by stringing together independently initialized jobs.
Run official Test ONCE only after complete validation selection.

## P0 — reuse old DEV predictions, no GPU

~~~bash
R=/inspire/hdd/global_user/zhangjiaquan-253108540222/baseline/jev
cd "$R"
D="$R/data/eurlex_decision_v1/pilot/dev.jsonl"
P="$R/runs/paper1_eurlex_pilot_dev_eval"
python -m paper_static.eurlex_p0_diagnostics \
  --gold "$D" \
  --arm "bce_continue=$P/bce_continue/seed_43/predictions.jsonl" \
  --arm "independent=$R/runs/paper1_eurlex_pilot_dev_eval_fixed/set_ce_independent/seed_43/predictions.jsonl" \
  --arm "goldcount=$R/runs/paper1_eurlex_pilot_dev_eval_fixed/bce_goldcount_w100/seed_43/predictions.jsonl" \
  --arm "anchor=$P/set_ce_anchor/seed_43/predictions.jsonl" \
  --arm "detach=$P/set_ce_anchor_detach/seed_43/predictions.jsonl" \
  --bootstrap 1000 \
  --output "$R/runs/paper1_eurlex_p0/pilot_s43.json"
~~~

The standard production scorer rejects outdated incorrectly selected
Independent Set-MAP. Re-decode using paper_static.redecode_set_map.py
FIRST where necessary (two commands in the prior Pilot report).
DEV-only threshold selection on 1k rows is exploratory; never report
the tuned DEV F1 as an independently held-out Test result.

Files:
- paper_static/eurlex_p0_diagnostics.py
- tests/test_paper1_eurlex_p0_p1.py

## P1a — full 5k Pilot, matched budget, with real runtime guard

Read actual training step timing from the existing completed 200-step Pilot:
train logs are at the corresponding QZ output roots (not GitHub).
First generate a PLAN (NO QZ submission):

~~~bash
python -m paper_static.eurlex_study_plan \
  --stage p1-full-5k \
  --template qz_c64/c64-warm-bce.json \
  --manifest "$R/data/eurlex_decision_v1/manifest.json" \
  --root "$R" --data "$R/data/eurlex_decision_v1" \
  --run-root "$R/runs/paper1_eurlex_pilot_full5k_v2" \
  --model /inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B \
  --observed-training-log "$R/runs/paper1_eurlex_pilot/set_ce_anchor/seed_43/training.jsonl" \
  --seed 43 44 45 --batch 8 \
  --output "$R/qz_eurlex_full5k_v2"
~~~

If status is STAGED_NOT_SUBMITTED, each output QZ JSON is inspected,
the warm BCE job goes FIRST, then exactly 625 steps per continued arm.
If status BLOCKED_ON_QZ_TIME_LIMIT, do not run the JSON; use a
measured faster GPU setup or implement trustworthy checkpoint/resume.
Note that 625 steps is one full 5k pass per arm, and warm and
continuation are separate passes from a shared checkpoint.

## P1b — frozen BCE scores and independent/anchored CountHead controls

The same 4B checkpoint is used for both cached feature exports. It must
be a finalized standalone BCE HF checkpoint (e.g. common warm BCE
from the completed Pilot, or the improved full5k warm if available).

~~~bash
BCE="$R/runs/paper1_eurlex_pilot/warm_bce/final"
CACHE="$R/runs/eurlex_frozen_p1/cache"

python -m qz.make_eurlex_frozen_cache_payloads \
  --template qz_c64/c64-warm-bce.json \
  --manifest "$R/data/eurlex_decision_v1/manifest.json" \
  --repo-root "$R" --data-root "$R/data/eurlex_decision_v1" \
  --bce-checkpoint "$BCE" --cache-root "$CACHE" \
  --output-dir "$R/qz_eurlex_frozen_extract_v1"
# REVIEW both JSON files; separately submit to QZ (does not use Test).
# Extraction is ONE frozen inference pass per train/dev row.

python -m paper_static.eurlex_frozen_head_jobs \
  --train-cache "$CACHE/train.pt" --dev-cache "$CACHE/dev.pt" \
  --train-gold "$R/data/eurlex_decision_v1/pilot/train.jsonl" \
  --dev-gold "$D" \
  --output-root "$R/runs/eurlex_frozen_p1" \
  --plan-dir "$R/qz_eurlex_frozen_heads_v1" \
  --seed 43 44 45 \
  --lr 0.001 0.0003 \
  --steps 600 --batch 64 \
  --python "$R/.venv-sii/bin/python"

# After cache files exist and match both Gold SHA256 values:
bash "$R/qz_eurlex_frozen_heads_v1/run_cpu_heads.sh"
~~~

This produces DEV predictions for:
- anchor/zero init;
- independent/zero init;
- independent/TRAIN-count-prior init;
for each seed and LR. Choose LR and stopping step on DEV Count NLL,
with equal selection budget; ensure each comparison uses identical cache
and same seed. Explicitly report the number of head parameters and
extra CPU/GPU time. The BCE candidate score vector is unchanged across
all frozen head arms. No prompt text or correctness oracle is put in
the head beyond the same frozen hidden state.

Run the P0 diagnostic on the saved frozen head predictions as well,
using names bce_continue, frozen_anchor, frozen_independent. For
the P2 gate, save 3 seed reports with SAME gold_sha256 and >=1000 DEV rows.

## P2 — full official 55k, three seeds, STRICT gate

~~~bash
python -m paper_static.eurlex_study_plan \
  --stage p2-55k \
  --template qz_c64/c64-warm-bce.json \
  --manifest "$R/data/eurlex_decision_v1/manifest.json" \
  --root "$R" --data "$R/data/eurlex_decision_v1" \
  --run-root "$R/runs/paper1_eurlex_full55k_v1" \
  --model /inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B \
  --observed-training-log "$R/runs/paper1_eurlex_pilot/set_ce_anchor/seed_43/training.jsonl" \
  --p0-report "$R/runs/paper1_eurlex_p0/p1_seed43.json" \
  --p0-report "$R/runs/paper1_eurlex_p0/p1_seed44.json" \
  --p0-report "$R/runs/paper1_eurlex_p0/p1_seed45.json" \
  --seed 43 44 45 --batch 8 \
  --output "$R/qz_eurlex_full55k_gate_v1"
~~~

Only if all three seed reports show improved proper Count NLL AND
proper Gold-Set NLL vs SAME-BACKBONE learned count and BCE, with
micro-F1 within 2pp, and observed upper runtime below QZ limit,
will actual JSON jobs be staged. An independent full official Test
is never loaded or submitted by these helpers. The plan_status.json
explains every rejection. Head-to-head data provenance and metric
normalization must be locked before looking at the official Test.

**Important:** P2 planning is implemented, but trustworthy
multi-job LoRA optimizer resume is NOT. If the 55k one-pass job
exceeds QZ's observed 3-hour limit, do not misrepresent P2 as ready
to launch until that engineering constraint is fixed.

## Research decision rule

1. If frozen Anchor improves count NLL but not proper set NLL, the
   evidence is count calibration, not joint decision improvement.
2. If frozen Anchor and frozen Independent Count are indistinguishable,
   there is no demonstrated Anchor-specific advantage; this is a
   cardinality modeling result, not a new expressive family.
3. If the frozen heads improve proper NLL but Set-MAP/threshold F1 do
   not, explicitly separate calibrated probabilities from the
   decision-optimal F1 decoder.
4. If the above fail on well-trained BCE after a full 5k pass,
   do not expand H100 expense to 55k or declare EUR-LEX a 'bad'
   benchmark merely because BCE wins. It is a valid negative test.
5. Existing synthetic C-regime K=16/32/64 generator still needs
   the public-strength and conditional-Gibbs sampling corrections
   before its scaling figures are made primary paper evidence.
