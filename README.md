# Multi-answer decision model (multichoice)

> **New experimental branch: Recurrent SetJev** — non-verbal multi-select latent recurrence, Qwen answer-slot feature caching, paired ablations and a sequential-action toy environment. See [BDH-CQ reading + reproducible pilot](docs/recurrent_set_decision_bdh_cq_reading.md). This is a BDH-CQ-inspired pilot, not a reproduction of its proprietary ARC solver.

Code, data splits and result files for a **multi-answer (multi-select) decision
model** built on Qwen3.5-4B. The readout follows the SII / InnerJev decision
recipe (answer-slot last-token readout over the 255 single-token option labels),
but replaces SII's single-label softmax with **independent per-option sigmoid +
BCE**, because a question can have several correct options.

The repository also contains a **multi-view self-teacher** pipeline and an
experimental `--objective set` model. Its teacher targets contain both candidate
membership and answer-count distributions. A small count head and exact dynamic
programming keep training and one-pass inference probabilistically consistent.
GPU pilots now exist, but **neither joint set CE nor raw soft-teacher
distillation has beaten the matched gold BCE baseline**. See
[three-seed and 5k results](docs/seed_results_2026-10-23.md).
The recursive teacher-utility hypothesis remains untested.

## SetJev v2 bug fixes and staged diagnostics (2026-10-24)

See [the latest Gate 1/3 implementation and runnable instructions](docs/setjev_v2_plan.md).

- `--count-anchor` now retains the **raw candidate logit scale**:
  `log Z_m(s) = log Z_m(s-max(s)) + m*max(s)`. The former centered-only
  implementation incorrectly made cardinality invariant to a common score shift.
- Structured dev checkpoint selection now uses the same `count_anchor` flag
  as the training loss and exported inference. Incompatible anchor modes are
  rejected for `--resume`; the flag is recorded in run metadata.
- `readout_audit.py` compares native candidate logits with no-reasoning Yes/No
  logits using a fixed checkpoint and a deterministic stratified dev cohort.
  It does **not yet** include reasoning-conditioned teacher extraction.
- `qz/submit_setjev_v2.sh` separates GPU submission into Gate 1 and Gate 3.
  Live Gate 3 requires an inspected real Gate 1 report and explicit sign-off.
  No Gate 1/3 GPU result is included by these fixes alone.

## New implementation: membership and cardinality distillation

See [the method, mathematics, prior-work audit and runnable recipe](docs/multiteacher_set_distillation.md).

- `sii_style_teacher.py`: candidate-conditioned Yes/No readout, direct/verify
  views, checked token IDs, provenance, explicit truncation policy and null
  targets on failure. Raw/centered label transforms are diagnostics only.
- `build_multiteacher_targets.py`: condition each chain on the task protocol,
  then mix fixed teacher weights; produce coherent membership and count targets.
- `set_distillation.py` and `train_multiselect.py --objective set`: hard or soft
  set CE, separately learned count head, exact marginals and set-MAP decoding;
  `--set-loss independent_marginal` / `induced_marginal` provide controls with
  the same teacher membership/count targets, architecture and decoder.
- `multiselect_to_unified.py`: passage/group-separated MultiRC splits; optional
  native single-choice plus multi-select CMB data using `--cmb-question-mode mixed`.
- `eval_multiselect.py`: frozen test evaluation, source/mode/count metrics,
  structured checkpoint support, stable joint-set NLL, membership and set-confidence
  Brier scores, checkpoint/input identity, and separately labeled oracle diagnostics.

## New exploration: stronger student, weaker next-generation teacher?

See [the hypothesis, mathematics, matched experiments and commands](docs/decision_teacher_generations.md)
and the [editable pilot configuration](experiments/decision_teacher_rounds.example.json).

- `plan_self_distillation_rounds.py` writes a bounded argv DAG for fixed versus
  refreshed teachers and common-initialization teaching probes. It does not
  launch GPU jobs or fabricate identities for future checkpoints.
- `audit_teacher_generations.py` compares native direct decisions and reasoning
  teacher distributions on the same covered dev questions, reporting coverage,
  completion, count quality and complete-set probability separately.
- `audit_teaching_utility.py` verifies completed next-student experiments with
  the same initial weights, seeds, data, updates and decoder, then measures
  teaching gains over gold-only training and between teacher generations.

Direct accuracy, reasoning-teacher quality, and downstream teaching utility are
different measurements. **The degradation hypothesis has not been demonstrated.**
The 3-seed gold-set comparison and a single-seed soft-teacher 5k pilot have
been run; both show negative or mixed signals as reported below. The recursive
teacher experiment has not been run.

The default trainer remains BCE. New data/output directories are required; the
archival datasets, pilot outputs and reported run below have not been rewritten.
Standard-library mathematical/contract tests run without model dependencies;
PyTorch tests explicitly skip if PyTorch is absent. Real-model forward,
FSDP export/reload and GPU training still need verification in the native runtime.

## Archived v1 task and data

Two multi-answer sources are unified into one JSONL schema
(`id/source/lang/split/state/question/options/target/n_options/gold_letters`):

| Split | Questions | CMB-Exam 多选 (zh) | MultiRC (en) | Source of split |
|---|---:|---:|---:|---|
| train | 28,753 | 23,981 | 4,772 | CMB train + MultiRC train |
| dev | 2,040 | 1,708 | 332 | holdout carved from the train pools + CMB val |
| test | 2,140 | 1,190 | 950 | CMB test + MultiRC validation |

- `target` is a **multi-hot** vector; it is not normalised to a simplex.
- Options per question: median 5, mean ~5.1, max 21.
- Gold positives per question: 1–10 (mode 3); 308 questions have 2 options.
- Dropped during conversion: 290 unparseable gold answers, 25 MultiRC rows with
  no positive candidate, 5 with too few candidates.
- The old summary reports zero question/prefix-key overlap. That audit did not
  ensure MultiRC passage isolation, so it is insufficient evidence of leakage-free
  evaluation. New experiments must use the corrected group split and full-content
  audit. `dev` is an internal holdout, not the official CMB/MultiRC dev split.

The file `data/multiselect/summary.json` contains the full statistics.

## Method

`harness/train_multiselect.py`:

1. Render the SII prompt: chat template with `enable_thinking=False`, then
   `State / Question / Options`, then the answer cue `The correct answer is (`.
2. At the last prompt position, take the hidden state and project it through the
   **frozen** LM head; keep the logits of the 255 option labels (A..Z, AA..,
   after filtering to single-token labels), cut to the first K options.
3. Score each candidate independently with sigmoid. Loss for one question:

   `L = (1/K) * sum_i [ softplus(s_i) - t_i * s_i ] + (lambda_g / K) * residual^T G residual`

   `--lambda-g 0 --graph none` is pure BCE. `G` can be the identity or a graph
   Laplacian over label-token embeddings; the reported run uses BCE only.
4. Full-rank FSDP over the 176 language linear modules (3.31B trainable
   parameters); embeddings and the output head stay frozen. Qwen3.5-4B base,
   lr 2e-6, global batch 64, 1 epoch = 450 steps, 8×H100.

`harness/multiselect_to_unified.py` builds the data from the raw CMB and MultiRC
files. With `--objective set`, the trainer instead uses
`[-sum_m rho_T[m] log rho_model[m] - dot(q_T,s) + sum_m rho_T[m] log Z_m(s)] / K`.
That is the default `--set-loss set_ce`. The stronger matched-information control
uses `--set-loss induced_marginal --count-weight 1`: BCE on the actual marginals
of the same set distribution plus teacher count CE. The complete derivation,
shared-optimum qualifications and baseline controls are in the linked method document.

## Archived result: gold-supervised multi-hot BCE

Run `runs/qz/ms_gold/v1`, selected checkpoint 405/450 by dev question EM
(dev = 2,040 questions, 10,332 candidates, threshold 0.5).

| Group | Questions | Question EM | F1 | Precision | Recall | BCE |
|---|---:|---:|---:|---:|---:|---:|
| **All** | 2,040 | **0.5863** | **0.9080** | 0.9003 | 0.9158 | 0.2787 |
| CMB-Exam 多选 (zh) | 1,708 | 0.5539 | 0.9056 | 0.8942 | 0.9172 | — |
| MultiRC (en) | 332 | 0.7530 | 0.9257 | 0.9468 | 0.9056 | — |

Question EM counts a question correct only if the predicted set equals the gold
set exactly. F1 is computed over all candidate decisions (a model that always
predicts the 3–5 most likely options can reach high F1 with low exact-set
accuracy, so question EM is the primary metric).

The final step (450) reached EM 0.5833 / F1 0.9082 / BCE 0.2777 but was not
selected. Per-checkpoint validation files, training log and threshold sweep are
included.

**No test-set metrics are reported yet.** No baseline (random / majority set /
SII single-label softmax) has been run on this unified dataset.
The 58.63% dev EM above is not a result of the new set objective or multi-teacher
distillation, and must not be directly compared with a rebuilt split.

## Set-loss controls (gold targets, 3 seeds)

A group-separated multi-select split was rebuilt
(`--cmb-question-mode multi`: 28,751 train / 2,042 dev / 2,140 test, no exact
train/dev/test overlap) and four objectives were run with gold targets, 4×H100,
450 steps, seeds 43/44/45. Full per-seed tables and paired deltas are in
`docs/seed_results_2026-10-23.md`.

| Arm | Question EM | F1 | Count mode acc | Count MAE |
|---|---:|---:|---:|---:|
| **BCE** | **0.5615 ± 0.0041** | **0.9058 ± 0.0021** | 0.6103 ± 0.0042 | 0.4680 ± 0.0048 |
| `set_ce` | 0.5573 ± 0.0021 | 0.9017 ± 0.0005 | 0.6152 ± 0.0045 | 0.4629 ± 0.0118 |
| `independent_marginal` | 0.5273 ± 0.0057 | 0.8959 ± 0.0009 | 0.5810 ± 0.0069 | 0.5224 ± 0.0127 |
| `induced_marginal` | 0.5606 ± 0.0063 | 0.9024 ± 0.0010 | **0.6324 ± 0.0122** | **0.4337 ± 0.0147** |

`set_ce` is lower than BCE on exact-set EM and F1 in all three seeds. The
induced-marginal control ties BCE on EM (paired mean -0.0010, mixed sign) but is
lower on F1; its seed-consistent gain is count calibration. `induced_marginal`
beats `set_ce` on count NLL/Brier/mode/MAE in every seed. The defensible claim
is therefore **count-calibrated set supervision**, not a superior joint set
objective. The seed-43 `set_ce` 8-GPU run (EM 0.5588, F1 0.9006) is consistent
with the 4-GPU pilot (0.5563 / 0.9012).

## Self-teacher

The teacher is the same base model with thinking enabled. The goal is to
distil the thinking chain's decision back into the first no-thinking forward
pass.

### Attempt 1: free-form answer parsing (`harness/generate_self_teacher.py`)

The model was asked to end with `Answer: AC`, and the answer text was parsed.
At 2 chains/question and 2,048 max tokens this is unusable on the full set:

| Group | Rows | Rows with ≥1 valid chain | Valid chains / all chains |
|---|---:|---:|---:|
| All | 30,793 | 2,313 | 3,516 / 61,586 = **5.71%** |
| CMB-Exam 多选 | 25,689 | 957 | 2.23% |
| MultiRC | 5,104 | 1,356 | 23.22% |

Reasoning almost always hits the token cap before emitting `</think>` and the
answer line. The 300-row pilot with a 4,096 cap still only reached 29.8% valid
chains.

### Archived attempt 2: label-logit teacher readout

No answer text is parsed. Phase A samples thinking chains with vLLM, stopping at
`</think>`. Phase B appends the reasoning to the answer cue and performs the SII
readout at the last token. Truncated chains can still be read out.

Pilot result (300 questions × 2 chains = 600 chains, 1×H100, max 1,024 tokens,
`centered_sigmoid`):

| Metric | Value |
|---|---|
| Rows with readout | 300 / 300 (100%) |
| Chains read | 600 / 600 |
| Reasoning truncated at 1,024 | 511 / 600 |
| q mean / p50 / max | 0.100 / 0.00138 / 0.500 |
| mean q on gold-positive options | 0.1777 |
| mean q on gold-negative options | 0.0191 |
| exact set agreement at q ≥ 0.5 | 0.0 |

Two issues invalidate these values as unmodified multi-label membership targets:

1. `centered_sigmoid` subtracts the log-sum-exp over all 255 labels, so `q` is
   capped at 0.5. A 0.5 threshold therefore predicts the empty set for every
   question. Changing a threshold does not establish correct membership semantics.
2. Raw sigmoid (`q_transform=sigmoid`) saturates at 1.0 for every option on the
   base head, so its chain variance is identically zero. This diagnostic does not
   invalidate the separately trained gold-BCE student, which can learn a sigmoid
   decision boundary; it invalidates treating untrained head logits as targets.

The current implementation replaces this default with a candidate-specific binary
readout and excludes truncated chains by default. Begin with two views × two
chains on a bounded pilot; assess teacher quality and coverage before full runs.
No new GPU cost or speed estimate has been measured. Correlated teacher views
are not independent evidence, and disagreement is not calibrated uncertainty.

## Repository layout

```
harness/                     data builder, trainer, self-teacher, eval helpers
innerjev/configs/4B/         module list used by the trainer (from InnerJev)
qz/                          qz job payloads and run order
scripts/qz/                  StartLux / SII eval payloads
data/multiselect/            unified train/dev/test JSONL + statistics
data/multiselect_pilot/      300-question pilot splits and teacher outputs
runs/qz/ms_gold/v1/          gold-supervised run: metrics, logs, selection
runs/qz/*_pilot*/            self-teacher pilot outputs
logs/qz/                     job logs and status files
logs/cpu_checks/             CPU validation and prompt-equivalence checks
对比报告.md                   SII-Decision vs StartLux-Decision comparison
复现清单.md                   reproduction checklist for the two upstream models
```

Not included: model weights, checkpoints, optimiser states, virtualenvs, the
JEVal/SII/StartLux downloads and large raw datasets. `runs/qz/ms_gold/v1/final`
is a weights directory and is intentionally absent.

## Reproduce

CPU checks (no weights / no GPU):

```bash
python -I harness/train_multiselect.py --self-test
python -I harness/train_multiselect.py --validate-only \
  --train data/multiselect/train.jsonl --dev data/multiselect/dev.jsonl \
  --model /path/to/Qwen3.5-4B \
  --modules innerjev/configs/4B/modules.json
python -I harness/sii_style_teacher.py --validate-only \
  --model /path/to/Qwen3.5-4B --input data/multiselect/dev.jsonl
```

Gold training (8×H100, full-rank FSDP):

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 8 harness/train_multiselect.py \
  --train data/multiselect/train.jsonl --dev data/multiselect/dev.jsonl \
  --model /path/to/Qwen3.5-4B \
  --modules innerjev/configs/4B/modules.json \
  --target gold --output runs/ms_gold \
  --lr 2e-6 --global-batch 64 --epochs 1 \
  --max-input-tokens 8192 --max-batch-tokens 8192 \
  --eval-every-frac 0.1 --lambda-g 0 --graph none --seed 43
```

Self-teacher (one view, finite pilot; use a frozen gold checkpoint):

```bash
MODEL=/path/to/frozen-gold-checkpoint CHECKPOINT_ID=gold-grouped-seed43 \
TEACHER_ID=direct TEACHER_VIEW=direct \
REASONING=1 READOUT=1 SHARDS=1 SAMPLES=2 LIMIT=200 MAX_TOKENS=4096 \
MAX_MODEL_LEN=16384 READOUT_MAX_INPUT_TOKENS=16384 Q_TRANSFORM=membership \
TRUNCATED_POLICY=exclude DATA_DIR=data/multiselect_grouped_mixed \
RUN_ROOT=runs/sii_style_teacher_pilot \
bash harness/run_sii_style_teacher.sh
```

## Notes and caveats

- Default teacher targets now use `membership` (candidate-conditioned Yes/No).
  `centered_sigmoid`, `sigmoid` and `softmax` teacher transforms are diagnostic
  outputs and are rejected by the coherent target builder.
- `dev` is carved from train pools. Corrected grouping checks exact normalized
  full passages/questions and paragraph identities; it does not detect all
  semantic near-duplicates or base-model pretraining contamination.
- The self-teacher here is **the same model**, not a stronger teacher. It matches
  the model's own thinking-conditioned decision, not a gold label.
- The teacher generations in `data/multiselect_pilot/` and the pilot runs are
  diagnostic outputs, not a finished training target.

## Sources and licences

- Base model: Qwen3.5-4B (Qwen licence).
- Trainer/readout recipe: adapted from
  [InnerJev](https://github.com/amazingljy1206/InnerJev) (Apache-2.0).
  `innerjev/configs/4B/*.json` comes from that repository.
- Data: CMB-Exam (Chinese medical benchmark) and MultiRC. Upstream licences
  apply; check them before redistributing the JSONL files.
- The SII-vs-StartLux comparison documents refer to
  [Startlux-Decision](https://github.com/StartLuxLabs/Startlux-Decision); no
  StartLux code or weights are included here.

