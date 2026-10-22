# Multi-answer decision model (multichoice)

Code, data splits and result files for a **multi-answer (multi-select) decision
model** built on Qwen3.5-4B. The readout follows the SII / InnerJev decision
recipe (answer-slot last-token readout over the 255 single-token option labels),
but replaces SII's single-label softmax with **independent per-option sigmoid +
BCE**, because a question can have several correct options.

The repository also contains the **self-teacher** pipeline that distils the same
model's thinking chains back into a single no-thinking forward pass, and its
pilot logs.

## Task and data

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
- Splits do not overlap (`train/dev`, `train/test`, `dev/test` = 0 exact hits).
  `dev` is an internal holdout, not the official CMB/MultiRC dev split.

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
files.

## Result: gold-supervised multi-hot BCE

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

### Attempt 2: SII answer-slot readout (`harness/sii_style_teacher.py`)

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

Two issues are visible and are the current work items:

1. `centered_sigmoid` subtracts the log-sum-exp over all 255 labels, so `q` is
   capped at 0.5. A 0.5 threshold therefore predicts the empty set for every
   question. The threshold must be calibrated on dev (or `q` recalibrated).
2. Raw sigmoid (`q_transform=sigmoid`) saturates at 1.0 for every option on the
   base head, so its chain variance is identically zero. Variance must be
   computed from `per_chain_centered` / `per_chain_softmax`, not raw sigmoid.

Planned: 8 chains per question (full train+dev ≈ 246k chains ≈ 238M completion
tokens, ~1.5–2 h on 8×H100), plus per-option variance and pairwise chain
Jaccard in the merged summary. Chains from the same prompt are correlated, so
the variance is an agreement signal, not a calibrated epistemic uncertainty.

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

Self-teacher (K chains, then readout):

```bash
REASONING=1 READOUT=1 SHARDS=8 SAMPLES=8 MAX_TOKENS=1024 \
MAX_MODEL_LEN=8192 Q_TRANSFORM=centered_sigmoid \
DATA_DIR=data/multiselect RUN_ROOT=runs/sii_style_teacher \
bash harness/run_sii_style_teacher.sh
```

## Notes and caveats

- The base-model label logits are all positive, so raw per-candidate sigmoid
  saturates; use `centered_sigmoid` or `softmax` and check `q_value_stats`.
- `dev` is carved from the train pools, not an official split. CMB and MultiRC
  upstream splits may have near-duplicates that this conversion does not detect.
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
