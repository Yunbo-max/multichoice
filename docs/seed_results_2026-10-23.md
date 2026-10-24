# Set-loss controls on gold targets (3 seeds)

Date: 2026-10-23. Status: completed pilot, negative result for the joint set
objective on gold targets. Not a teacher-distillation result and not a test-set
result.

## Setup

- Data: group-separated multi-select split built with
  `harness/multiselect_to_unified.py --cmb-question-mode multi`:
  - train 28,751 questions; dev 2,042 questions (10,135 candidates); test 2,140.
  - `selection_mode=multi` for all rows; `allow_empty=false`.
  - Train/dev/test exact-overlap audit is empty; MultiRC rows are grouped by
    paragraph before splitting.
- Targets: gold multi-hot targets (`--target gold`).
- Model: Qwen3.5-4B base, frozen embeddings/output head, full-rank FSDP over the
  176 language linear modules, 4×H100 per arm.
- Budget: 1 epoch = 450 steps, lr 2e-6, global batch 64, eval every 10%,
  seed 43/44/45. Selection: dev question EM, tie-break lower marginal BCE.
- Arms:
  - `bce`: ordinary per-option BCE, protocol threshold decoder.
  - `set_ce`: exact set cross-entropy, count head, set-MAP decoder.
  - `independent_marginal`: BCE on raw sigmoid scores plus count CE
    (`--set-loss independent_marginal --count-weight 1`).
  - `induced_marginal`: BCE on the actual marginals of the set distribution plus
    count CE (`--set-loss induced_marginal --count-weight 1`).
- The 4-GPU arms are an internally matched pilot. The seed-43 `set_ce` arm was
  also run at 8 GPUs; it is consistent with the 4-GPU result (see below).

## Mean over seeds 43/44/45

| Arm | Question EM | F1 | Count mode acc | Count MAE |
|---|---:|---:|---:|---:|
| **BCE** | **0.5615 ± 0.0041** | **0.9058 ± 0.0021** | 0.6103 ± 0.0042 | 0.4680 ± 0.0048 |
| `set_ce` | 0.5573 ± 0.0021 | 0.9017 ± 0.0005 | 0.6152 ± 0.0045 | 0.4629 ± 0.0118 |
| `independent_marginal` | 0.5273 ± 0.0057 | 0.8959 ± 0.0009 | 0.5810 ± 0.0069 | 0.5224 ± 0.0127 |
| `induced_marginal` | 0.5606 ± 0.0063 | 0.9024 ± 0.0010 | **0.6324 ± 0.0122** | **0.4337 ± 0.0147** |

Count mode/MAE for BCE are computed from its thresholded set sizes; the set arms
use the learned count head. Mean ± sample standard deviation over the three seeds.

## Per-seed values

| Arm | Seed | Step | EM | F1 | Count mode | Count MAE |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 43 | 360 | 0.5642 | 0.9049 | 0.6141 | 0.4647 |
| BCE | 44 | 405 | 0.5568 | 0.9041 | 0.6058 | 0.4736 |
| BCE | 45 | 360 | 0.5637 | 0.9082 | 0.6112 | 0.4657 |
| `set_ce` | 43 | 450 | 0.5563 | 0.9012 | 0.6126 | 0.4755 |
| `set_ce` | 44 | 450 | 0.5558 | 0.9016 | 0.6126 | 0.4613 |
| `set_ce` | 45 | 450 | 0.5597 | 0.9023 | 0.6205 | 0.4520 |
| `independent_marginal` | 43 | 450 | 0.5338 | 0.8968 | 0.5847 | 0.5230 |
| `independent_marginal` | 44 | 405 | 0.5240 | 0.8956 | 0.5730 | 0.5348 |
| `independent_marginal` | 45 | 360 | 0.5240 | 0.8951 | 0.5852 | 0.5093 |
| `induced_marginal` | 43 | 405 | 0.5588 | 0.9030 | 0.6244 | 0.4417 |
| `induced_marginal` | 44 | 360 | 0.5553 | 0.9012 | 0.6263 | 0.4427 |
| `induced_marginal` | 45 | 360 | 0.5676 | 0.9028 | 0.6464 | 0.4167 |

## Paired differences per seed

| Comparison | Metric | Mean delta | Per-seed |
|---|---|---:|---|
| `set_ce - BCE` | EM | **-0.0042** | -0.0078, -0.0010, -0.0039 |
| `set_ce - BCE` | F1 | **-0.0041** | -0.0037, -0.0025, -0.0059 |
| `induced - BCE` | EM | -0.0010 | -0.0054, -0.0015, +0.0039 |
| `induced - BCE` | F1 | **-0.0034** | -0.0019, -0.0030, -0.0054 |
| `induced - BCE` | Count MAE | **-0.0343** | -0.0230, -0.0309, -0.0490 |
| `induced - BCE` | Count mode | **+0.0220** | +0.0103, +0.0206, +0.0353 |
| `induced - set_ce` | EM | +0.0033 | +0.0024, -0.0005, +0.0078 |
| `induced - set_ce` | F1 | +0.0006 | +0.0018, -0.0004, +0.0006 |
| `induced - set_ce` | Count MAE | **-0.0292** | -0.0338, -0.0186, -0.0353 |
| `induced - set_ce` | Count mode | **+0.0171** | +0.0118, +0.0137, +0.0260 |
| `independent - BCE` | EM | **-0.0343** | -0.0304, -0.0328, -0.0397 |

## 8-GPU consistency check (seed 43, `set_ce`)

| Run | EM | F1 | Count mode | Count MAE |
|---|---:|---:|---:|---:|
| 4×H100 | 0.5563 | 0.9012 | 0.5779 | 0.4755 |
| 8×H100 | 0.5588 | 0.9006 | 0.5779 | 0.4750 |

The 4-GPU pilot is consistent with the 8-GPU run at the 0.25 EM-point level, so
the 4-GPU matrix is usable for relative comparisons. Official numbers still
belong to the 8-GPU recipe.

## Interpretation

1. **The joint set CE is not supported on gold targets.** `set_ce` is lower than
   BCE on exact-set EM in all three seeds and lower on F1 in all three seeds.
2. **The induced-marginal control ties BCE on EM but is lower on F1.** Its clear,
   seed-consistent advantage is count calibration (NLL/Brier/mode/MAE), not
   exact-set accuracy.
3. **`induced_marginal` beats `set_ce` on count calibration in every seed** and
   is slightly higher on EM in two of three seeds. The exact set likelihood does
   not add value over supervising the same moments.
4. **`independent_marginal` is clearly the worst** (about -3.4 EM points vs
   BCE), which confirms that the raw-sigmoid + count-CE control is not a free
   win and that the induced coupling matters for count calibration.
5. Therefore the defensible claim is **count-calibrated set supervision**, not a
   superior joint set objective. The exact-set EM claim should not be made.

This matches the decision rule in `docs/multiteacher_set_distillation.md`: if
only count information helps and exact set CE does not beat the induced-marginal
control, narrow the main claim to useful set statistics rather than a superior
optimization objective.

## Checkpoint/disk incident

The first seed replication (v1) failed at steps 180–270 with
`SafetensorError: Disk quota exceeded`: every EM improvement exported a new
full checkpoint (~10 GB), and `runs/qz` reached 749 GB. The trainer now deletes
the previous best checkpoint after exporting a new best (keeping the latest
selected checkpoint and the final export). After the fix, all eight v2 seed runs
completed. The failed v1 runs are not included in the table.

## Teacher status

- The teacher generation pipeline was plumbed on the base model with
  `TRUNCATED_POLICY=include`; the readout for 2 rows succeeded.
- At 4,096 max tokens, 151/155 of 200 chains per split were truncated, so
  `TRUNCATED_POLICY=exclude` leaves little coverage. The full teacher run is not
  started; a bounded pilot (K=2, 2 views, 8,192 tokens, include with a recorded
  truncation rate) is the next step.
- No teacher-target result is claimed here.

## Reproduction

Payloads: `qz_grouped/ms_grouped_*_4gpu*.json` and
`qz_grouped/ms_grouped_*_4gpu_s4*.json`. Commands run from this repository:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 4 harness/train_multiselect.py \
  --train data/multiselect_grouped_multi/train.jsonl \
  --dev data/multiselect_grouped_multi/dev.jsonl \
  --model /path/to/Qwen3.5-4B \
  --modules innerjev/configs/4B/modules.json \
  --target gold --objective set --set-loss induced_marginal --count-weight 1 \
  --output runs/qz/<arm> --lr 2e-6 --global-batch 64 --epochs 1 \
  --max-input-tokens 8192 --max-batch-tokens 8192 \
  --eval-every-frac 0.1 --lambda-g 0 --graph none --seed 43
```

The data split is in `data/multiselect_grouped_multi/`; its statistics are in
`data/multiselect_grouped_multi/summary.json`.

## Soft-teacher pilot on the 5k subset (10-23, single seed, preliminary)

A 5k stratified subset (`data/multiselect_grouped_multi_5k`, train 5,000 /
dev 1,000) was used for a first soft-teacher probe, seed 43 only, same 4×H100
recipe. The two `soft_gold*` arms are **gold controls on the same 5k subset**
(gold BCE and gold `set_ce`), so the soft-teacher arms are compared against an
equal-data gold baseline, not against the full 28k table above.

| Arm | Step | Question EM | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| gold BCE (5k control) | 79 | 0.5015 | 0.8855 | 0.8673 | 0.9045 |
| gold `set_ce` (5k control) | 64 | 0.3937 | 0.8554 | 0.7820 | 0.9439 |
| soft teacher `set_ce` | 48 | 0.2439 | 0.8115 | 0.6964 | 0.9722 |
| soft teacher `induced_marginal` | 64 | 0.2023 | 0.8058 | 0.6817 | 0.9851 |
| soft teacher `independent_marginal` | 48 | 0.1224 | 0.7817 | 0.6434 | 0.9960 |
| soft teacher `marginal` | — | (no `selection.json`) | | | |

Interpretation:

- On the same 5k subset, the soft-teacher arms are **strictly worse** than the
  gold control (best soft-teacher EM 0.2439 vs gold `set_ce` 0.3937 and gold
  BCE 0.5015). Raw soft teacher targets are not usable as-is.
- The soft-teacher arms collapse to over-prediction: recall saturates toward
  ~0.97–1.00 while precision falls to ~0.64–0.70, i.e. the model learns to mark
  nearly every candidate rather than to resolve the set.
- This is a single-seed 5k pilot and is reported as a negative signal only; it
  is not directly comparable to the full-data 3-seed table above.
- Teacher targets come from `teacher_gold5k_20261023`; the teacher-generation
  truncation issue described under "Teacher status" still applies.
