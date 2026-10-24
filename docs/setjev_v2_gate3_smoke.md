# SetJev v2 Gate 3 smoke: cold-base-start count anchor is unstable

Date: 2026-10-24. Status: empirical negative result on the cold-start variant;
fix is a BCE warm start (plan modification 3).

## What was run

`gate3-set_ce-anchor-5k-s43` (job `job-5c604c59-…`): gold `set_ce` with
`--count-anchor`, 5k subset, seed 43, 4×H100, ~79 steps. Cold start from the
base `Qwen3.5-4B` (the candidate scores are the frozen answer-slot head projected
from untrained internal layers).

## Result: unstable training

Compared against the matched non-anchor reference (`grouped_set_ce_4gpu_s44`,
gold `set_ce`, full 28k, no anchor):

| | non-anchor reference | anchor smoke (cold base start) |
|---|---:|---:|
| early loss | ~0.46–0.60, smooth | 0.6–6.8, oscillating |
| grad norm | 5.6–32.8 | **139–828 (10–25×)** |
| count CE | ~0.32, flat | 0.3–6.45, oscillating |

No NaN / non-finite values, but loss and count CE alternate every step, the
signature of two coupled dynamics fighting rather than converging.

## Root cause (verified, not a code bug)

The residual count head is `rho(m) = softmax[log Z_m(s) + a_phi(m,x)]` with
`a_phi = 0` at init. Two facts combine:

1. At cold start, the candidate scores `s` are the base model's uncalibrated
   answer-slot logits, so `log Z_m(raw)` is dominated by score magnitude and puts
   nearly all mass on the maximum allowed count. The initial count CE is ~6.45
   (vs ~0.32 for a uniform full count head), so the count head must learn a huge
   correction.
2. The anchor depends on the trainable candidate scores, so the count head and
   the candidate scores chase each other every step.

The gradient itself is correct: `torch_count_anchor_logits` (max-center +
`m·center` restore) and a direct `log Z_m(raw)` produce identical gradients
(diff ~4.8e-7, float32). The instability is a training-dynamics issue, not a
math bug.

This is exactly the "base start, not BCE warm start" caveat recorded in the
fix commit. The plan's modification 2 ("count head cold-starts from the
candidate-implied count") is only a *good* initialization when the candidate
scores are already calibrated, i.e. after the hard BCE stage (modification 3).

## Fix

Warm-start the Gate 3 arms from the BCE-trained checkpoint
(`runs/qz/grouped_bce_4gpu_s44/v1/final`) instead of the base model. The internal
language-model layers start BCE-calibrated, so `log Z_m(s)` is a reasonable
count prior and the residual head only has to learn the residual correction.

The generator now emits `--model <bce final>` for all four Gate 3 arms, so
"residual count head vs full count head" is compared on top of the same BCE
warm start. The cold anchor smoke is kept as the negative datapoint.
