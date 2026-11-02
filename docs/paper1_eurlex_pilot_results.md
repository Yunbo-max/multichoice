# Paper 1 — EUR-LEX pilot results and the two decode/budget fixes

Branch: `research/static-setjev-paper-20261027`. Date: 2026-11-01.
Seed-43 single-H100 LoRA pilot on the 100-label EUR-LEX main benchmark
(5k train / 1k dev), 120 BCE warm steps + 200 continuation steps. Two
implementation bugs were found and fixed during this pilot; the final numbers
below reflect both fixes.

## Bugs fixed

1. **Count-head underfit at K=100.** `count_ce = -(rho·log_count).sum()/k`
   divides the count loss by K, so at K=100 the count-head gradient shrinks
   100x and its weights stay ~1e-3 (count_ce 0.0216→0.0214 over 80 steps).
   Fix: `--count-lr-mult` (default 1), set to 100 for the set_ce arms, and
   raise continuation steps to 200. `--count-lr-mult 300` diverged.

2. **set-MAP over-penalises large counts for non-anchored heads.** The exact
   set MAP was computed as `log_rho[m] + prefix[m] - logz[m]`; for a
   non-anchored head `log_rho` lacks `log Z_m`, so `-logz[m]` shrinks the
   predicted count. goldcount's count mode was 5.73 but set-MAP chose 2.02.
   Fix: MAP = `argmax_m [ sum_{top-m} s_i + a_m ]` (raw residual + restored
   prefix). Anchor is unchanged (its `log Z_m` cancels).

## Results (seed 43, dev, 200 continuation steps)

| arm | set-MAP microF1 | best threshold microF1 (thr) | mean predicted count |
|---|---:|---:|---:|
| bce_continue | 0.4286 | **0.5005** (0.35) | 2.66 |
| bce_goldcount_w100 | 0.2849 | 0.3980 (0.15) | 3.20 |
| set_ce_independent | 0.4619 | 0.4667 (0.20) | 5.96 |
| set_ce_anchor | 0.4359 | 0.4703 (0.15) | 3.86 |
| set_ce_anchor_detach | 0.4699 | 0.4752 (0.25) | 5.33 |

Gold mean count is 4.95 (range 1–12).

## Decomposing the gap to calibrated BCE

Two distinct effects hold anchor below BCE:

1. **Candidate-score quality (~3pp).** The set objective's candidate term is
   the set-marginal CE `(q·s)/k`, constrained by the count law, whereas BCE is
   independent per-option sigmoid CE. On a 100-label task with ~5 positives per
   document, the independent sigmoid is a better fit: the best threshold decode
   of the set arms' sigmoid probabilities peaks at ~0.47 vs BCE's 0.5005.

2. **set-MAP hard count constraint (~3.5pp).** set-MAP commits to the top
   `argmax_m` options; when the count is off (3.86 vs 4.95) it truncates away
   correct labels. Threshold decoding of the same scores is more flexible and
   beats set-MAP on every set arm (e.g. anchor 0.470 vs 0.436).

## Interpretation

On real EUR-LEX the labels are nearly independent, so the count coupling is a
burden rather than a benefit: anchor's candidate scores are slightly worse than
BCE's and its set-MAP count constraint further degrades microF1. This is
consistent with the synthetic result — anchor wins on cardinality-dependent
data (K=8/16/32/64), where the count law is real, but not on near-independent
real multi-label data. The two fixes are important in themselves: the count-head
underfit and the `-log Z_m` set-MAP bug also affected the earlier independent-arm
numbers and should be re-audited.

## Cross-reference

- `docs/paper1_eurlex_execution.md`: the runbook.
- `docs/paper1_overview.md`: the three-regime framing and the "when to upgrade"
  claim.
- `docs/paper1_e4_k16_results.md`: the synthetic K-scaling result where anchor
  does beat BCE.
