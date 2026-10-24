# SetJev v2 Gate 3 warm results: count anchor wins, and warm start compounds it

Date: 2026-10-24. Status: warm Gate 3 completed; residual count head
(`--count-anchor`) beats the full count head for both loss families, and the BCE
warm start adds a further ~11pt over the cold anchor.

## Setup

All four arms: gold targets, `data/multiselect_grouped_multi_5k` (train 5,000 /
dev 2,042 for validation), seed 43, 79 steps, 4×H100, `lr 2e-6`,
`--eval-every-frac 0.2`. Warm start from `grouped_bce_4gpu_s44/v1/final`.
The only differences are `--set-loss` and `--count-anchor`.

## Results (dev, set-MAP decoder, selected checkpoint)

| arm | EM | F1 | precision | recall | selected step |
|---|---:|---:|---:|---:|---:|
| `set_ce` + anchor | **0.5705** | **0.9051** | 0.8916 | 0.9191 | 64 |
| `induced_marginal` + anchor | 0.5627 | 0.9037 | 0.8907 | 0.9171 | 64 |
| `induced_marginal` no anchor | 0.5049 | 0.8879 | 0.8362 | 0.9464 | 16 |
| `set_ce` no anchor | 0.4750 | 0.8775 | 0.8183 | 0.9459 | 16 |

Matched deltas (same loss, only `--count-anchor` differs):

| loss | Δ EM (anchor − noanchor) | Δ precision |
|---|---:|---:|
| `set_ce` | **+0.0955** | +0.0733 |
| `induced_marginal` | **+0.0578** | +0.0545 |

## Three findings

1. **The residual count head wins for both loss families.** The anchor lifts
   EM by +9.55pt (set_ce) and +5.78pt (induced), mainly through precision
   (+7pt) with recall only slightly lower (−3pt): the model stops over-selecting.

2. **The BCE warm start compounds the gain.** The warm anchor reaches 0.5705 vs
   the cold anchor's 0.4584 — a further +11.2pt. The warm start removes the
   degenerate cold-start count prior (see `setjev_v2_gate3_smoke.md`), so the
   residual head only has to learn a small correction.

3. **The anchor keeps training improving; the full count head over-fits early.**
   Both anchor arms select step 64 (still improving), while both no-anchor arms
   select step 16 (peak early, then degrade). This is the count/score coupling
   doing useful work: the count stays calibrated, so the candidate scores keep
   improving instead of being pushed into over-selection.

## Cross-reference

- `setjev_v2_gate1_readout_audit.md`: the teacher should use native option
  logits, not the Yes/No readout (EM 54.0 vs 45.3, BCE 0.30 vs 0.70).
- `setjev_v2_gate3_smoke.md`: the cold anchor is noisy early but converges and
  beats the cold no-anchor by +6.5pt.
- The warm anchor (0.5705 on 5k) exceeds the previous full-28k cold gold
  `set_ce` (0.5573) — warm start is worth more than the extra data here.

## Next

- Confirm with the remaining seeds (s44 / s45) before claiming a stable win.
- Fold the native-logit teacher (Gate 1) into the Count-Anchored teacher joint
  `Q_T(S|x) = t_|S| exp(sum u_i) / Z_|S|(u)` with `u_i` from native logits, then
  run the hard→soft KD arms (Gate 2).
