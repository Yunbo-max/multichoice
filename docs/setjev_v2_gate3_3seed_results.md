# SetJev v2 Gate 3 three-seed results: count anchor wins, and is far more stable

Date: 2026-10-25. Status: 3-seed confirmation complete. All warm Gate 3 arms
(s43/s44/s45, 5k, BCE warm start from `grouped_bce_4gpu_s44/final`, 79 steps,
4×H100) have finished.

## Results (dev, set-MAP decoder, selected checkpoint, mean ± std over 3 seeds)

| arm | mean EM | std | min–max | selected steps |
|---|---:|---:|---:|---|
| `set_ce` + anchor | **0.5640** | 0.0057 | 0.5597–0.5705 | 64 / 64 / 79 |
| `set_ce` no anchor | 0.4659 | 0.0318 | 0.4305–0.4922 | 16 / 16 / 16 |
| `induced_marginal` + anchor | **0.5624** | 0.0006 | 0.5617–0.5627 | 64 / 16 / 64 |
| `induced_marginal` no anchor | 0.4977 | 0.0291 | 0.4657–0.5225 | 16 / 79 / 16 |

Matched per-seed deltas (anchor − no anchor), every seed positive:

| loss | s43 | s44 | s45 | mean |
|---|---:|---:|---:|---:|
| `set_ce` | +0.0955 | +0.1312 | +0.0675 | **+0.0981** |
| `induced_marginal` | +0.0578 | +0.0970 | +0.0392 | **+0.0647** |

## Three findings

1. **The residual count head wins in every seed, for both loss families.** No
   seed shows the full count head beating the anchor.

2. **The anchor is far more stable.** Anchor std is 0.0057 (set_ce) and 0.0006
   (induced), versus 0.0318 and 0.0291 for the no-anchor controls — a 5–50×
   reduction. The induced anchor is nearly seed-invariant (0.5627 / 0.5627 /
   0.5617).

3. **The anchor matches gold BCE with less data.** Both anchor arms (0.5640,
   0.5624) are at or above the full-28k 3-seed gold BCE reference
   (0.5615 ± 0.0041), using only the 5k subset.

The selected-step pattern is consistent with the single-seed run: anchor arms
select late checkpoints (64–79, still improving) while no-anchor arms peak early
(step 16) and degrade — the anchor keeps the count calibrated, so the candidate
scores keep improving.

## Cross-reference

- `setjev_v2_gate1_readout_audit.md`: the teacher should use native option logits.
- `setjev_v2_gate3_warm_results.md`: the single-seed (s43) precursor.
- `setjev_v2_gate3_smoke.md`: the cold anchor is noisy early but converges.

## Next

- This closes the Gate 3 (residual count head) question for the static
  multi-select task. The remaining gates are the teacher readout (Gate 1 done),
  the Count-Anchored teacher joint with a native-logit `u_i` + gate-triggered
  reasoning (Gate 2), and the hard→soft KD arms.
