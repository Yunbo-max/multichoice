# Paper 1 — Option independence: preliminary audit, **not proof** (2026-10-30)

**Correction:** an earlier version of this document concluded that CMB/MultiRC options are conditionally independent, based on aggregate count frequencies and an all-zero saved count-head **bias**. Neither observation establishes that conclusion.

## What was actually observed

- The average TEST histogram of the gold number of selected answers and the Poisson-binomial count histogram induced by BCE are close (largest reported count-frequency discrepancy about 2.5 percentage points).
- On the frozen, already-opened 2,140-question TEST, anchored Set CE roughly ties BCE continuation: exact-set EM 0.6173 versus 0.6182; independent count-head Set CE scores 0.4005.
- The reported count-head *bias* vector was zero for inspected checkpoints. This is not a measurement of a_m(x)=W_m h(x)+b_m. A zero bias is compatible with a nonzero, input-dependent residual and could also expose a training/export bug if all parameters are unexpectedly constant.

## Why the prior inference was invalid

Matching **aggregate** count distributions does not prove that Y_i and Y_j are conditionally independent given the full question/context x. Different joint set laws can have the same option marginals and cardinality law. For example, the distribution with equal mass on AB and CD and the distribution with equal mass on AC and BD have identical membership and count statistics yet different co-occurrence rules.

## Follow-up: measured residual a_m(x) and the zero-residual ablation

The anchored law is `rho(m) ∝ exp(log Z_m(s) + a_m)`, where `a_m(x) = W·hidden + b`
is the count head's output. The count head **parameters** are tiny (weight/bias
norms ~5e-5), but that is not the right test: forward-measured `a_m(x)` is ~1000×
larger than the parameter norm because the hidden activation is large.

Forward audit on DEV (seed 43, set_ce_anchor, 2042 rows, 12177 (row, m) items):

| quantity | max | mean | std |
|---|---:|---:|---:|
| \|a_m\| | 0.158 | 0.067 | 0.046 |
| \|log Z_m(s)\| | 24.27 | 6.56 | 4.86 |

Ratio `|a_m| / |log Z_m(s)|` is ~1–2.4% (mean). So `a_m` is **not** negligible in
the logit space (~0.1), which shifts count probabilities by ~10% relative. The
count head does learn a small real count prior.

But the impact is minor. Forcing `a_m = 0` and re-decoding the same candidate
logits changes only 56 / 2042 = **2.74%** of predicted sets, and moves EM by
-0.05pp (0.5686 → 0.5681), count NLL +0.0065, count Brier +0.0025, count MAE
-0.0020.

## Conclusion

1. **The data is nearly option-independent.** The gold-count distribution is
   Poisson-binomial, and the learned count prior `a_m(x)` is real but weak: it
   changes only 2.7% of predictions and 0.05pp of EM.
2. **Anchoring recovers BCE, it does not beat it.** Anchoring the count to
   `log Z_m(s)` is what fixes the independent count head's failure (48%
   over-selection, count MAE 1.0), because that term already carries the
   (nearly-independent) count law. The small residual `a_m` adds a minor count
   prior on top, but not enough to exceed BCE on EM.
3. **The set objective's real value is in tasks where options genuinely
   co-vary** — e.g. Paper 2's Sokoban, where "which moves are optimal" and "how
   many moves are optimal" are strongly coupled. On near-independent multi-select,
   the set objective is a well-controlled degeneracy baseline, not a place to
   expect gains.

## Cross-reference

- [Frozen 24-checkpoint TEST](paper1_static_test_results.md)
- [24-run matched DEV](paper1_static_4b_results.md)
- [Original count/SetJev controls](seed_results_2026-10-23.md)
- `paper_static/audit_count_head.py`: the forward audit that measures `a_m(x)` and the a_m=0 ablation.
- `harness/multiselect_metrics.py`: over/under-selection and gold-count metrics.
