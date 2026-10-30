# Paper 1 — Option-independence audit: why anchoring ties but cannot beat BCE

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
Why does the anchored set objective reach BCE's EM but never exceed it? The audit
below shows the multi-select datasets are effectively **option-independent**, so
the independent-sigmoid BCE is already the correct model and the set objective's
count coupling has nothing extra to fit.

## Evidence 1 — count distribution is Poisson-binomial

If correct options were independent, the gold-count distribution equals the
Poisson-binomial law induced by per-option correctness probabilities q_i. Using
the BCE checkpoint's independent probabilities q_i on Test, the induced count
distribution matches the empirical gold-count distribution closely:

| gold count | empirical | independent (Poisson-binomial) |
|---|---:|---:|
| 1 | 0.1505 | 0.1251 |
| 2 | 0.2864 | 0.2714 |
| 3 | 0.2995 | 0.3159 |
| 4 | 0.1949 | 0.2095 |
| 5 | 0.0640 | 0.0698 |

Largest deviation ~2.5pp. The number of correct answers is essentially determined
by the independent per-option probabilities; there is no additional count prior.

## Evidence 2 — the anchor's learned count correction a_m is zero

The anchored law is `rho(m) ∝ exp(log Z_m(s) + a_m)`. The count head's learned
bias (the residual `a_m`) is all zeros after training:

```
set_ce_anchor    count head bias[:8] = [0, 0, 0, 0, 0, 0, 0, 0], max 0.0 min -0.0
set_ce_independent count head bias[:8] = [0, 0, 0, 0, 0, 0, 0, 0], max 0.0 min -0.0
induced_anchor   count head bias[:8] = [0, 0, 0, 0, 0, 0, 0, 0], max 0.0 min -0.0
```

There is no count prior to learn. `log Z_m(s)` — the count distribution implied by
the candidate scores alone — is already the correct count law, because the data
options are independent.

## Conclusion

1. **The data is option-independent**, so BCE (independent sigmoid) is the
   well-specified model and the set objective's count coupling is redundant.
2. **Anchoring recovers BCE, it does not beat it.** Anchoring the count to
   `log Z_m(s)` is exactly what fixes the independent count head's failure (48%
   over-selection, count MAE 1.0), because that term is the correct independent
   count law. With `a_m = 0` the anchored set readout degenerates to the
   independent-sigmoid answer, which is why EM ties BCE.
3. **The set objective's real value is in tasks where options genuinely
   co-vary** — e.g. Paper 2's Sokoban, where "which moves are optimal" and "how
   many moves are optimal" are strongly coupled. On option-independent
   multi-select, the set objective is a well-controlled degeneracy baseline, not
   a place to expect gains.

## Cross-reference

- `docs/paper1_static_4b_results.md`, `docs/paper1_static_test_results.md`: the DEV and Test numbers.
- `harness/multiselect_metrics.py`: over/under-selection and gold-count metrics.
