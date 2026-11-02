# Paper 1 EUR-LEX — P1b frozen-score count-head results (3 seeds)

**Date:** 2026-11-02. Branch: research/static-setjev-paper-20261027.
**Status:** DEV-only. Frozen BCE backbone = `bce_continue/seed_43/final` (320
steps, DEV micro-F1 0.435). Option scores s_i, hidden h(x) and log Z_m(s)
exported once per Train (5000) / DEV (1000) row; then only a single zero-init
101-way linear head is trained on gold cardinality CE (no /K), 600 steps, batch
64, matched seed/LR. No gradient enters Qwen4B. Candidate scores are identical
across all frozen arms.

## Head families (same frozen features)

- `anchor`:    `softmax(a_m = Wh+b + log Z_m(s))`, zero init.
- `independent`: `softmax(c_m = Wh+b)`, zero init.
- `independent-prior`: `softmax(c_m)`, bias initialized to TRAIN count histogram.

## 3-seed × 2-LR summary (LR chosen by DEV Count NLL; means over 6 runs)

| arm | Count NLL | Joint-Set NLL | set-MAP micro-F1 | count MAE |
|---|---:|---:|---:|---:|
| bce_continue (Poisson-binomial) | 2.132 | 12.118 | 0.4286 | 2.64 |
| anchor (zero) | 2.10 | 12.09 | **0.446** | **2.06** |
| independent (zero) | 3.91 | 13.88 | 0.153 | 27.0 |
| independent (prior) | **1.98** | **11.97** | 0.427 | 2.35 |

Gold mean count = 4.95.

## Paired cluster bootstrap vs bce_continue (seed 43, 1000 resamples)

| arm | Δ dev-tuned micro-F1 | 95% CI | Δ Count NLL |
|---|---:|---:|---:|
| anchor | +0.0050 | [-0.0033, +0.0129] (n.s.) | -0.029 |
| independent (zero) | -0.0488 | [-0.0572, -0.0398] | +1.764 |
| independent (prior) | +0.0074 | [+0.0018, +0.0128] | -0.145 |

## Interpretation

1. **log Z_m is an optimization prior, not an expressive family.** Zero-init
   `anchor` trains fine (Count NLL 2.10) because `log Z_m(s)` already carries a
   strong cardinality signal, while zero-init `independent` collapses (Count
   NLL 3.91, count MAE 27). This is the anchor inductive bias in action.

2. **A TRAIN-count-prior init is a stronger prior than log Z_m.** `independent`
   with the train histogram bias reaches Count NLL 1.98 / Joint NLL 11.97, at
   least as good as anchor. So the anchor advantage is optimization/prior
   quality, not a new expressible set distribution — runbook rule #2.

3. **Set-MAP favors anchor; marginal-threshold favors prior.** anchor's exact
   set-MAP decodes with log Z_m cancelling, so its set-MAP F1 (0.446) is the
   highest; `independent-prior` learns a slightly better count law, giving the
   highest dev-tuned marginal F1 (0.5078, significantly above BCE's 0.5005 by
   +0.0074). The two decoders diverge (runbook rule #3).

4. **No large F1 win.** All frozen-head improvements over calibrated BCE are
   small (<1pp); the exact set-MAP remains below calibrated-BCE thresholding.
   The evidence is cardinality calibration (proper NLL), not a decision-level
   F1 gain.

## Conclusion for the research question

Freezing the BCE scores and training only the count head shows that the
Anchor parameterization helps *optimization* (stable from zero init) but is
not required for *calibration* once a train-count prior is available. On
EUR-LEX the count signal is already well captured by BCE's Poisson-binomial
count (2.13), so the practical NLL/F1 headroom is small. Per runbook decision
rule #2/#3, this does NOT demonstrate an Anchor-specific superior set family;
P2 (full 55k) is not gated on these numbers.
