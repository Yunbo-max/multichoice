# Paper 1 — E0/E1 results: mathematical boundaries and existing-anchor audit

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
E0 (existing 4B anchor re-audit) and E1 (controlled-regime oracle verification)
from the three-regime research design. These confirm the mathematical boundary
before committing GPU to the E2 controlled training.

## E1 — controlled-regime oracle verification (CPU)

Five CPU regression tests pass (25 cases): `test_paper1_controlled_regimes`,
`test_paper1_count_pairwise`, `test_paper1_oracle_capacity`, `test_paper1_regime_eval`,
`test_paper1_regime_plan`.

`regime_oracle_check.py` reports the exact joint-KL gap that each model family can
close, given the true generating law:

| regime | BCE excess joint KL | count-anchor excess | pairwise excess |
|---|---:|---:|---:|
| I (independent) | ~0 (-4e-16) | — | — |
| C (cardinality) | 0.2654 | 0.0 | — |
| R (pairwise) | 1.1255 | 0.8367 | ~0 (4e-15) |

Rewiring the pair matching (AB/CD/EF/GH → AC/BD/EG/FH) leaves option marginals
and the count law unchanged (`marginals_and_count_invariant_under_rewiring=true`),
but changes the co-occurrence `P(A,B)` from 0.267 to 0.089. So only a pairwise
model can tell the two apart.

The generator (`controlled_regimes.py`) emits I/C/R rows with zero-strength
negative controls, four draws per context, and pair-structure-disjoint
train/dev/test splits.

## E0 — existing 4B anchor re-audit (GPU, 6 DEV jobs)

Frozen DEV inference for `bce_continue` and `set_ce_anchor` (seeds 43/44/45),
then `paired_count_audit.py` over the saved predictions.

1. **The count residual is nonzero.** `count_residual_logits` (a_m(x)) has
   mean |a_m| = 0.067, max 0.158, consistent with the earlier `audit_count_head.py`
   forward audit. The parameter norm (~5e-5) is not the right test.

2. **The residual's effect is small on EM but real on count calibration.**
   Zero-residual ablation: EM +0.05pp (0.5681 → 0.5686), count NLL −0.0065.

3. **Anchor's count gain scales with BCE's count mismatch.** Stratifying DEV by
   BCE per-item gold-count NLL:

   | BCE count-NLL quartile | anchor count-NLL delta |
   |---|---:|
   | Q1 (BCE best-calibrated) | −0.0003 |
   | Q4 (BCE worst-calibrated) | −0.035 |

   Anchor repairs cardinality exactly where BCE is miscalibrated and stays inert
   where BCE is already correct. This is the empirical "when to upgrade from
   independent to count modeling" boundary.

## Next

E2 trains the same 4B backbone on the controlled I/C/R data under matched
budgets (BCE continuation, independent set CE, count anchor, anchor detach), to
test whether anchor actually learns the true cardinality correction when one
exists. `plan_regime_pilots.py` preregisters 13 jobs (1 warm BCE + 12 matched
continuations).

## Cross-reference

- `paper1_three_regime_research_design.md`: the full design.
- `paper1_count_pairwise_protocol.md`: the E0 protocol.
- `paper_static/regime_oracle_check.py`, `paper_static/controlled_regimes.py`,
  `paper_static/paired_count_audit.py`: the executables behind these numbers.
