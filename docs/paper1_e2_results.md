# Paper 1 — E2 controlled 4B results: anchor wins on cardinality, BCE wins on independence

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
The 13-job controlled pilot (1 warm BCE + 12 matched LoRA continuations) on the
three-regime synthetic data, evaluated against the oracle joint laws. Mean over
seeds 43/44/45, DEV.

## Setup

- Backbone Qwen3.5-4B, LoRA r=16/a=32, batch 64, lr 2e-5.
- Stage 1: common BCE warm start, 450 steps (28,800 rows).
- Stage 2: 4 arms x 3 seeds, 100 matched continuation steps from the shared
  warm checkpoint — BCE continuation, independent set CE, count anchor, and
  anchor-detach.
- Data: controlled I/C/R regimes with zero-strength negative controls,
  four draws per context, pair-structure-disjoint splits.
- Metric: oracle joint KL (0 = exact representation of the true set law) and
  count KL, via `regime_eval.py`.

## Results (mean joint KL over seeds)

| arm | I (independent) | C (cardinality) | R (pairwise) |
|---|---:|---:|---:|
| bce_continue | **0.107** | 0.419 | 0.992 |
| set_ce_independent | 0.286 | 0.492 | 0.789 |
| **set_ce_anchor** | 0.181 | **0.366** | 0.780 |
| set_ce_anchor_detach | 0.140 | 0.475 | 0.796 |

## Results (mean count KL over seeds)

| arm | I | C | R |
|---|---:|---:|---:|
| bce_continue | 0.031 | 0.333 | 0.337 |
| set_ce_independent | 0.200 | 0.401 | 0.129 |
| set_ce_anchor | 0.098 | **0.278** | 0.122 |
| set_ce_anchor_detach | 0.059 | 0.395 | 0.137 |

## Interpretation

1. **Independent data: BCE is best.** On I, BCE has the lowest joint KL (0.107);
   anchor (0.181) and detach (0.140) do not collapse to BCE exactly, but the
   gap is small and the detach arm — which forces a_m=0 — is closer to BCE, as
   expected.

2. **Cardinality-dependent data: anchor beats BCE.** On C, anchor's joint KL
   (0.366) is lower than BCE (0.419) and count KL (0.278) lower than BCE (0.333),
   while independent set CE is worst (0.492). The count anchor learns the true
   cardinality correction; this is the preregistered "cardinality-dependent"
   hypothesis confirmed.

3. **The gradient-feedback path is load-bearing.** Detach is better than anchor
   on I (0.140 vs 0.181, where a_m should be zero) and worse on C (0.475 vs
   0.366, where a_m must be learned). The anchor's score-gradient through
   log Z_m(s) is exactly what lets it recover cardinality when it exists.

4. **Pairwise data needs pairwise capacity.** On R every arm fails (0.78-0.99),
   since none has a learned pair head. This is the E3 frontier.

## Cross-reference

- `docs/paper1_three_regime_research_design.md`: the full design and hypotheses.
- `docs/paper1_e0e1_results.md`: the oracle boundaries and existing-anchor audit.
- `paper_static/plan_regime_pilots.py`, `paper_static/regime_eval.py`: the E2 planner and oracle evaluator.
