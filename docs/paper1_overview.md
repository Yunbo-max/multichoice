> **2026-11-14 ICML-2027 plan update:** Paper 1 now prioritizes **RoMQA Closed Train/DEV** as a prospective real multi-constraint cardinality evaluation, keeping CMB/MultiRC and EUR-LEX as completed controls. This is a **code-only experiment preparation**: the official/third-party RoMQA archive has not been downloaded or independently checked, no RoMQA 4B GPU experiment has run, and no new positive result is claimed. See [RoMQA scientific protocol + paper outline](paper1_romqa_icml2027_protocol_20261114.md), [conversion](../paper_static/romqa_data.py), [shared frozen-score controls](../paper_static/romqa_frozen.py), [DEV scorer](../paper_static/romqa_metrics.py), and [plan-only scripts](../paper_static/romqa_plan.py). This updates the historical 2026-11-01 "EUR-LEX only" scope on this NEW research branch without rewriting the original branch's evidence.

> **Scope update, 2026-11-01:** New Paper 1 main experiment is **EUR-LEX only**. [Locked single-benchmark protocol](paper1_eurlex_only_main_experiment.md). Any RoMQA, ECtHR, SATA or other benchmark in this document is historical/planned, not part of the currently approved run. Prior E0–E4 experiments remain historical results; no additional GPU jobs are implied.

# Beyond Independent Choices: When Do Decision Models Need Structured Set Prediction?

Branch: `research/static-setjev-paper-20261027`. A consolidated summary of
Paper 1 (E0–E4). The scientific question is *not* "does anchor beat BCE on some
benchmark", but: **under what conditions must a multi-answer decision model
upgrade from independent option prediction (BCE) to cardinality modeling
(count anchor) to joint relation modeling (pairwise)?**

## The unified model

`P(Y|x) ∝ exp( Σ_i s_i(x) y_i + a_|Y|(x) + Σ_{(i,j)∈E(x)} J_ij(x) y_i y_j )`

with three nested conditions:

| regime | generating law | correct model |
|---|---|---|
| I (independent) | `exp(Σ s_i y_i)` | BCE |
| C (cardinality) | `exp(Σ s_i y_i + a_|Y|)` | count anchor |
| R (pairwise) | `exp(Σ s_i y_i + a_|Y| + Σ J_ij y_i y_j)` | pairwise |

## What the experiments established

### E1 — mathematical boundaries (CPU)

Exact oracle joint-KL gaps each family can close:

| regime | BCE excess KL | count-anchor excess | pairwise excess |
|---|---:|---:|---:|
| I | ~0 | — | — |
| C | 0.2654 | 0.0 | — |
| R | 1.1255 | 0.8367 | ~0 |

Rewiring the pairing (AB/CD/EF/GH → AC/BD/EG/FH) leaves option marginals and
the count law unchanged — only a pairwise model can tell the two apart.

### E0 — existing anchor on real data (CMB/MultiRC)

The count residual `a_m(x)` is nonzero (mean 0.067, max 0.158), and its benefit
scales with BCE's count miscalibration: on the worst-calibrated quartile the
anchor improves count NLL by −0.035 versus −0.0003 on the best. The real
CMB/MultiRC data is near-independent (Poisson-binomial count law), so this is
the anchor's *degeneracy* regime, not where it should win.

### E2 / E4 — controlled cardinality, growing with K

On synthetic C-regime data the anchor beats BCE on count KL, and the gap widens
monotonically with the candidate count:

| K | anchor count KL | BCE count KL | anchor − BCE |
|---|---:|---:|---:|
| 8 | 0.278 | 0.333 | −0.055 |
| 16 | 0.793 | 1.065 | −0.27 |
| 32 | 1.229 | 2.483 | −1.25 |
| 64 | 2.351 | 4.477 | **−2.13** |

This is the central positive result: **cardinality modeling pays off more as the
decision set grows**, because the independent-sigmoid Poisson-binomial law drifts
further from the true count distribution while the anchored count head tracks it.
The O(K²) log-partition DP (`paper_static/controlled_c_large.py`) makes K=255
feasible without 2^K enumeration.

### E3 — pairwise is an open frontier (negative)

A learned dense-J head, a greedy disjoint-block matching, and a Gumbel-perturbed
matching all fail to recover the true pairing (edge F1 ≈ chance 0.143). The
pairing shifts the joint law by only ~0.84 nats of KL while the cardinality term
dominates, so the joint set CE gradient cannot commit the pair head at this data
scale. Pairwise needs a stronger inductive bias (supervised edges or an explicit
matching parameterization). `paper_static/block_pair_dp.py` already provides the
exact O(K²) inference for that path.

## Claim, sharpened

1. **Independent data → BCE is optimal.** Structured losses do not add stable
   value (E0, E2-I).
2. **Cardinality-dependent data → count anchor wins, increasingly with K**
   (E2, E4: −0.055 → −2.13 nats).
3. **Pairwise-dependent data → neither is sufficient**; joint-relation learning
   needs a dedicated matching component (E3 negative).

## Cross-reference

- `docs/paper1_three_regime_research_design.md`: full design and gates.
- `docs/paper1_e0e1_results.md`, `docs/paper1_e2_results.md`,
  `docs/paper1_e4_k16_results.md`, `docs/paper1_e3_dense_j_negative.md`: stage results.
- `docs/paper1_option_independence.md`: the real-data near-independence audit.
