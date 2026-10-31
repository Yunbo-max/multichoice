# Paper 1 — Static SetJev 4B matched matrix: 24 runs (8 arms × 3 seeds)

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-27.
24 matched 4×H100 runs from the pre-registered manifest
`paper_static/runs_4b.jsonl`, 5k gold continuation of the shared BCE-S44
checkpoint, 1 epoch / 79 terminal-only updates, batch 64, lr 2e-6, seeds
43/44/45. Aggregate DEV only; no Test was opened. Audited by
`paper_static/collect_results.py` (all 24 COMPLETE, no SHA/step/seed mismatch).

## Exact-set EM and calibration (mean ± std over 3 seeds)

| arm | question_em | f1 | count_nll | count_mae | joint_set_nll |
|---|---:|---:|---:|---:|---:|
| bce_continue | 0.5624 ± 0.0062 | 0.9041 | — | — | — |
| set_ce_anchor | **0.5633 ± 0.0052** | 0.9039 | 0.9134 | 0.4664 | 1.5200 |
| set_ce_anchor_detach | 0.5493 ± 0.0059 | 0.9032 | 0.9785 | 0.5157 | 1.5818 |
| set_ce_independent | 0.4278 ± 0.0151 | 0.8695 | 1.4915 | 0.7963 | 2.0977 |
| induced_anchor | 0.5622 ± 0.0021 | 0.9035 | 0.9199 | 0.4656 | 1.5381 |
| induced_anchor_detach | 0.5500 ± 0.0021 | 0.9037 | 0.9636 | 0.5122 | 1.5660 |
| induced_independent | 0.4809 ± 0.0143 | 0.8860 | 1.4679 | 0.6386 | 2.2088 |
| bce_plus_gold_count | 0.4564 ± 0.0179 | 0.8756 | 1.4964 | 0.7473 | 2.1061 |

## Paired by-seed deltas (question_em)

| comparison | mean_delta | seed 43 / 44 / 45 |
|---|---:|---|
| set_ce_anchor − set_ce_independent | **+0.1355** | +0.1293 / +0.1513 / +0.1259 |
| induced_anchor − induced_independent | **+0.0813** | +0.0803 / +0.0955 / +0.0681 |
| set_ce_anchor − set_ce_anchor_detach | +0.0140 | +0.0176 / +0.0083 / +0.0162 |
| induced_anchor − induced_anchor_detach | +0.0122 | +0.0171 / +0.0098 / +0.0098 |
| set_ce_anchor − bce_continue | +0.0010 | −0.0005 / +0.0034 / 0.0000 |
| induced_anchor − bce_continue | −0.0002 | −0.0049 / +0.0020 / +0.0024 |

## Interpretation

1. **Count anchor strongly repairs the independent count head.** Anchored set CE
   beats independent count CE by +13.55pp and anchored induced-marginal beats its
   independent control by +8.13pp, uniformly across all three seeds. Count NLL,
   Brier and MAE improve by the same order (e.g. set CE count NLL 1.4915 → 0.9134).
   This is the mechanism the paper preregistered: the residual `a_m` reparameterizes
   the independent count logits and avoids the poor independent-head initialization.

2. **Anchored set supervision does not beat the matched BCE continuation on EM.**
   `set_ce_anchor − bce_continue` is +0.10pp and `induced_anchor − bce_continue`
   is −0.02pp, both mixed-sign across seeds. Per the preregistered rejection rule,
   the claim is therefore limited to *mitigation of poor independent-count
   initialization*, not a superior joint set objective over plain BCE.

3. **The gradient-feedback path is positive but small.** Anchored beats its
   detached twin by +1.40pp (set CE) and +1.22pp (induced) on EM in every seed,
   so the `log Z_m(s)` score gradient is not inert; but the effect is an order of
   magnitude smaller than the independent→anchor repair, so the covariance
   mechanism should not be overstated.

## Cross-reference

- `paper_static/EXPERIMENTS.md`: the locked protocol and preregistered rejection tests.
- `paper_static/results_4b.json`: the full audit output (rows, arm summaries, paired deltas).
- `docs/seed_results_2026-10-23.md`: the earlier 450-step warm-start results.
