# Paper 1 — Static SetJev frozen Test evaluation (24 checkpoints)

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
Frozen Test (2,140 questions) run once on each of the 24 terminal checkpoints at
threshold 0.5, decoder `auto`. No selection on Test. Mean over seeds 43/44/45.
New diagnostics added: `over_selection_rate` / `under_selection_rate` and a
`gold_count` group (1 / 2 / 3 / 4+ correct answers).

## Test exact-set EM and cardinality (mean ± over seeds)

| arm | question_em | f1 | count_accuracy | count_mae | over_sel | under_sel |
|---|---:|---:|---:|---:|---:|---:|
| bce_continue | 0.6182 | 0.9035 | 0.6699 | 0.3978 | 0.2101 | 0.1199 |
| set_ce_independent | 0.4005 | 0.8289 | 0.4290 | 1.0000 | 0.4783 | 0.0927 |
| set_ce_anchor | 0.6173 | 0.9038 | 0.6709 | 0.3961 | 0.2098 | 0.1193 |
| set_ce_anchor_detach | 0.5816 | 0.8949 | 0.6226 | 0.4908 | 0.2913 | 0.0861 |
| induced_independent | 0.4525 | 0.8496 | 0.4907 | 0.8097 | 0.4266 | 0.0827 |
| induced_anchor | 0.6168 | 0.9036 | 0.6707 | 0.3953 | 0.2086 | 0.1207 |
| induced_anchor_detach | 0.5911 | 0.8977 | 0.6318 | 0.4746 | 0.2808 | 0.0874 |
| bce_plus_gold_count | 0.4280 | 0.8380 | 0.4570 | 0.9321 | 0.4491 | 0.0939 |

## Gold-count grouped EM (bce_continue vs anchors)

| gold_count | bce_continue | set_ce_anchor | induced_anchor |
|---|---:|---:|---:|
| 1 | 0.6884 | 0.6822 | 0.6812 |
| 2 | 0.5856 | 0.5949 | 0.5949 |
| 3 | 0.6058 | 0.5965 | 0.6006 |
| 4+ | 0.6277 | 0.6283 | 0.6223 |

## Interpretation

1. **The anchor's effect is over-selection repair, not a BCE win.** The
   independent count head massively over-selects (over_sel 0.48 set CE, 0.43
   induced; count MAE 1.00 and 0.81), collapsing EM to 0.40 / 0.45. Anchoring
   brings over-selection back to the BCE level (~0.21) and count MAE to ~0.40,
   recovering EM to 0.617. The detach control sits in between (over_sel ~0.29),
   consistent with the gradient-feedback path contributing part of the repair.

2. **Anchored set supervision does not beat BCE on Test.** set_ce_anchor
   (0.6173) and induced_anchor (0.6168) tie bce_continue (0.6182), matching the
   DEV conclusion. The gold-count breakdown shows no advantage on harder
   multi-answer questions (2/3/4+), so the preregistered "narrow scope" claim
   stands: the contribution is mitigation of poor independent-count
   initialization, not a superior joint set objective.

3. **Metrics are consistent between DEV and Test** (anchor ≈ BCE, anchor ≫
   independent, detach in between), so the mechanism conclusion is stable across
   the internal holdout and the frozen test split.

## Cross-reference

- `paper_static/results_4b.json`: DEV audit output.
- `docs/paper1_static_4b_results.md`: the DEV results.
- `harness/multiselect_metrics.py`: the added over/under and gold_count metrics.
