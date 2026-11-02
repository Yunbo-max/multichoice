# Paper 1 — EUR-LEX pilot results and corrected Set-MAP audit

Branch: research/static-setjev-paper-20261027. Pilot measured on 2026-11-01;
**critical mathematical correction dated 2026-11-02**.

**Status:** Qwen3.5-4B seed-43, single H100, EUR-LEX 100-label 5k Train /
1k DEV, 120 BCE warm steps + 200 matched continuation steps. These
are DEV pilot scores, NOT full official EUR-LEX Test results.
The published raw results were generated BEFORE the 2026-11-02
Independent Set-MAP fix and must not be used as a new final-method ranking.

## Confirmed training issue: count-head undertraining at K=100

Set CE contains a count cross-entropy component divided by K. At K=100,
the count-head gradient was initially weak: per-item count_ce changed
~0.0216 -> ~0.0214 over 80 steps. The --count-lr-mult control was
added (100 used for the rerun) and continuation increased to 200 steps.
Count LR multiplier 300 caused divergence. A correctly scaled
BCE + Gold Count baseline uses --gold-count-weight=100; these are
distinct head-LR and loss-scaling changes, so do not interpret them as
a randomized single-factor ablation.

## Retracted prior decoding explanation (important)

A November 1 code change claimed that a term "-log Z_m" wrongly
penalized larger counts in independent heads, because count mode
might be 5.73 while the most probable set has 2.02 selected options.

**That explanation was wrong.** A cardinality distribution assigns
AGGREGATE probability to all sets of size m, not to each individual
set. An m=5 count can have the most mass while an individual m=2 set
still has greater probability. This is mathematically expected.

For a normalized subset law, both count-head variants must decode
using the same formula:

    argmax_{m in public M} [log rho_m
                           + sum_{i in Top-m(s)} s_i
                           - log Z_m(s)].

- Independent Count Head: rho_m = softmax(c)_m. Keep "-log Z_m".
- Count Anchor: rho_m proportional exp(a_m+log Z_m(s)); after cancellation
  the MAP objective becomes a_m + sum_{i in Top-m(s)} s_i.

The 2026-11-01 code used the Anchor residual formula for BOTH heads,
which is incorrect and non-invariant under adding a constant to every
candidate score in the independent-head model.

**Fix commits, November 2:** set_distillation.py and
multiselect_metrics.py now both use the common normalized subset law.
Existing exact-enumeration, count-normalization, option-logit-shift,
and 100-option regression tests pass in GitHub CI.

## Historical DEV pilot table (do not treat retracted cells as corrected)

| arm | previously reported Set-MAP Micro-F1 | best sigmoid-threshold Micro-F1 | mean selected count under old decoder |
|---|---:|---:|---:|
| bce_continue | 0.4286 | 0.5005 (threshold .35) | 2.66 |
| bce_goldcount_w100 | **0.2849 (INVALID old Set-MAP)** | 0.3980 (.15) | 3.20 (old decoder) |
| set_ce_independent | **0.4619 (INVALID old Set-MAP)** | 0.4667 (.20) | 5.96 (old decoder) |
| set_ce_anchor | 0.4359 (Anchor MAP algebra unaffected) | 0.4703 (.15) | 3.86 |
| set_ce_anchor_detach | 0.4699 (Anchor MAP algebra unaffected) | 0.4752 (.25) | 5.33 |

Gold average count = 4.95 (range 1..12). All values above are historical
pilot DEV measurements; no corrected independent/goldcount DEV scores are
claimed yet. Threshold scores and model candidate logits did not change
under the standalone decoding patch; the training loss was not modified
by the November 2 correction.

**We must rerun the off-line predicted-set decoding before interpreting the
independent and goldcount Set-MAP results.**

## No-GPU procedure to correct previously saved DEV predictions

All gold and saved prediction files must refer to the SAME DEV samples
and answer-slot order. The helper refuses already-existing output dirs,
mismatched IDs/prompts/choices, unsupported count probabilities and TEST.

Example:

~~~bash
python -m paper_static.redecode_set_map \
  --gold data/eurlex_decision_v1/pilot/dev.jsonl \
  --predictions runs/paper1_eurlex_pilot_dev_eval/set_ce_independent/seed_43/predictions.jsonl \
  --output runs/paper1_eurlex_pilot_dev_eval_fixed/set_ce_independent/seed_43

python -m paper_static.redecode_set_map \
  --gold data/eurlex_decision_v1/pilot/dev.jsonl \
  --predictions runs/paper1_eurlex_pilot_dev_eval/bce_goldcount_w100/seed_43/predictions.jsonl \
  --output runs/paper1_eurlex_pilot_dev_eval_fixed/bce_goldcount_w100/seed_43
~~~

The tool reads original raw candidate logits and stable per-count log
probabilities, recomputes exact Set-MAP, and exports corrected
predictions.jsonl, metrics.json, eurlex_paper_metrics.json and an audit
manifest. Saved original predictions remain untouched. If the required
stable log probabilities were not exported by an earlier run, re-evaluate
the frozen model on DEV; do not invent tiny mass from clipped probabilities.

## What remains genuinely unresolved

Corrected seed-43 DEV Set-MAP after off-line redecoding
(`paper_static/redecode_set_map.py`, `runs/paper1_eurlex_pilot_dev_eval_fixed/`):

| arm | corrected Set-MAP micro-F1 | mean selected count |
|---|---:|---:|
| bce_continue | 0.4286 | 2.66 |
| bce_goldcount_w100 | 0.2124 | 2.02 |
| set_ce_independent | 0.4441 | 4.82 |
| set_ce_anchor | 0.4359 | 3.86 |
| set_ce_anchor_detach | 0.4699 | 5.33 |

Gold average count = 4.95. The corrected independent-head count (4.82) now
lands near gold, but the independent/goldcount Set-MAP micro-F1 is lower than
the previously reported (invalid) values, as expected from the normalized
subset law. The corrected ranking is: detach > independent > anchor >
bce_continue > goldcount on Set-MAP micro-F1, while calibrated BCE (0.5005)
remains above every set arm.

The pilot's best calibrated BCE micro-F1 is 0.5005, above Anchor's
best threshold micro-F1 (0.4703); the full Anchor Set-MAP was 0.4359.
The real benchmark has not demonstrated a superiority claim.

Next: compare proper Set NLL and Count NLL across heads and calibrated BCE;
then run matched seeds and full 55k official training only if there is an
evidence-supported reason to proceed. A single pilot cannot establish
conditional independence of EUR-LEX labels.
