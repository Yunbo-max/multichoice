# Paper 1 EUR-LEX — P0 offline DEV diagnostics (seed 43)

**Date:** 2026-11-02. Branch: research/static-setjev-paper-20261027.
**Status:** DEV-only, no GPU, no new training. Reuses the already-saved
120-step warm + 200-step continuation pilot predictions on the same 1000
DEV rows (gold sha256 `5c4326e4e4...`). Independent/goldcount Set-MAP were
re-decoded with the corrected normalized subset law
(`paper_static/redecode_set_map.py`); anchor/detach are unchanged.

Command: `python -m paper_static.eurlex_p0_diagnostics --bootstrap 1000`
(see `docs/paper1_eurlex_p0_p1_p2_runbook.md`). No automated superiority
claim is emitted; interpret the proper-NLL and candidate-oracle gap below.

## 1. Proper probability quality (all-count support 0..K, lower is better)

| arm | Count NLL | Proper Joint-Set NLL | Count Brier | count-mode acc |
|---|---:|---:|---:|---:|
| bce_continue (Poisson-binomial) | **2.132** | **12.118** | **0.855** | — |
| goldcount (gold-count CE head) | **2.052** | 14.928 | 0.854 | 0.150 |
| anchor | 6.125 | 16.709 | 1.225 | 0.246 |
| detach | 5.819 | 16.354 | 1.575 | 0.126 |
| independent | 7.633 | 18.148 | 1.518 | 0.127 |

Reading: the plain BCE sigmoid model already induces a well-calibrated
cardinality distribution (Poisson-binomial Count NLL 2.132, statistically
indistinguishable from the explicitly gold-count-trained head 2.052). The
set-CE count heads (anchor/detach/independent) are far worse on Count NLL
(5.8–7.6) and on proper Joint-Set NLL (16.3–18.1) than plain BCE (12.12).
This is the 120+200-step undertraining signature: the count heads never
reached a calibrated cardinality distribution.

## 2. Decoder micro-F1 (same DEV)

| arm | fixed-thr .5 Set-MAP | count-mode top-k | DEV-tuned thr | candidate-oracle (gold count + top-k, NOT deployable) |
|---|---:|---:|---:|---:|
| bce_continue | 0.4286 | 0.4968 | **0.5005** (.35) | **0.5182** |
| independent | 0.4441 | 0.4617 | 0.4667 (.20) | 0.4901 |
| goldcount | 0.2124 | 0.3693 | 0.3980 (.15) | 0.3542 |
| anchor | 0.4359 | 0.4333 | 0.4703 (.15) | 0.4788 |
| detach | 0.4699 | 0.4593 | 0.4752 (.25) | 0.4780 |

Mean gold count = 4.95. The **candidate-oracle** column is the decisive
number: it uses each arm's own option scores with the GOLD count, so it
isolates option-ranking quality from count calibration. BCE has the best
option ranking (0.5182); every set-CE arm ranks options worse (anchor
0.4788, detach 0.4780, independent 0.4901, goldcount 0.3542).

## 3. Paired bootstrap vs BCE (dev-tuned), 1000 cluster bootstraps

| arm | Δ micro-F1 | 95% CI |
|---|---:|---|
| anchor | -0.0302 | [-0.0376, -0.0221] |
| detach | -0.0252 | [-0.0331, -0.0175] |
| independent | -0.0338 | [-0.0421, -0.0255] |
| goldcount | -0.1025 | [-0.1121, -0.0927] |

All count-head arms are significantly WORSE than dev-tuned BCE on micro-F1.

## P0 conclusion

1. **Count calibration:** BCE's induced cardinality is already as good as
   the gold-count head (Count NLL 2.13 vs 2.05); the anchor/independent
   heads are far worse — i.e. undertrained at 120+200 steps, not a proven
   family advantage.
2. **Option ranking:** anchor/detach rank options worse than BCE
   (oracle 0.478 vs 0.518), so even a perfect count cannot close the gap
   on this pilot.
3. **No Anchor superiority is demonstrated.** This is the runbook decision
   rule #4 negative-test case, but it is confounded by (a) count-head
   undertraining and (b) candidate-score drift away from BCE in the joint
   LoRA training.

## Decisive next step

P1b (frozen-score control) is the clean test that removes both confounds:
freeze the BCE backbone, option scores s_i and log Z_m, and train ONLY the
count head (anchor vs independent, matched seed/LR/steps). Watch whether the
frozen anchor beats the frozen independent AND BCE on proper Count NLL /
Joint-Set NLL without touching the candidate oracle (which is then identical
across arms). If the frozen anchor does not beat BCE's Poisson-binomial
count on proper NLL, do NOT escalate to 55k.
