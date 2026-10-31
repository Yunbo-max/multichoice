# Paper 1 — Count mismatch and pairwise dependence: executable diagnostic (2026-10-30)

**Status:** code and CPU unit/synthetic checks; no new GPU inference or real-model improvement is claimed by this addition. Do not use any already-opened TEST outputs to choose methods. The existing 24-run, 3-seed, frozen 2,140-question TEST already shows anchored Set CE (0.6173 EM) roughly ties BCE continuation (0.6182). These are warm starts from the same S44 BCE checkpoint, *not* 5k-from-scratch comparisons.

## Scientific hypotheses (separate these!)

- **H0 / degeneracy:** when the count prior implied by candidate logits is already right, Count Anchor should recover BCE rather than guarantee an EM improvement. a_m(x)=0 is a legal submodel; it is exactly BCE only when all counts 0..K are permitted, and conditional BCE when the protocol prohibits empty sets.
- **H1 / count mismatch:** when candidates are correctly ranked but BCE's count distribution is wrong, a learned count residual can improve *count NLL*, and may improve exact-set EM. It is not mathematically guaranteed to improve EM. A per-item -log P_BCE(|Y|) is a *label-dependent retrospective difficulty score*; it is not the unknown conditional KL or a deployable router.
- **H2 / pair dependence:** even with perfectly matched membership q_i and count rho_m, pairwise co-occurrence may remain unidentified. Count Anchor cannot represent distributions that distinguish two same-size sets with equal additive option scores. A learned pairwise potential can; exact partition functions are exponential in general.
- **H3 / fair baselines:** compare BCE at the same starting checkpoint, data, optimizer steps, decoder definition and tuning budget. Add BCE + learned count / Top-k and BCE + constrained reranking before attributing any improvement specifically to anchored gradients.

## DEV replay from the existing checkpoints (six jobs, no training)

Run at the QZ checkout of research/static-setjev-paper-20261027 after confirming that all six frozen trained checkpoint directories exist. Source templates are the existing qz_static_eval/paper1-eval-*.json TEST jobs; the generator *converts* them to DEV, verifies branch, refuses an existing output directory, and removes the original rm -rf behavior. **It never submits any job.**

~~~bash
# Six pre-generated QZ DEV payloads are already committed in qz_static_count_dev/.
# Optional: regenerate safely into a NEW folder, without overwriting the committed versions:
python -m paper_static.plan_dev_eval --output-dir qz_static_count_dev_regenerated
# Inspect the committed payloads and launch separately (requires QZ login and compute):
qz train CreateJob --data "$(cat qz_static_count_dev/paper1-dev-bce_continue-s43.json)"
qz train CreateJob --data "$(cat qz_static_count_dev/paper1-dev-set_ce_anchor-s43.json)"
# Repeat for s44 and s45; check job exit codes, logs and predictions.jsonl.

# After every DEV job completed successfully, on the same remote checkout:
python -m paper_static.paired_count_audit \
  --gold data/multiselect_grouped_multi/dev.jsonl \
  --predictions-root runs/paper_static_dev_eval/qwen3.5-4b-bce-s44 \
  --seeds 43 44 45 --bootstrap 1000 \
  --output runs/paper1_count_audit/paired_count_dev.json \
  --per-question runs/paper1_count_audit/paired_count_dev.jsonl
~~~

Evaluator outputs include candidate_logits, p, predicted_indices, count_probs, *stable* count_log_probs and newly exported count_residual_logits for structured arms. The last is the full input-dependent count head output, **not just its bias**. All prediction identities, option label order and prompt digest must match the gold DEV rows. Wrong checkpoint/seed, missing ID, stale evaluator, missing raw logits, wrong split, or an existing result output cause a hard failure. No fallback to clipped probabilities.

The audit uses public allowable counts to compute BCE's conditional Poisson-binomial law from raw logits by O(K^2) log-domain DP. For each seed, rank the **same DEV examples** by BCE's *gold-count* NLL and form stable rank quartiles Q1..Q4. Compare paired EM, count MAE, count NLL, strict correct-vs-incorrect ranking, and anchor a_m variation. Cluster-bootstrap DEV deltas by source/passage group (within seed; 95% percentile CI). Each seed receives its own quartile boundaries; do **not** pool 3 predictions of the same question as 3 independent samples. These are exploratory post-hoc strata; cannot claim causal effect or prospective test from this stratification alone.

**Critical extra ablation (no extra GPU):** using the anchor checkpoint's *same candidate logits*, force a_m=0 and recompute count NLL and legal-set MAP. Comparing full anchor with a=0 determines whether the learned residual actually affects predictions. A zero saved bias alone proves nothing: a_m(x)=W_m h(x)+b_m, so check W, input-dependent outputs and the shared-shift-invariant per-example residual span. The raw head outputs are only present after the evaluator export patch; re-run DEV inference to populate them.

## Controlled count-shift test (next training experiment, not executed)

Do not alter old sealed TEST. Use train/DEV splits with matched K, sources and candidate ranking difficulty while shifting target set-size frequencies. Pre-register count bins, seeds, and a separate held-out evaluation partition **before** training. Hold base and update budget constant and include BCE-threshold tuning and an ordinary learned Top-k baseline. If Anchor improves count NLL but not EM, report that narrower result.

## Exact pairwise mathematical control (CPU, not real benchmark)

~~~bash
python -m paper_static.pairwise_exact
python -m unittest discover -s tests -p 'test_paper1_count_pairwise.py' -v
~~~

Define E(Y)=sum_i s_i y_i + a_|Y| + sum_{i<j}J_ij y_i y_j and P(Y)=exp(E)/Z with public count support. pairwise_exact.py enumerates 2^K **only for K<=12** and uses a toy mixture with equal probability for {A,B} and {C,D}. This target has q_i=1/2 and exactly two true options but zero probability on cross-pairs. Independent BCE, Count Anchor (given exactly two), and a trained small exact pairwise J have distinct joint likelihoods; no GPU or model-generalization claim is implied. The reference pairwise fit holds candidate logits fixed and updates only J from expected pair indicators.

The full-size 255-option model **is not implemented** by this prototype. Large-K structured models need controlled interaction graph width, a tractable structured family, pseudolikelihood or approximate inference, calibrated partition estimates, and fair speed/accuracy comparisons. A generic dense CRF is not new by itself.

## Decision gates for a future arXiv claim

1. If Anchor ~ BCE on DEV count NLL/EM and a=0 does not change performance, claim a *boundary of usefulness*, not superior multi-select accuracy.
2. If Anchor beats the same-checkpoint BCE + tuned-threshold and BCE + learned-count baselines across 3–5 seeds and independent sources, examine whether improvement comes from feedback gradients, initialization, or extra parameters (detach/head-only/learning-rate controls).
3. If only Pairwise helps when the target has same marginals/count but distinct valid combinations, the method claim must shift to relational modelling; existing CRF, Ising, and constrained-inference baselines are compulsory. Report constraint validity and joint NLL as well as EM (an ambiguous valid-set target can have at most 50% deterministic hard-set EM).
4. Do not tune a model, cutoff, method selection gate, pairwise graph or other hyperparameters on the old already-opened 2,140-question TEST; create a genuinely independent held-out evaluation source.
