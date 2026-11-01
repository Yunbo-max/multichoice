# Paper 1 — Anchor-first complete benchmark protocol
**Research plan dated 2026-11-01. Not new results or a submitted GPU job.**

## Decision

Submit a focused first paper on COUNT-ANCHORED SET DECISION. Defer learning
dense/greedy/Gumbel pairwise matching, since E3 edge F1 is around chance.
Keep pairwise as a formal expressivity counterexample and future work.
Primary question: WHEN is independent BCE enough, and WHEN do residual
cardinality potentials improve count calibration and exact answer sets,
relative to strong independent and learned-count baselines?

## Existing datasets: preserve their official splits

1. MAIN: LexGLUE EUR-LEX (configuration eurlex). Official 55,000 train,
   5,000 validation, 5,000 test; fixed 100 EuroVoc concepts. Use ALL
   official 100 choices for every document. Do not sample positive/negative
   evaluation choices using hidden gold labels. Freeze one authoritative
   EuroVoc ID -> English descriptor mapping (and hashes); IDs alone are
   not meaningful descriptions. Maintain original chronological splits.
   Official references: https://github.com/coastalcph/lex-glue and
   https://huggingface.co/datasets/coastalcph/lex_glue.

2. CROSS-DOMAIN: LexGLUE ECtHR Task A, 9,000 train, 1,000 validation,
   1,000 test, 10 Article labels. Treat empty answer sets according to
   its public task protocol, not based on each row's gold count. Long
   facts require fixed, auditable head+tail truncation. Train matched
   model arms separately and do not describe that as zero-shot transfer.

3. EXTERNAL: SATA-BENCH, 1,604 human-annotated Select-All-That-Apply
   evaluation questions. Test ONLY: no benchmark test tuning or mixing
   with the approx. 7,983 raw questions, whose gold labels are not
   human verified. Compare author's First Token, Debias, Yes/No and
   Choice Funnel code and measure how many Transformer passes each
   approach uses. https://github.com/sata-bench/sata-bench.

4. ARCHIVAL NEGATIVE CONTROL: original grouped CMB + MultiRC. The
   2,140-question old Test is already opened; report prior 61.73%
   Anchor vs 61.82% matched BCE Continue only as historical evidence.
   Do not optimize anything new on the opened Test.

NOT required before first arXiv: EURLEX57K 4,271 labels or SWE-rebench V2
~32k software tasks. A fixed 255-candidate train-derived vocabulary
would define a NEW task protocol, not standard EUR-LEX numbers, and
needs gold-label coverage and candidate-source auditing.

## Data and readout contracts

- Lock raw dataset commit/revision, license, train/dev/test counts, SHA256
  and ordered label IDs and descriptors. Inspect empty/gold-count/label
  frequency and document length distributions before model submission.
- For EUR-LEX use all K=100 concepts, in a fixed order first, then
  separately test five label-order randomizations with every model.
  Labels must be tokenizer-verified single-token output slot IDs.
  If 100 choices + text exceed context budget, truncate only document
  content, using a fixed head+tail rule. Never silently truncate choices.
- Freeze legal count set M(x) per public dataset protocol. Model cannot
  see per-question gold answer count. Empty-set handling is global.
- Keep identical Qwen3.5-4B backbone, answer slot, data, prompt and input
  token cap for every arm. First run 5k train / 1k dev pilot, then the
  full official 55k / 5k / 5k protocol. Pilot sampling from the original
  TRAIN only, do not open official Test.

## Mandatory baselines (all labels identical)

A0 Original Gold BCE: independent sigmoid, fixed 0.5 AND DEV-tuned
global threshold; keep initial original BCE checkpoint in the main table.

A1 BCE Continue: SAME warm checkpoint and identical extra updates as
Anchor. A0 and A1 are different scientific controls; A0 is not a fair
replacement for A1.

A2 BCE + Learned Count Head: supervised gold count CE with candidate BCE;
use learned predicted count and conditional Top-m / set MAP. Match extra
head parameters and data/optimizer steps. This is a strong alternative.

A3 Independent Count Set CE: learn c_m independently of s; same MAP.

A4 Our Anchor: rho(m|x) proportional exp(log Z_m(s)+a_m(x)), joint Set CE.

A5 Anchor Detach: same forward distribution as A4, but detach s->logZ_m
gradient through the count-head path. Tests training feedback.

A6 BCE + frozen-score, two-stage count calibration: fit a count head
while leaving BCE scores fixed; compare same decoder to A4.

A7 BCE + DEV-calibrated label/global threshold and temperature scaling.
This is especially important if Count NLL/Brier rather than set EM wins.

PRIMARY FROM-BASE comparison: initialize all methods from the SAME
Qwen3.5-4B base, train on the SAME split with matched optimizer steps,
seeds (43/44/45), batch/token budget, prompt and dev LR-selection budget.

SECONDARY WARM comparison: first train common BCE on train only; freeze
checkpoint; continue A1/A2/A3/A4/A5/A6 with identical EXTRA TRAIN
examples and optimizer updates. Report A0 separately. One common
init alone does not justify a performance superiority claim if one
model receives extra updates. Tune LR/scale fairly for different losses.

Count Anchor and Independent Count Head have the SAME model family
under c_m = a_m + log Z_m(s), so any gain over A3 concerns
parameterization, optimization and inductive bias, not new expressivity.

## Reported metrics

On REAL datasets (one observed label set per context): exact set EM;
micro/macro F1, Jaccard; count NLL, count Brier, count ECE, count MAE,
count mode accuracy, over-/under-selection; set NLL on the SAME legal
support, option Brier/ECE; inference latency (model, DP, complete),
token length, number of forward passes and peak memory. BCE's joint
distribution must also be conditioned on public legal support for a
fair proper-set likelihood comparison. Report unrestricted BCE separately.

By-gold-count 0/1/2/3/4+, correct-vs-incorrect ranking, and
over-/under-selection rates. BCE per-item gold-count NLL quartiles
are RETROSPECTIVE DEV DIAGNOSTICS ONLY: labels cannot be used in a
deployed model router or to choose a method at Test inference.
Oracle gold-count Top-k is a NON-DEPLOYABLE rank upper bound.

On CONTROLLED data only (known ground truth distributions):
true count KL, joint KL, and count-strength sweep. A single
real gold set cannot reveal the conditional true distribution
rho*(m|x), so do not label real count NLL as 'oracle Count KL'.

Use paired 3-seed results for the pilot, 5 if clear improvement;
by-document bootstrap (by passage for MultiRC), report seed
variance separately. Freeze checkpoints/thresholds on official
validation, then evaluate each unseen official Test once.
An aggregate BCE/Anchor win from incomparable subsets is invalid.

## Correct the synthetic K-scale controls BEFORE reporting their results

Current C-regime K=16/32/64 script controlled_c_large.py:
1. Prints constant "collective pressure zero" even when hidden
   cardinality strength is nonzero. Store and expose the correct
   public factor, but never the sampled gold set.
2. Samples Gumbel Top-k, which induces a different fixed-size
   subset law than exp(sum chosen s_i)/Z_m(s). Replace by a
   suffix elementary-symmetric polynomial conditional sampler.
   Verify K=4,6 empirical frequencies against exact enumeration.
3. c16_count_kl.py silently skips missing context predictions and
   uses a rounded-probability Poisson-binomial DP. Require exact
   one-to-one contexts, seeds, and arm identities and stable
   log-domain scoring. Never compare differing cohorts.
4. K comparisons previously changed batch size and training steps;
   report this confound. Re-run K=8/16/32/64 on identical context
   counts/update budgets where memory permits. K=128/255 remains
   optional algorithm/runtime validation rather than a paper-ready
   claim of 4B performance.

## Execution gates

G0: benchmark download and DATA AUDIT (test unopened), numerical
oracle-sampler tests, original E0 DEV residual audit. Abort on
label-map mismatch, positive truncation, missing prediction IDs,
wrong count protocol or non-finite gradients.

G1: on EUR-LEX, 5k TRAIN / 1k DEV smoke with 4B LoRA and A0/A1/A2/A4/A5;
check count head outputs, training progress and exact K=100 DP.

G2: full 55k EUR-LEX, official 5k DEV, frozen official 5k TEST;
same seeds/budget, all mandatory arms with realistic LR tuning.
ECtHR-A follows as independent domain with A0/A1/A2/A4/A5.

G3: SATA-BENCH external 1,604 items once after non-SATA
checkpoint selection and frozen hyperparameters; include Choice
Funnel pass counts and same-backbone baseline when possible.

G4: corrected C-regime K-sweep; 2B/4B and maybe 9B if real result
survives A2/A6. Learned Pairwise is NOT part of G0-G4.

Success = real-data Count NLL and proper set NLL consistently
improve over strong BCE + Learned Count controls, with EM/F1
non-inferiority or improvement and cross-domain confirmation.
If only equivalent to a learned count head, claim cardinality
calibration/initialization, NOT a unique new probability family.
Do not claim an arXiv superiority result from synthetic oracle KL alone.

STATUS: This file is an experimental DESIGN, not downloaded
benchmarks, QZ submissions or completed GPU results.
