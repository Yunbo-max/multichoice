# Paper 1 — When does Cardinality/Anchor modeling help? Dataset selection criteria

**Date:** 2026-11-02. Branch: research/static-setjev-paper-20261027.

This note fixes the precise boundary condition that separates the synthetic
C-regime (where Count Anchor beats BCE) from EUR-LEX (where it does not), and
turns it into a concrete, checkable filter for sourcing a real benchmark where
the method has a genuine chance of winning.

## 1. What our method actually adds over BCE

The set law we model is

    P(Y|x) = rho_|Y|(x) · exp(sum_i s_i(x) Y_i) / Z_|Y|(s(x)),

where `s_i` are the candidate/option scores and `rho_m` is the count law.
A plain BCE model is the special case `rho_m = Z_m(s) / sum_k Z_k(s)` (the
Poisson-binomial count induced by the independent marginals). Our Count
Anchor parameterizes `rho_m = softmax_m(a_m(x) + log Z_m(s))`.

Therefore **the ONLY thing Count Anchor adds over BCE is the learned residual
`a_m(x)`**, and `a_m(x)` can only capture cardinality signal that is NOT already
encoded in `log Z_m(s)` — i.e. cardinality signal that is *orthogonal* to the
candidate scores.

## 2. Why synthetic C-regime won and EUR-LEX did not

Synthetic C-regime (`paper_static/controlled_regimes.py`) generates two
*independent* information sources:

    scores   = 0.8 · utility        # decides WHICH options (within-count ranking)
    residual = -strength·(m-center)^2 # decides HOW MANY (workload cue), strength up to 1.8

Because `a_m` is chosen independently of `s_i`, the count law cannot be read off
the candidate scores; the learned residual has real signal to fit, so Anchor
wins (and the `strength=0` control collapses it back to I-regime, proving the
effect is causal).

EUR-LEX is the opposite: the count (how many EuroVoc labels a document gets)
and the scores (which labels are relevant) are both driven by the same document
content, so the count is already read off the marginal probabilities. The
measured gap is tiny: BCE Poisson-binomial Count NLL 2.13 vs an explicit
gold-count head 2.05 vs frozen Anchor 2.10. There is almost no orthogonal
count signal left for `a_m` to learn.

## 3. The formal, measurable criterion

Let `s_i(x)` be a well-trained BCE option score and `m=|Y|` the true set size.
Define the **count gap**:

    count_gap = CountNLL(Poisson-binomial(s)) - CountNLL(explicit gold-count head)

- `count_gap ≈ 0`  -> cardinality is already captured by the marginals -> Anchor has nothing to add (EUR-LEX case).
- `count_gap >> 0`  -> there is marginals-orthogonal cardinality signal -> Anchor has real headroom (C-regime case).

This is cheap to estimate on any candidate benchmark (train a plain BCE,
then fit a count head on frozen features, compare Count NLL on DEV). It is the
single most decisive pre-filter.

## 4. Checklist for sourcing a favorable benchmark

A dataset is a *candidate* for a positive Anchor result only if it satisfies
the necessary conditions, and is *promising* only if it also satisfies the
strong signals.

### Necessary (all must hold)
1. **Set-valued task**: the answer is a subset of >1 items (multi-select,
   multi-label, multi-evidence, multi-tool, multi-answer).
2. **Cardinality varies**: |Y| has non-trivial variance across examples
   (a dataset where every answer is always exactly 2 has no count signal).
3. **Cardinality has a signal source that is separable from content**:
   there is a publicly observable or inferable quantity that drives "how many"
   independently of "which ones".
4. **Scores alone cannot recover the count**: after a well-trained BCE, the
   count gap (Section 3) is clearly positive.

### Strong signals (makes a win likely)
5. **Explicit cardinality constraint / quota / budget**: the task states
   "select exactly k", "select up to N under a budget", or an external rule
   fixes the count (e.g. exactly 2 supporting passages, exactly the articles
   violated, a fixed committee size).
6. **"How many" vs "which" are independent decisions**: the count is set by
   task structure or an external cue, while option selection is set by content.
7. **Repeated draws / multiple valid sets** (optional but ideal): lets us
   measure the true joint law instead of a single hard label set.

### Exclude (Anchor will not win — the EUR-LEX pattern)
- Count is a direct function of content relevance (topic count -> label count).
- Count has low variance or is effectively constant.
- The only "structure" is already captured by per-option marginals.

## 5. Concrete directions to search (give these to a data-hunting agent)

Prioritized by likelihood of a large, marginals-orthogonal count signal:

1. **Fixed-count retrieval / evidence selection** — tasks requiring *exactly*
   k evidence passages / supporting facts (e.g. HotpotQA-style top-2 evidence,
   MultiRC multi-answer with a stated number). The count is imposed by the
   protocol, not by content.
2. **Budget / capacity-constrained tool or resource selection** — choose a
   subset of tools, actions, or resources under an explicit cost/budget/quota;
   "how many" is an economic constraint orthogonal to "which are relevant".
3. **Combinatorial / matching selection** — pick a matching, a committee of
   fixed size, or items that must pair up (compatibility codes, pair edges);
   the cardinality is structural.
4. **Medical / legal coding with a count prior** — ICD/CPT/statute coding where
   the number of codes is driven by a separable factor (severity, length of
   encounter, procedural complexity) rather than by content overlap alone.
5. **Exam multi-answer questions** — questions that explicitly ask for "all
   and only the N correct options" with N stated or inferable from the stem.

For each candidate, run Section 3's count-gap probe before committing GPU time:
train plain BCE on the train split, fit a frozen-feature count head, and report
the DEV count gap. Only benchmarks with a clearly positive gap and one of the
strong signals in Section 4 are worth an Anchor continuation.

## 6. Decision rule

- If the count gap is ~0 across several sourced benchmarks, the honest paper
  conclusion is the *boundary* result: cardinality modeling helps only when a
  marginals-orthogonal count signal exists, and it is absent in content-driven
  multi-label benchmarks like EUR-LEX.
- If at least one real benchmark shows a large count gap AND a strong signal
  source, that benchmark is the one to run the full matched Anchor vs BCE vs
  learned-count continuation (P1a/P2) on.
