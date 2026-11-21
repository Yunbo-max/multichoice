# Paper 1 — Motivation and Introduction (ICML 2027 working draft)

**Updated:** 2026-11-21  
**Canonical research branch:** `research/static-setjev-paper-20261027`  
**Provisional title:** *When Do Multi-Answer Decision Models Need Cardinality Potentials?*  
**Status:** manuscript draft / motivating examples are proposed use cases, **not** evaluated downstream deployments.

> **Claim discipline.** This paper studies *when* cardinality potentials help, not a universal superiority claim. Independent BCE already supports multi-label outputs. An unrestricted independent learned count head and our residual cardinality potential can parameterize the same family of set distributions after reparameterization. Thus the potential contribution concerns the inductive bias, optimization, calibration, and computationally tractable structured prediction—not strictly greater expressivity than a well-designed count head.

## 1. One-sentence thesis

Fast, single-pass language-model decision readouts should output **sets of varying size**, and the crucial modeling question is whether a global answer-count constraint supplies predictive information beyond independent option scores, under properly matched baselines and costs.

## 2. Motivation: what is the actual problem?

### 2.1 Why fast decisions rather than more generated reasoning?

Not every language-understanding task requires a long autoregressive reasoning trace or free-form answer. In many repeated decisions, a context and a finite candidate inventory are already provided, and the operational output is *which candidates qualify*. A one-pass decision readout can be attractive when many decisions must be made under a tight inference budget. This is a motivation about interface and potential efficiency, **not a measured wall-clock speedup for this method over all generative baselines**.

Examples illustrating the interface (future application candidates, not current validation):
- **Code triage:** select candidate files that plausibly need inspection or modification; subsequent debugging and patch generation still require an agent.
- **Evidence selection:** select all passages or sources that support an answer; assess evidence quality separately.
- **Tool / workflow routing:** select relevant tools or workflow steps from a known registry; the workflow executor still determines dependencies and order.
- **Multi-intent / structured NLU:** identify multiple simultaneously applicable intents or labels.

These examples matter only when the candidate inventory is given and the selection itself is a useful subtask. We **do not** claim that selecting files is a substitute for coding, that selecting tools solves workflow planning, or that semantic judgments are inherently trivial.

### 2.2 Single-choice interfaces are not set-valued decisions

A categorical decision model returns one label from `K` candidates using a softmax. But many tasks require a subset `S ⊆ {1,...,K}`, with `|S|` unknown in advance. A fixed single-answer simplex cannot directly encode multiple simultaneous positives. The straightforward generalization is `K` sigmoid scores trained by binary cross-entropy (BCE).

**Important correction:** BCE *can* select multiple options. Its limitation is not that it returns one answer; it assumes conditional independence in the induced joint set distribution.

### 2.3 The real gap: independent membership versus global cardinality

Let `q_i(x)=sigmoid(s_i(x))` be each candidate's selection probability. Independent BCE induces

```math
P_{BCE}(S\mid x) = \prod_{i\in S}q_i(x)\prod_{j\notin S}(1-q_j(x)),
\quad
P_{BCE}(|S|=m\mid x) = \sum_{|S|=m} P_{BCE}(S\mid x).
```

The resulting count distribution is **determined by the membership scores** (a Poisson-binomial law). There is no separate residual degree of freedom to encode a count constraint once those scores are fixed. This can matter when question semantics jointly constrain the number of valid answers, e.g., evidence must cover several requirements or a query requests *exactly two* valid entities.

A four-option illustration clarifies the distinction. Two tasks may have identical marginal inclusion probabilities `P(i∈S)=0.5` for each candidate, yet one draws exactly two answers each time and the other draws each answer independently. Independent Bernoulli predictions cannot represent both full joint distributions while preserving those marginals. **This is a distributional thought experiment, not an empirical claim about a benchmark.** Moreover, a count mechanism alone cannot encode arbitrary pairwise co-occurrence patterns.

### 2.4 Why predicting the count is not by itself the novelty

A strong comparison is to learn a separate `P(|S|=m\mid x)` and condition candidate selection on the count. Such a count head can already capture global cardinality. Therefore the testable research question is:

> Can a *residual cardinality parameterization*, tied to the same candidate scores and trained with exact set likelihood, offer a useful inductive bias or better optimization/calibration compared with both independent BCE and a **properly initialized** learned-count baseline, and under what data regimes?

This formulation embraces negative controls: if count is already accurately explained by independent scores, Anchor should provide little benefit and may be worse.

## 3. The proposed model, without overclaiming

Given `K` option scores `s_i(x)`, define the set distribution

```math
P_\theta(S\mid x)=
\frac{\exp\left(\sum_{i\in S}s_i(x)+a_{|S|}(x)\right)}
{\sum_{m=0}^{K}\exp(a_m(x))Z_m(s(x))},
\qquad
Z_m(s)=\sum_{|T|=m}\exp\left(\sum_{i\in T}s_i\right).
```

Here `a_m(x)` is a learned **residual cardinality potential**. With `a_m(x)=0`, this distribution reduces to independent sigmoid/BCE for the same option logits, because the partition function is `∏_i(1+exp(s_i))`. The exact count law is

```math
\rho_m(x) = P_\theta(|S|=m\mid x)
=\frac{\exp(a_m(x))Z_m(s(x))}
{\sum_r\exp(a_r(x))Z_r(s(x))}.
```

Elementary-symmetric-polynomial dynamic programming evaluates `Z_m` and supports exact normalization and set MAP without enumerating `2^K` subsets; the standard all-count DP costs `O(K^2)` time and `O(K)` working memory. Inference is single-pass through the backbone followed by structured decoding, not free-form autoregressive answer generation.

**Expressivity equivalence:** An independent-count model with count logits `c_m(x)` and conditional selection proportional to `exp(∑_{i∈S}s_i)/Z_m(s)` induces exactly the same family when `c_m(x)=a_m(x)+log Z_m(s(x))`. This equivalence is central to the ablation design. Any claimed advantage must be attributed to optimization, finite-head capacity, initialization, or calibration rather than a new universal probability family.

## 4. Introduction — manuscript-ready working draft

Decision-making with language models does not always require generating a long sequence of reasoning tokens. In many practical settings, the context and candidate actions are already specified, and the task is to make a compact structured decision. Examples include selecting relevant evidence from a supplied passage collection, identifying which intents apply to a user request, choosing tools from a known registry, or triaging candidate files before a coding agent performs edits. When such decisions must be repeated at scale, a single-pass decision interface is a useful alternative to autoregressively verbalizing every selection. The relevant output, however, is often not a single label: it is a **set whose size varies with the input**.

Recent language-model decision readouts motivate extracting answers directly from internal representations rather than requiring a generated final answer. Their categorical form naturally models the choice of one candidate. For set-valued tasks, the standard adaptation is to predict a binary label for every candidate and optimize binary cross-entropy (BCE). This gives a simple and competitive multi-answer decision model, but it also imposes a specific probabilistic structure: conditioned on the input, candidate membership decisions are independent. Consequently, the full distribution over the number of selected candidates is entirely induced by the individual inclusion probabilities.

This restriction is not always innocuous. The semantics of a request may specify not only *which* candidates are plausible but also *how many* candidates should jointly be selected. A question may require a fixed number of entities, or evidence selection may be governed by a shared coverage requirement. Even when individual membership probabilities are reasonable, the probability assigned to complete answer sets—and to the answer count—can be miscalibrated. The issue is therefore not whether BCE can output multiple positives, which it can, but when independent membership scores are sufficient to model a structured subset decision.

We study this question through **cardinality-anchored set decisions**. Starting from native candidate scores produced by a single-pass language-model readout, we introduce an input-dependent residual potential for the size of the predicted answer set. This yields a normalized distribution over variable-size subsets, for which count probabilities, complete-set likelihoods, and maximum-a-posteriori set predictions can be computed exactly by dynamic programming rather than by enumerating every candidate subset. The independent BCE model is recovered when the residual cardinality potential is zero, providing a controlled connection between the two objectives.

Importantly, cardinality modeling is not unique to this parameterization. A separate learned count distribution combined with within-count candidate scores can represent the same family of set distributions after reparameterization. Thus we do **not** posit an expressivity advantage over a sufficiently flexible independent-count head. Our empirical question is more specific: under which forms of count dependence, model capacity, initialization, and supervision does anchoring the count law to candidate scores improve calibrated structured decisions? We compare against both independent BCE and matched learned-count alternatives, including strong training-count-prior initialization, while examining regimes where additional count modeling is unnecessary or harmful.

Our controlled results provide initial evidence of this distinction. On a synthetic cardinality-dependent setting with eight candidates, the anchored model improves count and joint-distribution KL relative to BCE, whereas independent-data controls favor BCE. Real-world results are less conclusive: grouped CMB-Exam/MultiRC exact-set performance is essentially tied between BCE and Anchor, and an appropriately initialized independent-count head outperforms Anchor on the current EUR-LEX pilot's count NLL. These mixed findings motivate a deliberately conditional assessment rather than a claim of universal superiority. Our goal is to identify **when global cardinality information contributes beyond marginal option evidence**, and to establish evaluation and baseline protocols for reliable, fast, set-valued language-model decisions.

### Proposed contributions (draft; evidence-qualified)

1. **Problem formulation:** characterize the mismatch between independent option membership and global answer-count / complete-set probabilities in one-pass set-valued decisions.
2. **Method and equivalence:** give a cardinality-residual formulation with exact dynamic programming, an explicit BCE special case, and an equivalence to properly parameterized learned-count models.
3. **Controlled evaluation:** separate independent, cardinality-dependent, and relational dependence, testing count and joint-set calibration with matched controls rather than relying only on binary F1.
4. **Empirical boundary:** report both favorable synthetic settings and negative/near-tie real-data results; reserve claims about real-world advantages until stronger audited evidence exists.

## 5. Evidence ledger for the Introduction (as of 2026-11-21)

| Setting | Valid reported observation | Interpretation |
| --- | --- | --- |
| Controlled-C, K=8, Qwen3.5-4B, 3 seeds | Count KL: BCE 0.333, Anchor 0.278; Joint KL: BCE 0.419, Anchor 0.366 | Narrow support for cardinality-dependent synthetic regime |
| Controlled-I | BCE outperforms Anchor on joint KL | Necessary negative control for independent target structure |
| Controlled-R | Pairwise selector near chance | No supported relational-interaction superiority |
| CMB-Exam / MultiRC grouped frozen TEST | BCE exact-set EM 0.6182, Anchor 0.6173 | Near tie / slight negative; do not claim real-data gain |
| EUR-LEX frozen features, 5k TRAIN / 1k DEV | Count NLL: BCE 2.132, Anchor 2.10, TRAIN-prior Independent Count 1.98 | Properly initialized learned count is currently strongest |
| K=16/32/64 early synthetic generator | Invalid pressure cue and conditional sampler | Exclude from headline results pending repair |
| RoMQA Closed | Converter and CPU/mock checks prepared; official archive unavailable | **No real RoMQA results** |
| Code / evidence / tool selection | Conceptual application examples | **No direct evaluation yet** |

See `docs/BRANCH_CLEANUP_20261118.md` on `main` and `docs/paper1_romqa_icml2027_protocol_20261114.md` on this branch for the audit, precise protocols, and benchmark restrictions. Do not merge DEV numbers, frozen-feature comparisons, and held-out TEST into one ranked table without preserving their separate setups.

## 6. Experiments necessary to support a stronger ICML claim

- Compare **BCE / Anchor / Anchor Detach / Independent Count (zero init) / Independent Count (TRAIN-count-prior init)** under matched backbone scores, training budgets, splits, and checkpoint rules.
- Prioritize proper **Count NLL, Joint-Set NLL, Count KL, Joint KL**, plus exact-set EM, set F1, count MAE, calibration, and measured decision throughput. Do not treat high option-level F1 alone as proof of successful set prediction.
- Audit the relationship between the input's cardinality cues and gains. Explicitly separate **I** (independent), **C** (count dependent), and **R** (pairwise/relational) synthetic mechanisms.
- Find an actually accessible real-world setting with varying, auditable global count constraints; **RoMQA is presently blocked** and cannot be used as evidence.
- Treat code triage, evidence retrieval, multi-intent detection, and tool/workflow routing as **potential follow-up evaluations**, with clearly defined candidate inventories and downstream utility metrics.
- Measure efficiency rather than assuming it: backbone forward latency, structured DP overhead, throughput versus autoregressive generation, and candidate-set-size scaling.

## 7. Writing guardrails

**Use:** “when global cardinality dependence matters”, “set-valued decision”, “calibrated answer-count distribution”, “proper scoring rules”, “single-pass readout plus exact DP”, “conditional / mixed empirical evidence”.

**Avoid:** “BCE cannot predict multiple answers”, “Anchor is strictly more expressive than independent count”, “we solve pairwise interactions”, “we already improve coding/tool routing”, “we outperform BCE on real data”, “validated RoMQA gains”, or “universal SOTA”.

**Paper positioning:** a question-driven study with a cardinality-residual inductive bias, rather than an overclaimed general-purpose replacement for reasoning agents.
