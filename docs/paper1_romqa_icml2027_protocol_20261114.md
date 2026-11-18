> **Data availability correction, 2026-11-14:** The official RoMQA README's third-party dataset link `https://s3.us-west-1.wasabisys.com/vzhong-public/RoMQA/romqa_data.zip` currently returns **HTTP 404 Not Found** when accessed through the README link. No verified RoMQA Train/DEV ZIP is available in this workspace, and the pipeline is therefore **BLOCKED**; no real benchmark run should start. Official reconstruction scripts exist but would require source Wikidata/T-REx and independent data provenance verification. A currently responsive alternative is the QAMPARI official ZIP at `https://aggreg-qa.s3.amazonaws.com/qampari.zip` (served as application/zip), but QAMPARI has **no fixed official 100-candidate closed-choice set**; changing the candidate construction would define an explicit derived task, not a drop-in RoMQA replication. SATA-BENCH human-annotated multi-select data is at `https://huggingface.co/datasets/sata-bench/sata-bench` (1,604 human-labeled evaluation items, not a comparable full training set). **Do not treat a working download as evidence that Anchor beats BCE.**

# ICML 2027 Paper 1 — RoMQA Closed: code-complete staging / results not run

**Status as of 2026-11-14: PREPARED CODE + CPU MOCK VALIDATION ONLY.**
No verified RoMQA archive has been materialized locally, no full-data
Train/DEV audit, no trained RoMQA BCE model, no RoMQA Anchor experiment,
no RoMQA positive-result claim, and no blind Test inference/submission.

**Single canonical Paper 1 branch (after 2026-11-18 merge):** research/static-setjev-paper-20261027.
RoMQA preparation code was consolidated through two-parent merge `236daacddb2d26fbf5f945b122d1390a32d4df22`; the older RoMQA branch pointer is redundant. Code integration does not imply RoMQA experimental validation or replace existing EUR-LEX pilot results.
No coding/Battery/SWE-rebench data is mixed into this paper experiment.

## 1. Scientific question

When a task asks for a subset of valid entities, does a residual
cardinality law improve **proper count and whole-set probabilities**
and robust set selection beyond strong independent BCE and a properly
initialized learned count distribution?

For K candidate logits s(x) the anchored family is
  P(S|x) ∝ exp(sum_{i in S} s_i(x) + a_{|S|}(x))
with legal counts 0..K unless the public benchmark disallows empty.

The count law equals:
  rho_m(x) = exp(a_m(x)) Z_m(s(x))
             / sum_r exp(a_r(x)) Z_r(s(x)),
  Z_m(s)=sum_{S:|S|=m} exp(sum_{i in S}s_i).

BCE is the zero-residual special case a_m=0. Independent learned count
head is:
  P(S|x) = softmax(c(x))_|S|
           exp(sum_{i in S}s_i) / Z_|S|(s).
**With unrestricted heads they represent the same probability family
under c_m=a_m+log Z_m.** Any claim is about optimization, finite
capacity, initialization and calibrated decisions; NOT novel
expressivity or universal pairwise dependency capture.

RoMQA tests **constraint-varying multi-answer selection**, not gold
pairwise interaction matrices. One observed Gold subset per question
does not identify an arbitrary multimodal joint distribution.

## 2. Data validity gate (BLOCKED on actual official archive)

Official benchmark: https://github.com/facebookresearch/romqa
Official Closed files:
  data/closed/top_20.train.json.bz2
  data/closed/top_20.dev.json.bz2
  data/closed/top_20.test.noanswer.json.bz2

Meta says it cannot host reproduced Wikidata; its README gives a
third-party Wasabi archive or scripts for reconstruction. That
external ZIP has **NOT been downloaded/verified in this workspace**.
Do not claim the benchmark is fully data-ready. Never score blind TEST
as though it has public labels; only a formally locked CodaLab
submission can yield a genuine official TEST result.

Existing paper_static.romqa_data_screen checks real Train/DEV
candidate K distribution, gold count, empty answers, question cluster
count shifts, duplicate candidates, missing answers and DPR evidence.

NEW paper_static.romqa_data:
- Reads exactly official Train+DEV, NEVER opens Test.
- Converts to native harness multi-select rows with 100 candidate
  entities (or explicitly reject K deviation); no gold chosen candidates.
- Uses official question, candidate names and deterministic **first
  DPR evidence** with >=0.65 similarity, fallback first available.
  Globally reduces chars per candidate to fit the audited real model
  chat-template input budget, never drops candidate entities, never
  uses oracle gold evidence. No-evidence ablation is separately
  declared, not mixed into the main evidence protocol.
- Preserves source IDs and cluster IDs; prevents split overlap.
- Writes full official DEV answers to an evaluation-only sidecar; they
  are NOT model-visible. Never adds full answers to candidate sets.
- Samples complete clusters, not isolated questions, for the optional
  pilot; pilot's sidecar is aligned with the exact selected DEV IDs.
- Saves SHA256s, prompt lengths, evidence coverage, candidate coverage
  and actual per-split counts. Refuses overwrite.

## 3. Runbook (NO commands were run in this code-preparation turn)

After acquiring official archive, run:
~~~bash
python -m paper_static.romqa_data_screen \
  --archive /path/to/romqa_data.zip \
  --output /runs/romqa_source_audit.json

python -m paper_static.romqa_data \
  --archive /path/to/romqa_data.zip \
  --tokenizer /models/Qwen3.5-4B \
  --output /datasets/romqa_closed_v1 \
  --expected-k 100 \
  --evidence-mode dpr-first \
  --evidence-chars 256 \
  --max-input-tokens 16384 \
  --pilot-train-clusters 120 \
  --pilot-dev-clusters 40
~~~
The above are **examples, not verified existing paths**. Some archive
rows may not satisfy K100 or prompt budgets, requiring explicit
protocol review on Train/DEV; do not silently filter hard rows.

Generate executable BUT NEVER AUTO-RUN shell steps:
~~~bash
python -m paper_static.romqa_plan \
  --manifest /datasets/romqa_closed_v1/manifest.json \
  --data /datasets/romqa_closed_v1 \
  --repo /absolute/path/to/multichoice \
  --base-model /models/Qwen3.5-4B \
  --modules /absolute/path/to/multichoice/innerjev/configs/4B/modules.json \
  --run-root /runs/romqa \
  --output /runs/romqa/staged_plan \
  --pilot --seed 43 44 45 --lr .001 .0003
~~~
Plan script produces separate **ordered** scripts. No scheduler,
background job, training or Test submission is launched. Verify the
GPU memory/window first and use one-step real-Qwen readiness in a
separate job directory. A full 100-entity + evidence prefill can be
very expensive and may not fit an unmeasured QZ job wall limit.

## 4. Must-run method matrix, same Train/DEV and model

(1) BCE native answer-slot model, from Qwen3.5 base.
(2) Matched BCE Continue, from common RoMQA BCE warm.
(3) Joint Set CE with Count Anchor, from same BCE warm.
(4) Anchor Detach, same forward law, count-logZ score gradient removed.
(5) Independent Count Set CE from zero init.
(6) Frozen BCE candidate scores + zero-init Anchor linear Count Head.
(7) Frozen BCE candidate scores + zero-init Independent Count Head.
(8) **Frozen BCE + TRAIN-count-prior init Independent Count Head**, the
    mandatory scientifically strong comparison discovered on EUR-LEX.

All end-to-end arms must receive same architecture, candidate evidence,
training instances, batch/token budget, LoRA rank/alpha, extra optimizer
steps and fair DEV LR selection. Frozen heads share the same cache and
are trained with identical Count CE and LR grid/seeds. Do NOT use one
arm's Gold size at inference. Do NOT compare a zero-init independent
Count Head alone against an Anchor and call it a win.

The frozen-score control is a scientific isolation test, not a full
alternative answer scorer. A frozen BCE candidate ranking cannot be
improved by a count-only head except through count/set probabilities.

## 5. Scores, official denominator and validity

Score on exact same Train/DEV identities; no score filtering by
selected candidate completeness. Main custom probabilistic scores:
- Proper Count NLL, complete candidate-set Joint-Set NLL.
- Exact Set EM, per-question Set F1, count MAE, predicted-set size bias.
- Paired cluster bootstrap 95% intervals, group by official cluster ID.

RoMQA official Closed metrics:
- Average F1 excludes empty-candidate-Gold rows;
- official evaluation excludes instances with empty complete_answer;
- Cluster-min F1/EM evaluate worst member of each related-question
  cluster (do not replace with ordinary macro-F1);
- complete-answer recall/accuracy can be limited by the original 100
  candidate pool. Do NOT call candidate set score full-KB recall.

paper_static.romqa_metrics reports both ALL-QUESTION proper scores
and official-denominator-equivalent set scores, plus uncovered true
answers. paper_static.romqa_compare refuses mismatched data,
question IDs, cluster IDs, Gold coverage and produces paired cluster
bootstrap effects; all work is strictly DEV-only.

Important: DEV used to choose LR and threshold is not an independent
generalization test. Freeze model, evidence mode, hyperparameters,
and threshold BEFORE any official blind TEST submission, and report
DEV selection optimism as appropriate.

## 6. Existing evidence (do NOT replace or cherry-pick)

A. CMB-Exam/MultiRC grouped:
   BCE Continue Test Exact EM 0.6182; Count Anchor 0.6173.
   Historical negative/degeneracy control; already-opened Test,
   never tune new loss on it.

B. Controlled-I/C/R Qwen3.5-4B:
   Cardinality-only C: Anchor Joint KL .366 vs BCE .419,
   Count KL .278 vs BCE .333, three seeds.
   Independent I: BCE better.
   R: learned pair selector near chance; Anchor not universal.

C. K16/32/64 cardinality sweep:
   An early claimed larger-K positive trend is NOT yet trustworthy
   for ICML: generator displays constant pressure text, and Gumbel
   Top-k sampler does not implement exact conditional-subset law.
   Repair sampler and public cues, then rerun matched budgets before
   using the trend as a headline or figure.

D. EUR-LEX (5k Train / 1k DEV):
   Frozen Count NLL: BCE 2.132, Anchor 2.10,
   Independent Count TRAIN-prior **1.98**.
   This is a strong negative for Anchor-specific superiority.
   Do not hide it. Explain optimization/inductive bias, not a
   fundamentally new distribution family.

## 7. ICML 2027 claim gates (falsifiable)

1. Count gap BCE vs TRAIN-prior learned Count on RoMQA DEV:
   if near zero, there is little useful cardinality signal beyond BCE.
2. Anchor must improve proper Count or Joint Set NLL over BOTH
   BCE and prior Count (paired cluster uncertainty), and not destroy
   all-case Exact Set/F1; otherwise narrow the claim to a negative
   calibration result or initialization robustness.
3. Cluster-min robustness is a **secondary** hypothesis; it must be
   measured, not presumed from the word "robust" in the dataset title.
4. Pairwise claims require separate verifiable co-occurrence evidence.
   Do not use RoMQA as direct proof of a learned J_ij.
5. No hidden post-hoc benchmark selection or Test result peeking.

## 8. Paper writing can start now

Provisional title:
"When Do Multi-Answer Decision Models Need Cardinality Potentials?"

Abstract outline (no invented RoMQA result):
- Problem: independent multi-option predictions induce restrictive count law.
- Method: native logits + residual count potentials + exact O(K^2)
  partition and MAP, with a clear mathematical connection to BCE.
- Controlled result: benefits on count-dependent generator, not on
  independent generator; learned pair interactions remain open.
- Real evidence: CMB/MultiRC and EUR-LEX negative controls;
  RoMQA validation TODO.
- Limitations and science: learned prior alternative is a real competitor.

Proposed 8-page main section split:
1 Introduction and problem / claims
2 Independent subset prediction and count law
3 Count-anchored set model / exact DP / gradient mechanism
4 Synthetic I/C/R representational and learning controls
5 Real data: CMB/MultiRC, EUR-LEX, prospective RoMQA
6 Calibration, robustness, ablations, qualitative question-cluster figures
7 Related work / limitations / conclusion

Figures/tables:
- Fig 1: Same option marginals but different count distributions (the
  4-option example is a PROOF/ILLUSTRATION, not RoMQA empirical evidence).
- Fig 2: Controlled C-vs-I Count/Joint KL; **use K8 only until larger-K
  generator and sampler are repaired**.
- Fig 3: RoMQA real constraint-cluster response: BCE/Anchor/prior
  predicted count distributions per example; produce only after data.
- Table 1: Same-data BCE/Anchor/independent-prior, Count NLL, Joint NLL,
  Set EM/F1, cluster-min, matched seeds.
- Table 2: Frozen feature heads / detach / evidence/no-evidence and
  candidate-count effects, including EUR-LEX failure.

The official ICML 2027 Call for Papers currently specifies **8 main
pages** (excluding references/appendix), **2027-01-14 AoE abstract**
and **2027-01-22 AoE paper** deadlines. It also requires disclosure of
substantive generative AI assistance in research and manuscript work:
https://icml.cc/Conferences/2027/CallForPapers

No paper should claim RoMQA gains, released dataset, completed GPU
ablation, or an official TEST result until the underlying artifact
exists and is independently audited.
