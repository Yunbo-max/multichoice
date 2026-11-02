# Paper 1 — Real multi-answer benchmark hunt after EUR-LEX
Date 2026-11-02. Source-verification and data-screening implementation ONLY.
NO new model inference, actual RoMQA archive processing, or favorable count gap claimed.

## 1. Why change datasets?

Latest DEV results:
- BCE: Count NLL 2.132, proper Gold-Set NLL 12.118.
- Frozen Anchor (zero init): Count NLL 2.10, Set NLL 12.09.
- Frozen Independent Count, train-count-prior init: Count NLL 1.98,
  Set NLL 11.97, outperforming Anchor on both.
See paper1_eurlex_p0_results.md and paper1_eurlex_p1_frozen_results.md.

On EUR-LEX, merely adding a count head does NOT establish an Anchor-specific
advantage. This is a useful real negative control. The earlier controlled
K=16/32/64 experiments have separate data-generation audit issues and should
not be the main positive evidence.

Necessary dataset screening test on HELD-OUT DEV:
    gap = CountNLL(Poisson-binomial(BCE scores))
          - CountNLL(frozen TRAIN-prior initialized independent count head).

A positive gap is a proxy for useful learnable count information, NOT proof
of an Anchor win or mathematically orthogonal information. Require paired
bootstrap confidence interval, candidate coverage and option-ranking
upper bound. The strong count head must receive the same initial
checkpoint features, training data, budget and train-only prior. Select the
benchmark only using TRAIN/DEV; never screen on sealed TEST.

## 2. Actual verified choices

| Choice | Size and real label | Why it might matter | Unresolved drawback |
|---|---|---|---|
| FIRST conceptual pilot: RoMQA CLOSED | ~11k train/~7k DEV/~11k blind TEST; paper's closed mode has 100 candidate entities; gold positive subset | Natural-language constraints are changed across related questions, so answer set and count may change; cluster-worst-F1 is an existing official metric | Third-party ZIP reachability unverified here, Test via CodaLab only, 100 candidates' DPR evidence may not fit 8k model context |
| FIRST coding backup: SWE-rebench V2 | 32,079 official fixing TASKS, 20 languages/3,617 repos; patch lists several changed files in some questions | Real code issue + modified file subset; variable cardinality; Battery project already contains three input views and baselines | Number of accepted MULTI-FILE 16–32k-context questions is unknown until full build; reference patch is not uniquely necessary set; candidate retrieval is a bottleneck |
| Maths alternative: LeanDojo | 98,734 theorem/proof traces (Lean 3), fine-grained used-premise dependencies | Formal proof uses variable premise sets and Lean verifier | One observed proof is not all possible valid proofs, whole theorem vs tactic-level premise count can differ |
| Alternate: QAMPARI | 61,911 train/1k DEV/1k TEST; average ~13 answers | Many correct entities per question, variable answers | No fixed official 100-candidate list; gold incomplete and retrieval bottleneck |
| Alternate: FEVEROUS | 87,026 verification claims, up to three evidence sets per claim | Multiple true evidence spans/cells | Multiple acceptable evidence SETS, often alternative rather than union; requires Wikipedia/table retrieval |
| Deprioritize: HoVer/HotpotQA | HoVer 18,171 train/4k DEV, 2–4 hops; HotpotQA 2 supporting pages | Evidence retrieval tasks | Logical hop count is not gold evidence sentence count; publicly known Top-2 is a strong no-head baseline |

Source-of-truth references:
- RoMQA official: https://github.com/facebookresearch/romqa
  paper: https://aclanthology.org/2023.findings-emnlp.470/
- SWE-rebench V2: https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2
  paper: https://proceedings.mlr.press/v306/badertdinov26a.html
- Existing Coding implementation:
  https://github.com/yuhanlydia/Battery/tree/main/source_localization
  readiness: https://github.com/yuhanlydia/Battery/blob/main/docs/SOURCE_LOCALIZATION_VERIFICATION_2026-11-01.md
- LeanDojo: https://leandojo.org/leandojo.html
- QAMPARI: https://github.com/samsam3232/qampari
- FEVEROUS: https://fever.ai/dataset/feverous.html

## 3. RoMQA data schema and actual availability caveat

Official RoMQA GitHub is archived (read-only since October 2023).
Its author says that reproductions of Wikidata cannot be hosted by Meta,
but provides reconstruction scripts and a third-party Wasabi ZIP at:
https://s3.us-west-1.wasabisys.com/vzhong-public/RoMQA/romqa_data.zip

The archive was not accessible from the research session, so source code
and paper ARE verified but ZIP contents/current download availability
ARE NOT. Do not claim a real train/dev count histogram before running
the data audit. The official closed format reads:
  data/closed/top_20.train.json.bz2
  data/closed/top_20.dev.json.bz2
  data/closed/top_20.test.noanswer.json.bz2
The 'top_20' filename refers to retrieved evidence, not K=20.
The official binary_dpr_nl baseline scores each entity separately and
uses cached DPR snippets (similarity >=0.65, up to ten, fallback top1).
RoMQA's reported metrics include set F1, exact set accuracy and the
WORST question score within each related-question cluster.
Test answers are hidden and CodaLab submission is required.
License is primarily CC BY-NC; consult repository LICENSE.md.

Read only the real TRAIN and DEV archive entries, audit full-answer
coverage, positive-count variation per contrast cluster, 100-candidate
assumption, and evidence availability:

    python -m paper_static.romqa_data_screen \
      --archive /path/to/romqa_data.zip \
      --output runs/romqa_screen_v1.json

Or if unzipped:
    python -m paper_static.romqa_data_screen \
      --data-dir /path/to/romqa/data/closed \
      --output runs/romqa_screen_v1.json

This audit uses stdlib only and refuses any TEST split or overwrites.
Regression tests cover mock data, not the actual archive.

Pilot: use the official candidate set without adding any gold negatives
or positives; group by cluster ID; ensure at least a usable 2k TRAIN
and 1k independently heldout DEV; use the SAME no-evidence or
DPR-evidence protocol, tokenizer budget, and snippets for all models.
Measure count-gap vs BCE and prior-independent head first.
If the gap is negligible or candidate coverage bad, RoMQA does not
justify further Anchor training.

RoMQA may fit the science of Anchor but is NOT guaranteed to beat BCE:
BCE can encode constraint-dependent cardinality through individual
candidate probabilities. Do not interpret count variance itself as a
positive residual count gap.

## 4. SWE-rebench Coding backup is not just a proposal

Battery main branch already implements source_localization:
- Input: issue + file paths, skeleton, or complete original source for
  up to 32 candidate files, retrieval pool 128.
- Labels: changed preexisting implementation files parsed from the
  gold patch; NO gold-injected candidates or gold code in prompts.
- Split: repository-level deterministic 60/10/30, not a claim of
  pretraining novelty; downloaded model-source license kept.
- GPU and full-corpus training are NOT completed. One real source
  sample was accepted with 16 candidate files and 32,409 prompt tokens;
  one was excluded under addition/deletion scope. Real 32k dataset
  acceptance counts remain unknown. The local CPU/test integration passed.
- Existing baselines: slot-BCE, AR-SFT, pointwise BCE, BM25; do NOT
  describe them as reproductions of SWERank or SWE-Fixer.

Before adding Anchor, execute Battery's official export/build and audit
its actual manifest counts and retrieval coverage:
    python -m source_localization export \
      --output runs/swerebench-v2-raw
    python -m source_localization build \
      --raw runs/swerebench-v2-raw/raw.jsonl \
      --cache runs/localization-repositories \
      --output runs/localization-32k \
      --tokenizer /path/to/Qwen3.5-4B \
      --fetch --candidates 32 --pool 128 \
      --min-tokens 16384 --max-tokens 32768 --reserve 256

Only after seeing actual multi-file accepted/coverage counts should
we call this a large multi-file benchmark. For joint-set NLL,
candidate-excluded gold files must be handled explicitly rather than
pretending all gold is in the candidate set. Include candidate-complete
conditional analysis AND all-case retrieval failure penalties.
Run matched frozen BCE vs train-prior independent Count Head DEV gap.

## 5. LeanDojo Maths alternative

LeanDojo gives proof-state/premise annotations and published splits
for retrieval. But one recorded proof's premises are a proxy for one
solution path; mathematically valid alternative proofs can use a
different set. Do not claim these are the uniquely correct choices.
First audit the distribution of distinct premise counts per THEOREM
(not per-tactic unless that is the declared task), and top-100
candidate retrieval recall. Then run the same count-gap test.
Maths is a scientifically useful alternative, not the fastest first
Paper-1 positive experiment.

## 6. Avoid an easy but invalid 'Anchor wins' task

If the task explicitly fixes the exact number (e.g. 'select exactly
two'), independent BCE + known-count Top-2 solves the count decision
without any learned Anchor; this is a mandatory baseline.
Do not manufacture a hidden quota or reveal a gold answer count.
A weak BCE threshold is not sufficient as the only control.

Recommendation at time of writing: TRY RoMQA CLOSED data audit first
because original related-question constraint clusters let us measure
count shifts and worst-cluster robustness on published labels. If the
upstream archive is inaccessible or DEV count-gap is not positive,
switch to the existing SWE-rebench V2 Coding implementation, subject
to its manifest yield and count-gap gate. Only ONE main benchmark
should proceed to full Qwen4B training.
