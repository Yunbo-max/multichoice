> **Implementation update (2026-11-01):** The single-benchmark EUR-LEX data converter, hash-audited QZ pilot/evaluation payload generators, learned-count loss scaling, official F1/proper-NLL scorer, and DEV threshold tuning have been implemented and pass offline GitHub Actions checks. See [the runnable end-to-end manual](paper1_eurlex_execution.md). Real dataset download, actual 4B pilot, final Test and PyTorch gradient checks remain UNRUN; no new performance claim is made.

# Paper 1 — SINGLE main benchmark locked: LexGLUE EUR-LEX (2026-11-01)

**Decision:** Only LexGLUE EUR-LEX is the NEW main benchmark to train and test. No simultaneous ECtHR/SATA/RoMQA/other datasets. Previous CMB + MultiRC checkpoints and the already opened 2,140-row Test may be mentioned only as historical negative-control results, **not** used to tune this project.

**State:** experiment scope and exact protocol, not a completed EUR-LEX conversion, QZ job, or GPU result.

## Official benchmark

- Source: `coastalcph/lex_glue`, configuration `eurlex`, pin exact dataset revision and SHA256 before training.
- Official splits: **55,000 train, 5,000 validation, 5,000 held-out test**.
- Complete fixed **100 EuroVoc class label IDs**, with official English label descriptions from `nlpaueb/multi-eurlex/data/eurovoc_descriptors.json`. Check that the 100 `ClassLabel.names` IDs map 1-to-1 to nonempty correct descriptors; store exact ordered mapping and SHA.
- This is **multi-label document classification reformulated as a one-forward-pass 100-option decision task**. Not a claim to have invented the original EUR-LEX benchmark; do not call it multi-hop QA.
- Source: https://huggingface.co/datasets/coastalcph/lex_glue and https://github.com/coastalcph/lex-glue.
- An earlier EUR-LEX label-loader bug was documented by the official maintainers. Check revision and mapping rather than trusting stale HF cache.

## Required unified input

Each record must have `state` = legal text (truncated only according to frozen document-only policy),
`question` = "Select all EuroVoc concepts applicable to the legal document.",
`options` = 100 stable ordered dicts with distinct `key`, nonempty `desc` containing the *true concept name*, `target` = full 100-dimension multihot, `n_options=100`, `selection_mode="multi"`, `allow_empty` fixed from the official dataset-level protocol, `source="lexglue_eurlex"`, `lang="en"`, `split` and unique `id`.

**No per-example candidate sampling, no gold-informed negative selection, no dataset-specific count injected into the prompt.** All classes are evaluated. Freeze tokenizer (100 one-token answer slots), descriptions, input truncation, option ordering and allowable counts `M` before model comparison. Use head+tail deterministic token truncation of the *document only*, never drop candidates. Log document-retained fraction and token length.

## Three stages

### Step 0: CPU data-integrity gate (do not touch official test)

Load only train and validation initially. Verify 55k/5k rows, label-ID mapping and all 100 concept descriptors, no duplicate (document, split) overlap, per-question count distribution (including zeros), train label frequency, token length p50/p95, prompt context budget and each 100-slot single-token label index. Gold labels cannot silently be dropped. Prepare a 5k train + 1k validation pilot from the official train/validation (fixed deterministic IDs). No data generation based on gold cardinality. Run the production `train_multiselect.validate_row` over every converted record.

### Step 1: pilot (4B LoRA; no Test)

Same Qwen3.5-4B, prompts, fixed 100 options, lr-grid, steps, batch and update data. Run:
- BCE native single-readout baseline; select threshold on validation.
- BCE Continue, identical warm initial checkpoint and additional steps.
- BCE + learned count CE head + Top-m decoder, matched train budget.
- Count Anchor + joint Set CE.
- Anchor Detach.

For soundness, compare same-head parameters and same public legal set support and *proper set probabilities*. Retain a true-original BCE checkpoint separately; it is not substituted for matched BCE Continue. If anchor fails to beat BCE+learned-count in count NLL/Joint NLL, stop before full 55k-scale claims and diagnose.

### Step 2: full official train/validation and one frozen Test

Train on **all 55k train rows** with exactly the same data/compute budget per arm, 3 paired seeds (43/44/45) for decisive variants. Tune thresholds/temperatures/learning rate only on 5k official validation with a matched search budget. Lock model, optimizer, weight/hash, protocol, tokenizer, all mapping, number of epochs, decoder and calibration before opening the 5k Test. Evaluate once. If positive, 5-seed confirmation is optional.

## Main table and diagnostics

Primary (one benchmark): **official micro/macro F1** (comparable to EUR-LEX), **Count NLL**, **proper Gold-Set NLL**, **Exact-Set EM**, count MAE, count ECE and Brier, over/under-selection, calibrated threshold comparison, model forward passes, wall time and memory. Count marginalization must use stable logZ DP. For fair comparison, evaluate BCE's joint law conditioned on the *same public allowable counts* as Anchor. Also separately show unrestricted BCE F1 from the usual threshold decoder.

Failure/diagnostic: group by true count 0,1,2,3,4+ and dev-only BCE gold-count-NLL quartiles; do not use gold counts to route methods at inference. Check candidate ranking before inferring Count wins. On the same trained Anchor candidate scores, force a_m=0 and reevaluate. Match group bootstrap by document ID; do not treat 100 labels as 100 independent examples.

## Critical interpretation

- Strong real EUR-LEX result is **not assured** by the synthetic K=64 trend. The C-regime generator currently has two audit issues (constant "zero" strength cue, Gumbel Top-k not the intended fixed-cardinality Gibbs law); those must be repaired before reusing those curves as evidence.
- Count Anchor and independent Count CE share the same expressible set family via count-logit reparameterization; only claim better optimization or calibration if supported by matched controls.
- 100 options may be very imbalanced and exact-set EM difficult; don't call Count NLL gains a general correctness win without F1/EM comparison.
- One main benchmark is a deliberate **scope choice**, not a demonstration of cross-domain universality.
- Pairwise is out of this paper's training matrix; a short mathematical counterexample in the limitations section is enough.

## Prioritized actions

1. Implement EUR-LEX train/validation conversion with 100 frozen official labels/English descriptor mapping and robust label audit.
2. Tokenized prompt-length audit on Qwen3.5-4B; no candidate truncation.
3. Correct BCE + learned-count baseline and shared legal-set NLL, then 5k/1k pilot.
4. Run full 55k with paired 3-seed BCE Continue/learned-count/Anchor/Detach, official validation, one-time Test.
5. Write abstract only from measured authentic results, not oracle or synthetic results.

**No new EUR-LEX model results are claimed here.**
