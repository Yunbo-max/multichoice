# Implementation verification — 2026-10-22

Base repository revision: `870aa6f36a5d9693452e176396443bb9f62f9d74`.

## Checks executed on the original development host

| Check | Observed result |
|---|---|
| `python -m unittest discover -s tests` | 85 tests discovered; 82 passed, 3 explicitly skipped |
| `python -m compileall -q harness tests` | Passed |
| `bash -n harness/run_sii_style_teacher.sh` | Passed |
| Teacher QZ JSON parsing and embedded shell `bash -n` | Passed for the three updated teacher payloads |
| Isolated Python CLI help for builder/evaluator/teacher | Passed |
| Fake inference engine: direct/verify readout → actual teacher schema → target aggregation | Passed; no neural model was executed |

The tests cover exact-enumeration probability and MAP oracles, numerical score
and count-logit gradients, the single-choice CE reduction, the BCE degeneracy,
same-membership/different-cardinality teachers, group split/deduplication leakage
regressions, target/provenance failures, truncation policy, and frozen evaluation.
Error messages printed by teacher tests come from deliberate failure injection.

## Original development-host limitations

The host has Python 3.12 and no PyTorch, Transformers, vLLM, pyarrow, model weights,
or GPU. Installing the CPU torch wheel from the official wheel index was
unsuccessful (`No matching distribution found`); no package was installed.

The three skipped tests require real PyTorch and cover:

1. Native autograd versus the exact moment gradient.
2. Mixed-protocol/extreme-logit finite gradients.
3. Count-head checkpoint export/reload and tokenizer-label validation.

Also pending: actual tokenizer boundary checks, Qwen forward/training smoke,
native FSDP checkpoint export/reload, real parquet conversion, complete teacher
generation, and any teacher/student accuracy or calibration measurements.
The existing 58.63% dev EM belongs to the archival gold-BCE run, not this method.

## External follow-up report — attributed, not rerun here

In [the PR review comment](https://github.com/Yunbo-max/multichoice/pull/1#issuecomment-6349668026),
Claude reports running the PR code on its native PyTorch host:

* `python -m unittest discover -s tests -v`: 85/85 passed, including the three
  tests skipped on the original development host.
* Real Qwen3.5 tokenizer validation: Yes ID 9175 and No ID 2665 are single tokens;
  all 2,040 dev rows passed the membership-readout validation.

This documentation update checked the review and its commit, but did not rerun
that host or obtain attached raw execution logs. These reported results do not
replace the original local execution record above. The reviewer also explicitly
leaves real model forward, FSDP export/reload, and GPU teacher/student training
unverified. No accuracy, calibration, or throughput result for the new method
follows from the unit tests or tokenizer checks.

The [mathematical corrections](review_corrections_2026-10-22.md) distinguish the
implemented independent-head model from the unimplemented alpha-mixture and
empirical-map proposals. That earlier follow-up changed documentation only.

## Objective controls and teacher-generation exploration — later code update

Parent revision: `c8cf9bc5fbefc4555164401b9b1653007a60e98b`.

This update adds information-matched marginal/count losses, stable probability
metrics, generation-quality diagnostics, a bounded stage planner, and native
artifact checks for matched next-student teaching utility. It also records
sampling/readout context budgets so changes cannot silently pass as matched
teacher settings. The full protocol is in
[decision_teacher_generations.md](decision_teacher_generations.md).

| Check executed on this development host | Observed result |
|---|---|
| `python -m unittest discover -s tests -q` | **150 discovered: 144 passed, 6 skipped because PyTorch is absent** |
| `python -m compileall -q harness tests` | Passed |
| `bash -n harness/run_sii_style_teacher.sh` | Passed |
| Isolated Python CLI help for planner, generation audit, utility audit and evaluator | Passed |
| Native teacher schema through fake engine → target builder → audit protocol signature | Passed; no neural inference |
| Example pilot JSON | Parsed; paths and checkpoint placeholder still require actual experiment inputs |

The new scalar checks compare induced inclusion/exclusion probabilities with
exhaustive subset enumeration, verify objective values and finite-difference
stationarity/count coupling, and retain extreme-logit losses. The objective API
preserves the old set CE default and rejects ambiguous control combinations.

The new audit fixtures test same-membership/different-count teachers, true
mixture probabilities, exact small-K MAP, native token-budget provenance, paired
cohorts, coverage loss, model/input/order contamination, train/dev passage leakage,
fixed/refreshed lineage, unresolved checkpoint bindings, actual update budgets,
common student initialization, teacher extraction recipes, and fabricated or
missing result summaries. They are synthetic contract checks, **not empirical
model results**. A diagnostic can inspect raw dev outputs with missing chains;
training still requires complete train targets.

Two issues found in review were corrected and covered by regressions:

1. Stable count log probabilities now drive count NLL as well as joint-set NLL;
   extremely wrong high-confidence predictions are not hidden by clipping.
2. Structured records containing candidate logits must have the matching induced
   marginals and deterministic set MAP. Saved raw sigmoids cannot be scored as
   if they came from the structured law. Legacy clipping never creates MAP mass.

The six skipped tests are the original three native PyTorch checks plus three
new autograd checks: both control objectives and all gradients against scalar
finite differences, mixed-protocol/extreme-score gradients, and log-DP backward
against exhaustive differentiable enumeration. Neither the system Python nor
the primary runtime Python has PyTorch/Transformers installed. No new real-model
forward, GPU training, FSDP export/reload, latency measurement, or teaching-utility
experiment was executed. The earlier external 85/85 report does not verify these
new PyTorch tests or new code paths.

## Scope

Only code, tests, teacher job definitions, and documentation are changed. Published
datasets and run results are retained. No GPU job was submitted, no model was
trained, and no new empirical advantage or first-ever novelty is asserted.
The multi-worker review was an engineering/math cross-check, not an independent
formal research acceptance process.
