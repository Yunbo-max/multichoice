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
empirical-map proposals. This follow-up changes documentation only.

## Scope

Only code, tests, teacher job definitions, and documentation are changed. Published
datasets and run results are retained. No GPU job was submitted, no model was
trained, and no new empirical advantage or first-ever novelty is asserted.
The multi-worker review was an engineering/math cross-check, not an independent
formal research acceptance process.
