> **Further consolidation (2026-11-18):** The RoMQA preparation branch was **merged into** the retained `research/static-setjev-paper-20261027` branch using a two-parent merge commit [`236daacd`](https://github.com/Yunbo-max/multichoice/commit/236daacddb2d26fbf5f945b122d1390a32d4df22); all Paper 1 code and commit history now live on **one canonical Paper 1 branch**. The former RoMQA ref is now a tenth deletion candidate. The final target is **3 branches** (`main`, canonical Paper 1, latest Paper 2), not 4. PR #9 now follows the merged canonical branch. Nothing was force-pushed or cherry-picked; no real RoMQA results are claimed. 

> **Cleanup actions completed (2026-11-18):** PRs #1–#8 were commented and **closed without merge**; PRs [#9](https://github.com/Yunbo-max/multichoice/pull/9) and [#10](https://github.com/Yunbo-max/multichoice/pull/10) now both target `main`. GitHub verification: **2 open PRs, 13 branches**. The ten obsolete branch REFS **have NOT been deleted** because the available GitHub connector exposes create/move-ref but no delete-ref action. This is deliberately not reported as a completed remote branch cleanup. [Exact SHA-audited pruning script](../scripts/prune_obsolete_research_branches.py) is available for an authenticated clone. Run its dry-run first. After explicit deletion via the script or GitHub UI, verify 3 remaining remote branches.

# multichoice branch cleanup — audited 2026-11-18

## Goal and safety

Retain the reproducible scientific record for Paper 1 (Count Anchor) and the
latest separately maintained Paper 2 (Sokoban), while removing obsolete
intermediate refs. **A branch is not an independent benchmark or evidence of
a research gap.** Earlier pilot numbers with flawed task construction,
leakage, undertrained controls or invalid evaluation are historical diagnostics
only, not baselines for a new paper.

At audit start there were 13 branches, 10 open PRs. GitHub comparisons verified
that **every branch proposed for deletion is an exact commit ancestor of a
retained research leaf** (GitHub status `ahead`, `behind=0`). The latest
leaf retains all past code and commit ancestry even after old ref deletion.
Default `main` is retained. No research content is force-rewritten.

## Keep these three refs

| Branch | Role | Scientific status |
| --- | --- | --- |
| `main` | Stable default and repository entrypoint | Historical code, not the latest research experiments |
| `research/static-setjev-paper-20261027` | **Single canonical Paper 1 branch**, full Count Anchor / CMB/MultiRC / controlled I/C/R / EUR-LEX **plus the RoMQA preparation code, exact frozen controls and offline CI** | RoMQA real data blocked; only audited prior results can support paper claims |
| `research/episodic-tree-sokoban-20261027` | Separate Paper 2 latest branch | Earlier episodic experiments contain failures; incomplete visual DEV checks; do not present as positive Paper 1 baseline |

## Delete only these ten superseded branch pointers

All are exact ancestors of the named leaf; the table's SHA is the audited
remote head for a **force-free lease/snapshot check** before deletion.

| Proposed removal | Audited SHA | Retained descendant | Rationale |
| --- | --- | --- | --- |
| `codex/multiteacher-set-distillation` | `c77214b5c930140f4767cffea6deee83cf2f842f` | Paper 1 canonical | Old self-teacher/soft-label pilots (truncated or overselecting) cannot serve as strong baselines |
| `research/romqa-anchor-icml2027-20261114` | `d6fd50a99b8f03cb84fd9e617d0356512ba19244` | **Paper 1 canonical** | Former separate RoMQA preparation branch; both commit lineages are now preserved in the canonical Paper 1 merge commit `236daacddb2d26fbf5f945b122d1390a32d4df22` |
| `research/recurrent-set-decision-20261024` | `07d152b12018e3cc92deea577e80e18af255c82c` | Paper 2 latest | Initial recurrent revision experiment, later discovered loss/metric faults |
| `research/recurrent-set-debug-20261024` | `735bc69b83414e12ba80407d2ddd0865a17ca2f2` | Paper 2 latest | Intermediate bug fixes incorporated downstream |
| `research/episodic-set-memory-20261024` | `823ec93ca6ed3366a7f2ab71ec125031e7a80a3a` | Paper 2 latest | Initial persistent-memory formulation, unverified Qwen benefit |
| `research/episodic-set-debug-20261025` | `c34ef0982f5eecf8c02400a18813a9d76f46d6f7` | Paper 2 latest | Causal chronology fix; earlier online results confounded |
| `research/episodic-identifiability-debug-20261025` | `aa793886851dc46c2b542f03c3a4a39e966c02cd` | Paper 2 latest | Initial symmetry shortcuts; no independent valid positive result |
| `research/episodic-probe-group-split-20261025` | `d8e56f03c302d248cd625766f61290b706a92757` | Paper 2 latest | Repair of episode-level probe leakage |
| `research/episodic-action-source-diagnostic-20261025` | `8fcbc13bf0156f9e8eb19406275ce42f40798735` | Paper 2 latest | Diagnosed expert-action shortcut and policy distribution shift |
| `research/episodic-lora-reliability-20261026` | `e3923206fa59c79d720fe9cabf569eaabcdfc4a3` | Paper 2 latest | Predecessor with LoRA checkpoint fixes; all commits included in latest Sokoban branch |

**Do not delete an old ref if it moved after this inventory.** The audit
provides a dry-run, SHA-frozen local removal script:
`scripts/prune_obsolete_research_branches.py`.
The script refuses unexpected SHA changes, validates preserved ancestry, and
does not delete on its default invocation. Run it with `python scripts/prune_obsolete_research_branches.py` for dry-run and `python scripts/prune_obsolete_research_branches.py --delete` for interactive, authenticated deletion. **No branch is reported as deleted
until remote Git refs are checked afterward.**

## Validity / baseline ledger for the paper

| Experiment | What is actually supported | What must NOT be claimed |
| --- | --- | --- |
| CMB-Exam + MultiRC 3-seed frozen TEST | BCE EM 0.6182 vs Anchor 0.6173: near-tie negative real-world evidence | Anchor superiority or proof of conditional independence |
| EUR-LEX 5k Train / 1k DEV frozen BCE feature heads | Count NLL BCE 2.132, zero-init Anchor 2.10, **TRAIN-count-prior Independent 1.98** | Anchor beats a properly initialized learned count head |
| Controlled-C K=8 matched 4B, 3 seeds | Count KL BCE .333 vs Anchor .278; Joint KL BCE .419 vs Anchor .366 (synthetic) | A general real-world/ICML-ready improvement by itself |
| Controlled-I | BCE beats Anchor on independent-data Joint KL | Any universal improvement over independent BCE |
| Controlled-R / learned pairs | Dense/greedy/Gumbel matching edge F1 near chance | Learned pairwise interaction head successfully captures true relations |
| K16/32/64 early synthetic sweep | Historical exploratory numbers | Paper main positive figure: public pressure cue and conditional Gumbel Top-k sampling were defective, needs corrected rerun |
| Old 5k soft teacher | Historical negative pilot with high truncation and over-selection | A fair state-of-the-art baseline/research gap argument |
| Independent Count from zero initialization only | Useful optimizer diagnostic | Strong Independent Count baseline absent TRAIN-count-prior correction |
| RoMQA Closed | Converter, frozen-head comparison, CPU tests on mock data ready | Real RoMQA result or usable download: third-party official README ZIP returned HTTP 404 |
| QAMPARI / SATA / Coding | Other candidates and separately staged code | Direct substitute for official RoMQA Closed or a positive Paper 1 benchmark |
| Old episodic and Valen visual | Retain latest Paper 2 research code/history | Old memory carry benefit or earlier 85.94% 64-row visual DEV as reliable main benchmark result (padding-position caveat) |

## Paper 1 primary focus

**When Do Multi-Answer Decision Models Need Cardinality Potentials?**
Compare native BCE, full Anchor, Anchor Detach and **TRAIN-count-prior
Independent Count**, with matched candidate inputs/data/compute and proper
Count NLL + Joint-Set NLL. MultiRC/CMB and EUR-LEX are valid negative controls;
Controlled-C supports a narrow synthetic positive mechanism. Real
constraint-varying positive evidence still needs an accessible and audited
benchmark. Do not claim that RoMQA is available or that alternate dataset
results have been obtained.

Both the validated core results and the RoMQA preparation/ICML runbook now live
on **one canonical Paper 1 branch**:
`research/static-setjev-paper-20261027`.
The consolidation is a true two-parent commit `236daacddb2d26fbf5f945b122d1390a32d4df22`,
with the original static and the former RoMQA tip as parents. The
`docs/paper1_romqa_icml2027_protocol_20261114.md` runbook remains available
under that same branch.

## PRs are not the same as branch refs

The old PR stack used ten PRs chained through ancestor branches.
For clarity, obsolete PR conversations can be **closed without merging**
and are still available for provenance. This does not itself delete a
branch; the ten Git ref deletions require GitHub's branch-delete UI,
an authenticated CLI, or another authorized delete-ref action.
Retain only Paper 1 / Paper 2 active review PRs and point their bases
at `main` so no live PR depends on a deleted intermediate base.

**No numerical result, paper claim, or method should be silently changed
just to reduce the number of branches.**
