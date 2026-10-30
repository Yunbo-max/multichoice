# Paper 1 — Option independence: preliminary audit, **not proof** (2026-10-30)

**Correction:** an earlier version of this document concluded that CMB/MultiRC options are conditionally independent, based on aggregate count frequencies and an all-zero saved count-head **bias**. Neither observation establishes that conclusion.

## What was actually observed

- The average TEST histogram of the gold number of selected answers and the Poisson-binomial count histogram induced by BCE are close (largest reported count-frequency discrepancy about 2.5 percentage points).
- On the frozen, already-opened 2,140-question TEST, anchored Set CE roughly ties BCE continuation: exact-set EM 0.6173 versus 0.6182; independent count-head Set CE scores 0.4005.
- The reported count-head *bias* vector was zero for inspected checkpoints. This is not a measurement of a_m(x)=W_m h(x)+b_m. A zero bias is compatible with a nonzero, input-dependent residual and could also expose a training/export bug if all parameters are unexpectedly constant.

## Why the prior inference was invalid

Matching **aggregate** count distributions does not prove that Y_i and Y_j are conditionally independent given the full question/context x. Different joint set laws can have the same option marginals and cardinality law. For example, the distribution with equal mass on AB and CD and the distribution with equal mass on AC and BD have identical membership and count statistics yet different co-occurrence rules.

With full count support, setting the anchor residual to zero reduces its *probability law* to BCE; with nonempty support, it reduces to BCE conditioned on nonempty sets. However the optimization gradient of a trained full anchor may differ and the input-dependent head has not yet been verified to be negligible.

## Required follow-up, using DEV only

See [the executable count and pairwise audit](paper1_count_pairwise_protocol.md). It adds (1) per-question BCE gold-count NLL quartiles; (2) paired same-example delta EM and count NLL with passage-group bootstrap and matched seeds; (3) direct residual output a_m(x) in the exported evaluator; (4) frozen-candidate zero-residual ablation; (5) a synthetic exact pairwise counterexample.

**Do not interpret post-hoc DEV strata as a deployable selection rule, and do not tune a method on the old TEST.** The new GPU DEV replay has been staged, not run by this documentation update.

## Cross-reference

- [Frozen 24-checkpoint TEST](paper1_static_test_results.md)
- [24-run matched DEV](paper1_static_4b_results.md)
- [Original count/SetJev controls](seed_results_2026-10-23.md)
