# Recurrent SetJev static-debug audit — 2026-10-24

**Audit base:** 07d152b12018e3cc92deea577e80e18af255c82c  
**Original branch:** research/recurrent-set-decision-20261024  
**Safe fix branch:** research/recurrent-set-debug-20261024

**Important:** This is a source-level review with new regression tests,
NOT a claim that the QZ/H100 experiments have been executed. There are no
recurrent GPU training metrics/checkpoints committed in the audited GitHub
branch. Do not infer remote-job completion from code commits.

## Findings and actual fixes

### [High] New revision loss was dead code in training

At audit base, revision_nll exists in harness/recurrent_set_decision.py, but
harness/run_recurrent_set.py and harness/toy_sequential_set.py only call
loss_over_steps(... gold_set_nll ...). Running the prior
qz/run_recurrent_set_pilot.sh "train" cannot test Experiment D.

**Fix:** Extend loss_over_steps with opt-in revision_previous,
revision_weight, unchanged_weight. Qwen static pilot constructs
revision_previous from the **frozen r=0 native MAP**, not the gold set.
New flags: --revision-weight, --unchanged-weight. QZ runner has a separate
"revision" stage; it does not launch automatically. The synthetic sequential
environment uses its preceding action state as the optional previous set.
Set NLL is ALWAYS retained as primary objective.

### [High] "Pure revision" semantics and gradients were false

The original revision_nll had

    weighted BCE(scores, new, changed-only) + exact-set NLL(scores,new),

but described it as having gradients "only" on changed options, with
unchanged options inherited from prev. The log-partition of the NLL has
gradients for every candidate score. When unchanged_weight=1, that original
function was not ordinary set re-prediction; it was set NLL + BCE.

**Fix:** A separate revision_auxiliary_bce implements changed-option
supervision. revision_nll is retained for compatibility but precisely
documented as exact set NLL + revision auxiliary BCE. No claim that unchanged
scores are mathematically held fixed or that this is a pure edit policy.

### [High] Repair/broken metrics used the wrong comparison

toy_sequential_set.rollout previously passed the **previous environment
state bits** into revision_metrics as if they were the first prediction for
the new target. Its repair/broken rates thus measured state-transition
differences, NOT whether recurrence fixed or broke an initial decision.

**Fix:** compare the exact **r=0 MAP** against **r=R MAP** on the SAME
observed state and gold target. Updated toy output key:
revision_vs_initial_decision. Qwen evaluate_model also reports initial_em,
repair_rate, broken_rate, rescued_questions, damaged_questions using this
same-query pairing. This restores a meaningful "did deliberation help"
audit.

### [Medium] MLP control is not matched in parameter count / FLOPs

Current "mlp" has one distinct d->d layer per step and no latent carry;
GRU uses shared recurrent gates and more weights; both also have different
information paths. Treat as an unshared feed-forward diagnostic, NOT as
an isolated test of parameter sharing or an equal-compute control.
Source comments corrected. QZ runner includes this diagnostic as another
arm. A properly capacity/FLOP-matched control remains future work.

### [Medium] Revision label validation silently truncated floats

set_revision._validate_labels used int(v), accepting 0.3 as 0 and 1.7 as 1.
It now requires 0/1 values without implicit rounding.

## What is still *not* implemented or verified

1. **No cross-environment recurrent hidden-state carry.** The existing
   rollout calls RecurrentSetDecision afresh at every environmental time step.
   The GRU loops inside a *single decision request*, but its latent z_R is
   not automatically passed into the next environment request. The toy has
   real state feedback through executed action sets, not a persistent latent
   memory policy. Treat those as two separate mechanisms.
2. **No end-to-end Qwen updates** during recurrent training. The Qwen encoder
   is frozen into a feature cache; only the small recurrent head trains.
3. **No actual dynamic SetPivot dataset** with structured, semantically
   annotated relevant vs irrelevant environment interventions is in this
   branch. Static native MAP is a useful previous-set probe, not a substitute.
4. No full StartLux RLCD², GRPO, learned policy-value update, or true
   sequence-return optimisation has been implemented.
5. Model checkpoint identity uses an explicit label, not a cryptographic
   hash of all Qwen weights. Verify weight provenance before a paper claim.
6. The count-potential set family lacks general pairwise option dependencies.
7. No parameter/FLOP-matched recurrent vs non-recurrent model is available.
8. **Tests added but not executed on the QZ/GPU environment in this audit.**
   No GPU test or numerical result is asserted.

## Reproduction checklist — in order

Run with the branch checked out on your trusted machine:

    git fetch origin
    git switch research/recurrent-set-debug-20261024

    python -m unittest discover -s tests -p 'test_recurrent_set_decision.py' -v
    python -m unittest discover -s tests -p 'test_recurrent_debug_regressions.py' -v
    python -m unittest discover -s tests -p 'test_revision_and_controls.py' -v
    python -m unittest discover -s tests -p 'test_set_revision.py' -v
    python harness/run_recurrent_set.py smoke

    # Small action-environment smoke (train only on toy oracle demonstrations)
    python harness/toy_sequential_set.py --train-episodes 128 \
       --eval-episodes 32 --epochs 2 --steps 4 --revision-weight 1.0

    # GPU: explicitly chosen stages, NEVER all stages at once
    bash qz/run_recurrent_set_pilot.sh cache
    bash qz/run_recurrent_set_pilot.sh train
    bash qz/run_recurrent_set_pilot.sh revision

The 5k training split / full 2042 dev split must use same BCE model identity.
Each "train" arm is 5 epochs, 4 inner steps, matched seeds 43/44/45.
The revision arm adds only an auxiliary BCE term; it still optimizes exact set
NLL. For scientific evaluation, report initial/final EM, rescue/damage, NLL,
count error, total parameters, activated trainable parameters, FLOPs per depth,
data split hashes, and wall-clock. Do not use the selected dev scores as
held-out test results.

## Research decision gate

Before BDH-CQ-like or RLCD² claims, demand:
(a) +EM / +NLL against the same native 0-step decoder;
(b) positive net rescue - damage;
(c) gain versus non-iterative compute-matched controls;
(d) evidence the learned state handles interventions (not just score sharpening);
(e) separate demonstration that carryover memory improves true multi-step tasks.
