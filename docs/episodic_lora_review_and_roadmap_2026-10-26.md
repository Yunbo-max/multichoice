# SetJev post-push technical review and staged research plan
Date: 2026-10-26
Base audited: 8fcbc13bf0156f9e8eb19406275ce42f40798735
Safe fix branch: research/episodic-lora-reliability-20261026

## Executive judgment

A/B/C experiments do NOT isolate a unique failure cause. A/B use a frozen
Qwen backbone and differ in action trajectories; B is one round of offline
policy-cache aggregation and resets the episodic head for training.
C trains Qwen LoRA on true self-executed dynamics, but its current published
pilot covers only 200 episodes x 2 epochs with a train loss of 0.439 -> 0.429
under the normalize-by-7 objective. This is insufficient to rule out
representational limits or suggest millions of extra episodes are necessary.

A relevant random-set uniform reference (2-of-7) is:
    log(21)/7 = 0.435
and a known reachable-six rule classifier's uniform NLL is log(6)=1.792
(unnormalized). Keep these distinct.

The latest upstream doc docs/episodic_abc_conclusion.md states C cannot be
evaluated with a LoRA-aware evaluator. That statement was superseded by
harness/eval_episodic_lora.py in a later commit. The evaluator was present
but restored LoRA with unverified strict=False and hard-coded 16/32.

## Actual code issues fixed in this branch

### P0: Double 4B model loading, discarded SSM config
train_episodic_lora.TrainableQwenEncoder.__init__ loaded Qwen twice:
the first passed a modified mamba_ssm_dtype config, then the second
from_pretrained immediately overwrote self.model without that config.
It now loads exactly once with the chosen config, records the loading
precision and adapter metadata, and has a loader-source regression test.
This alters the effective base model relative to the old broken C pilot:
do not claim old/new differences are caused only by training changes.

### P0: Silent partial/wrong LoRA restore in C evaluation
eval_episodic_lora originally used the default LoRA 16/32 and
load_state_dict(strict=False), not verifying any matching adapter names
or shapes. The evaluator now:
- detects checkpoint type/schema and model identity;
- derives LoRA rank, alpha, target module names and dtype from new metadata;
- checks exact adapter key set, shapes, finiteness and restored values;
- supports older C checkpoints with explicit legacy rank/alpha fallback
  (still requires exact key and shape match).
The base model path provenance is recorded; full checkpoint-by-weight
hashing remains a future requirement.

### P1: best.pt saved at last epoch, no validation or gradient proof
Earlier C training accepted --dev-jsonl but never used it. It overwrote
best.pt at the end of EVERY epoch and only printed train_nll.
The fixed implementation:
- selects a fixed DEV-only episode subset independently from the train shuffle;
- performs model-own-action, on-policy DEV evaluation for each epoch;
- selects best by DEV step exact-set EM, tie-break lower normalized set NLL;
- writes atomic best.pt and a small training.json report;
- proves the LoRA submodules receive nonzero finite gradients on first
  optimizer step, rather than only the tiny episodic head changing;
- stores LoRA hyperparameters, base model ID/path, selected epoch and metrics;
- errors on an existing output folder instead of overwriting old evidence.

The DEV sample is intentionally small by default (12 episodes) to keep 4B
pilots affordable. It is a smoke checkpoint selector, not a final benchmark;
increase to 64/256 after the first pipeline gate. Never tune on TEST.

### Regressions / smoke entrypoints
- harness/episodic_lora_checkpoint.py
- tests/test_episodic_lora_checkpoint.py
- qz/run_episodic_set.sh smoke
- qz_lora/episodic-lora-reliability-train-smoke-s43.json
- qz_lora/episodic-lora-reliability-dev-eval-s43.json

The staged QZ jobs execute only when explicitly submitted; they do not launch
automatically. They use fresh runs/episodic_lora_reliability_smoke_s43 and DEV
evaluation only. Run CPU regression first.

## Seven sequential gates: do not skip to bigger training

### Gate 0 — engineering integrity (blocking)
Run full episodic smoke; ensure one base Qwen load per encoder, finite nonzero
LoRA gradients, finite head gradients, correct adapter save/restore, a true DEV
selected epoch, and deterministic eval on a small fixed DEV subset.

    git fetch origin
    git switch research/episodic-lora-reliability-20261026
    bash qz/run_episodic_set.sh smoke

Then review/submit, in order:
    qz_lora/episodic-lora-reliability-train-smoke-s43.json
    qz_lora/episodic-lora-reliability-dev-eval-s43.json

Stop if adapter restored tensor count is zero, if gradients vanish, if
model/LoRA output differs suspiciously after checkpoint readback, or if
DEV is not actually evaluated. Do not open TEST yet.

### Gate 1 — verify a single-decision rule foundation (non-recurrent control)
Create balanced single-step examples: known rule demonstration D, visible set
x, explicit public 2-of-7 protocol, target shift_k(x). Train a small LoRA
and a set-only head with the *same answer slot*, without persistent GRU.

Controls:
  (i) current frozen Qwen and native BCE readout,
  (ii) LoRA one-step + full 21-subset set decoder,
  (iii) LoRA one-step + reachable-6 structured decoder,
  (iv) a symbolic rule translator (oracle upper bound),
  (v) a randomized six-shift predictor (1/6 reference).

Test counterfactual pairs with IDENTICAL x but distinct initial demos.
First prove overfitting of a small, deterministic 16-32 example subset.
If tiny train exact EM remains below 95% after a sufficient, audited
optimization budget, debug answer-slot labels, evidence encoding and
gradients rather than scale up. For DEV, report question EM, conditional
NLL, prediction entropy and pair accuracy for both contradictory demos.

### Gate 2 — isolate memory writing/reading
Using the proven one-step encoder and decoder, test:
  (i) just-prime S_0 with zero rule switches;
  (ii) carry updates with rule switch prob 0;
  (iii) carry updates with rule switch prob 0.25 and after-switch evidence;
  (iv) reset every environment step;
  (v) the same model given the rule demo explicitly in current prompt
      (a text-history oracle information reference).

Measure held-out episode group-disjoint rule probes, counterfactual paired
decisions, memory representation distances, and error by steps since last
rule update. A linear probe alone is not a definitive capacity test.
Do NOT infer "observe doesn't forget" solely from carry > prime-only when
rule switching is enabled.

### Gate 3 — evidence-conditioned memory write (if Gate 2 fails)
Separately represent rule evidence from self-chosen action identity:
  z_t = enc(visible demonstrated evidence_t)
  M_{t+1} = (1-g_t) M_t + g_t update(M_t, z_t)
Here g_t is supervised by *whether new rule evidence was actually observed*,
not by whether the agent action was correct. Do not infer k directly from
the model's own action. Add a modest, separate auxiliary rule classifier
on M_t with explicit gradient-norm/cosine audit versus set NLL; avoid
uncontrolled shift-head competition seen in v5/v6.

Compare:
  memory that updates on every action;
  evidence-gated memory;
  prime-only;
  explicit text history given equal information budget.

### Gate 4 — iterative on-policy data aggregation (if needed)
Current B is one policy-cache collection + expert merge, trained from
scratch; it is NOT an exhaustive DAgger null result.

Run R=0,1,2,3 policy rollouts with a fixed rollout budget and oracle
label for the actually visited current state. Store the real executed set
and its resulting observation as future memory inputs; do not use the
expert action in future policy context. Keep an equal-updates replay/SFT
control and original frozen-backbone B arm. Analyze teacher-forcing fraction,
visited-state coverage, shift changes and recovery time.

Use multiple seeds and compare stage-wise on-policy DEV EM/NLL rather than
only train loss. Continue only if genuine within-seed improvement exceeds
a matched no-aggregation run; otherwise stop data aggregation and investigate
the state representation.

### Gate 5 — scale only after a positive mechanism gate
Extend from 32 to 200 to 1024+ episodes and from the 4B LoRA head to more
adapter modules if (and only if) single-step competence and cross-step
retention are verified. Larger scales before those gates may merely produce
more correlated duplicates of 7-cell shifts.

Evaluate new permutations of action descriptions, independent rule families,
alternate numbers of options/answer counts, history lengths beyond training,
and switching frequencies 0 / 0.25 / 1.0. Use three seeds, locked Test,
wall clock / GPU hours, Train/DEV/Test dataset hashes.

### Gate 6 — StartLux-style calibrated deliberation after accuracy exists
Borrow StartLux's ordering: decision supervision first, reasoning-improvement
optimization second. Do NOT copy a text-reasoning GRPO objective blindly
into differentiable recurrent latent updates.

For coherent multi-select set distributions Q_r(S), train a verified
improvement in proper set log score from initial to final readout plus
a cost for extra recurrent steps:

    J = -log Q_R(S_gold) + lambda_compute * R

If initial Q_0 is held frozen and fixed, the improvement in log-probability
telescope to the final objective, so report a genuine adaptive-budget or
memory-state benefit rather than claiming a novel "delta reward" alone.

Add StartLux Instant / AutoThink as external comparators only when supplying
the SAME visible history and accounting for thought tokens / latency.
StartLux's high numbers on static closed-set decisions are not evidence that
its recurrent Memory solution is already solved.

## Stop/go recommendation

Immediately: Gate 0.
Then: Gate 1 (single-step demonstration-conditioned 2-of-7 choice).
Only then: Gate 2/3 Memory, Gate 4 genuine closed-loop aggregation,
Gate 5 scaling and cross-rule generalization, Gate 6 decision-policy refinement.

Prior A/B/C docs' "failure proven due to training scale, not representation"
should be treated as a **hypothesis**. A small C pilot with 0.439 -> 0.429
normalized train loss and no valid LoRA readback does not identify its cause.
