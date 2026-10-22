# Does a stronger decision student remain a useful reasoning teacher?

Status: implemented planning, quality diagnostics and completed-run audits.
**No new model has been trained and no generational degradation has been observed
by this patch.** This is a separate, falsifiable exploration alongside the
[set-distillation objective](multiteacher_set_distillation.md).

## 1. The question specific to this decision-model workflow

The same checkpoint is used in two different computational roles:

* The deployed student reads candidate scores in one no-thinking forward pass.
* The self-teacher generates reasoning, then reads candidate membership with
  separate Yes/No queries and transfers an answer-set distribution.

Let D(theta) denote direct complete-set accuracy. Let R_theta(r|x) be the
reasoning policy, and Q_theta(S|x,r) the set law supplied by its reasoning readout.
Before filtering unusable chains, the effective teacher law is

$$Q_\theta(S\mid x)=\mathbb E_{r\sim R_\theta(\cdot\mid x)}
 [Q_\theta(S\mid x,r)].$$

Training the direct readout changes the shared Transformer. A better D(theta)
does not mathematically imply a better reasoning policy, better reasoning-endpoint
probabilities, or better targets for a future student. Freezing token embeddings
and the output projection alone does not freeze those functions.

This role mismatch is relevant to decision-model self-distillation, but it is
not a claim that generative forgetting or self-distillation failure is new in
machine learning. [SII, sections 6.1–6.3](https://arxiv.org/html/2610.03935v2)
already uses same-model reasoning to improve a direct readout and discusses
teacher-quality limitations. [StartLux, section 2.3.3](https://arxiv.org/html/2610.17309v1)
reports reasoning problems after decision training and subsequent reasoning
repair. Neither source establishes the recursive effect in this repository.

The meaningful proposed observation is:

$$D(\theta_{g+1})>D(\theta_g),\qquad
 U(T_{\theta_{g+1}})<U(T_{\theta_g}),$$

where U is **measured teaching utility**, not teacher accuracy under a new name.

## 2. Define teaching utility with matched students

For a fixed initial student theta_init, training budget B, data and seed xi, let

$$\theta'(T,\xi)=\operatorname{Train}(\theta_{\rm init},Q_T;B,\xi).$$

Use the same readout family, loss, gold mixture, data order, optimizer schedule,
update count and inference rule for all teachers. Define

$$U(T)=\mathbb E_\xi[D(\theta'(T,\xi))
 -D(\theta'(\mathrm{gold},\xi))].$$

The gold reference receives the same additional update budget with gold-only
targets. Direct comparisons between T0 and T1 also use paired seeds. The common
student initialization includes the count-head weights, when present.

Comparing a student initialized from theta_0 with another initialized from
theta_1 would conflate initialization and teacher quality. Comparing dev-selected
checkpoints from different steps would also change the effective training budget.
The planner therefore starts every teaching probe from the same base via
`--model`, with a fresh optimizer, and sets `--eval-every-frac 1.0` so only the
terminal update is eligible for selection. The result auditor requires the
actual selected step to equal the declared budget.

## 3. Keep two experiments separate

| Experiment | What changes | What it establishes |
|---|---|---|
| Fixed-teacher trajectory | Student continues for several rounds; teacher remains T0 | Behavior with an unchanged target source |
| Refreshed-teacher trajectory | Student continues; each new checkpoint supplies the next round's teacher | Behavior of recursive refreshing |
| Common-initialization teaching probes | Only teacher checkpoint changes; each student restarts from the same base | Relative teaching utility at the declared budget |
| Gold-only probe | Same initial model and update budget, no teacher targets | Gain attributable to distillation beyond additional gold training |

The two trajectories start from the same checkpoint. Their first round has the
same teacher and is a useful pipeline consistency check. Later rounds inherit
their own previous students; this is intentional for trajectories, but cannot
by itself isolate teaching utility. Future checkpoint identities are unresolved
bindings until the actual export exists.

Default pilot: two rounds, two views with two chains each, 16 student updates per
run, three paired probe seeds. These are bounded initial settings, not optimized
hyperparameters or evidence of sufficient statistical power. A 16-step pilot can
fail to reveal an effect that needs longer training. All teacher generation runs
are still potentially expensive: the plan includes every requested train/dev
row and K binary readouts per chain. Build an explicit group-separated pilot
dataset before planning if a smaller trial is required; no hidden `--limit`
changes the cohort for different arms.

## 4. What the code measures

### `audit_teacher_generations.py`: direct quality versus teacher quality

For each checkpoint, supply native direct predictions and teacher membership
files on the same held-out **dev** probe. The audit computes:

* Direct complete-set accuracy, F1 and probability metrics.
* Teacher membership BCE/Brier and cardinality NLL/Brier.
* The complete teacher mixture's probability of the gold set and its stable NLL.
* Exact teacher mixture MAP accuracy when K is at most the declared cap
  (default 12; maximum 16). Larger K skips this enumeration, not the probability
  metrics. Thresholding q or taking the count mode is not substituted for MAP.
* Completion, truncation, usable-chain counts and coverage by source.

Teacher components are conditioned on task protocol **before** mixing, with
fixed weight per view and equal valid-chain weight within a view. No product of
mean marginals is substituted for the mixture. It remains the documented
within-chain conditional-independence approximation, not a measured unrestricted
joint law or a rating of the prose reasoning's correctness.

Paired deltas use exactly the intersection of covered questions across all
generations. Available-cohort metrics are descriptive and separate. A teacher
that fails on more hard questions cannot appear better merely by dropping them.
Coverage collapse is a result to report even if the remaining common cohort is
small. No missing chain is converted into a negative target.

The generation diagnostic consumes raw dev membership files; it has no dev
target-building prerequisite. It can inspect existing partial outputs after a
readout reports unusable chains, without declaring that stage successful or
unblocking training. Missing or malformed files still fail validation. Train
target building and student training require complete coverage of the declared
training cohort.

Matched-setting flags require known and matching views, weights, chain counts,
sampling parameters, generation context length, readout token budget, prompt
versions and Yes/No token identities. Older caches without those fields remain
descriptive evidence; they cannot produce a verified matched-setting claim.
New native teacher caches record those fields. Resuming a cache with missing or
changed required budget metadata is rejected; use a new run directory.

The audit explicitly sets `teaching_utility_measured=false`. Direct EM going up
while teacher likelihood or MAP gets worse is an informative proxy pattern;
it is not yet evidence that a next-generation student learns less.

### `audit_teaching_utility.py`: actual downstream student evidence

After the matched probes finish, the utility audit reads native
`training_config.json`, `training_complete.json`, `evaluation_config.json`,
`metrics.json` and `predictions.jsonl`. It checks initial checkpoint identity,
file hashes, train/dev separation, optimizer and schedule, seeds, actual update
counts, terminal checkpoint selection, teacher provenance and the same decoder.
It recomputes aligned held-out metrics from per-question predictions rather than
trusting summary numbers alone, and reports per-seed gains over gold and between
teachers. It does not turn three seeds into a claim of statistical significance.

`--allow-planned` can inspect an entirely unexecuted plan and reports that teaching
utility is unmeasured. Missing pieces of a partially completed experiment are
errors; an incomplete run is not a successful comparison.

Checkpoint IDs must be explicit immutable content/revision identifiers. They
are operator attestations, not independent verification that local weight files
match the declaration. Include the Transformer, tokenizer/configuration and
count-head sidecars when identifying a locally modified export. Do not reuse an
ID after changing weights or compare runs under a mutable `latest` identity.

## 5. Commands and file contracts

Copy and edit
[`experiments/decision_teacher_rounds.example.json`](../experiments/decision_teacher_rounds.example.json).
Replace its checkpoint placeholder with `sha256:` followed by a real 64-character
digest, or `organization/repository@` followed by an immutable 40–64-character
revision. Set actual data, model and Python/torchrun paths. The example is a
template, not an assertion that those models or datasets exist locally.

```bash
python -I harness/plan_self_distillation_rounds.py \
  --config experiments/decision_teacher_rounds.example.json \
  --output-dir runs/decision_teacher_rounds_plan --validate-only

python -I harness/plan_self_distillation_rounds.py \
  --config experiments/decision_teacher_rounds.example.json \
  --output-dir runs/decision_teacher_rounds_plan
```

The output is a stage DAG in `plan.json`, plus teacher, generation-audit and
utility manifests. Commands are argv arrays, not shell strings. The planner
**does not launch** training or inference. Respect each stage's `depends_on` and
`argv_resolved` fields. An unresolved binding is not an executable command.

After a checkpoint is exported, supply a JSON mapping from its plan key (for
example `refreshed.r1`) to its real pinned identity using `--bindings`. Recreate
the resolved plan in a new output directory; it references the same planned
run paths. Resolve identities only after the corresponding export exists. Use
the emitted audit stage argv, which points to the matching resolved manifests.

The generation-quality audit can also be run independently:

```bash
python -I harness/audit_teacher_generations.py \
  --input data/multiselect_grouped_mixed/dev.jsonl \
  --manifest /path/to/generation_audit_manifest.json \
  --output runs/generation_quality.json
```

Its manifest declares `generations`, each with `id`, `checkpoint_id`,
`direct_predictions` and `teachers` (an existing teacher manifest). Paths are
relative to the manifest; view IDs such as `direct` and `verify` stay consistent
across generations. Native direct evaluation must receive `--checkpoint-id`:

```bash
python -I harness/eval_multiselect.py \
  --model /path/to/frozen/export --checkpoint-id "$CHECKPOINT_ID" \
  --input data/multiselect_grouped_mixed/dev.jsonl --expected-split dev \
  --output runs/frozen_direct_dev
```

Existing predictions retain their original provenance when rescored; the scorer
does not add an unverified model identity to an old file. Structured probability
records are checked against their candidate/count logits and MAP decoder, so
raw sigmoid probabilities cannot silently masquerade as induced marginals.

## 6. Mathematical interpretation and possible follow-ups

Under the usual differentiability/support conditions, changing the teacher
checkpoint changes both parts of its effective distribution:

$$\nabla_\theta Q_\theta(S\mid x)=
\mathbb E_{r\sim R_\theta}\left[
\nabla_\theta Q_\theta(S\mid x,r)
+Q_\theta(S\mid x,r)\nabla_\theta\log R_\theta(r\mid x)
\right].$$

This explains two possible failure mechanisms: the generated reasoning changes,
and the membership readout conditioned on that reasoning changes. Direct-only
student supervision does not separately constrain either term. The equation is
an explanatory identity, not a new gradient implemented by this patch. Our
student consumes frozen offline JSON targets, so it does not differentiate
through teacher sampling. Filtering incomplete chains also changes the observed
conditional mixture; completion and coverage must be examined separately.

If the matched teaching-utility drop is reproducible, useful next interventions
include delaying teacher refresh, keeping a frozen teacher mixture, or retaining
teacher-mode probability behavior on a separate replay set. A reference KL or
reasoning-replay term has substantial precedent (for example
[LLM2Jev](https://arxiv.org/html/2610.02076v1)); merely adding one is not a new
contribution. Define which function it preserves and test the resulting
direct-performance/teaching-utility tradeoff.

To distinguish reasoning generation from endpoint readout, a later experiment
could cross fixed old/new reasoning caches with old/new readout checkpoints.
The current same-checkpoint cache validator intentionally rejects that mismatch;
such a cross experiment needs its own explicit provenance/schema, not a bypass
of existing checks. This intervention is not implemented here.

## 7. Success and falsification criteria

The strong claim needs D(theta_1)>D(theta_0) **and** lower downstream utility
from T1 than T0 under matched probes, replicated across seeds and source/protocol
groups. A worse teacher NLL by itself is insufficient. If refreshed trajectories
underperform while matched probes do not, investigate changing initialization,
coverage or compounding errors before attributing the effect to teacher quality.

If both direct performance and teaching utility improve, reject the degradation
hypothesis for the tested regime. If only count information helps and exact set
CE does not beat the induced-marginal control, narrow the main method's claim to
useful set statistics rather than a superior optimization objective. Keep final
test data untouched until choices made on dev are frozen.
