# Multi-view self-distillation of answer sets

Status: implemented research candidate, **not a trained or validated model result**.
The existing gold-BCE run and its published data remain an archival v1 reference.
Rebuild group-separated data and rerun matched baselines before making a method claim.

The [review corrections](review_corrections_2026-10-22.md) supersede the inaccurate
claims in the initial simplification review. The implementation below still uses
the independent count head. Alpha-mixture and empirical-map targets are proposed
ablations, not implemented commands or demonstrated results.

## 1. Research question and nearest work

Can a one-pass decision LM learn a better **complete answer set** when self-teachers
transfer both candidate membership and the distribution of the number of correct
answers, rather than only an averaged membership vector?

Multiple self-teachers, reasoning samples, and adaptive teacher weights already
have substantial precedent. The proposed contribution is their precise adaptation
to **set-valued decision readout**, with a coherent training/inference distribution.
It is not a new probability-model family or a claim of first-ever self-distillation.

| Primary source | Closest overlap | Distinction to test here |
|---|---|---|
| [SII / General Decision Models](https://arxiv.org/html/2610.03935v2), 2026 | Two reasoning-chain categorical readouts teach a no-thinking decision head | Multiple simultaneously correct candidates require set membership and cardinality |
| [Large Decision Models / SPRD](https://arxiv.org/html/2610.15501v1), 2026 | Frozen teacher, four sampled answers, categorical vote targets and consistency filtering | Our target represents subsets rather than a vote over mutually exclusive labels |
| [Self-Distilled Reasoner / OPSD](https://arxiv.org/html/2601.18734v3), 2026 | Same model, privileged solution context, on-policy token distillation | Offline frozen teachers transfer answer-set statistics; students generate no reasoning |
| [AVSD](https://arxiv.org/html/2605.20643v1), 2026 | Multiple privileged views of the same model; geometric consensus and gated arithmetic residual | Merely adding prompt views or disagreement weighting is not our claimed contribution |
| [MT-SDPO](https://arxiv.org/html/2609.02548v1), 2026 | Verified frozen teacher feedback consolidated through an EMA self-teacher | We do not require verifier-selected specialists or on-policy token trajectories |
| [COMPACT](https://arxiv.org/html/2601.13992v1), 2026 | Adaptive teacher weights using consensus, information gain, and student difficulty | Fixed teacher weights isolate the set target from weighting-policy effects |
| [Tarlow et al., UAI 2012](https://proceedings.mlr.press/r10/tarlow12a.html) | Classical cardinality potentials and exact inference | Credit the model family; novelty must come from the decision-LM construction and evidence |

This bounded audit found no identical complete construction in the listed close
papers. It does not prove absence from all prior work. An application-level
contribution to Jev-style decision models still needs reproducible comparisons.

## 2. Data protocol is part of the input

Every new row declares `selection_mode` and `allow_empty`:

* `single`: exactly one answer is allowed, so permitted counts are `{1}`.
* `multi`, `allow_empty=false`: counts are `1,...,K`.
* `multi`, `allow_empty=true`: counts are `0,...,K`.

**Never infer task mode from the gold positive count.** A MultiRC question that
happens to have one positive remains an all-that-apply question at inference.
Legacy rows default to multi/nonempty. The optional CMB `mixed` conversion adds
native single-choice questions, based on their original task type.

MultiRC passages and original paragraph identities are kept together. Grouping
uses connected components under source-scoped paragraph ID or normalized full
passage equality. Train/dev splitting is deterministic and approximately matches
requested sizes without breaking groups; full-content/group overlap is audited.
This is not a semantic near-duplicate detector. Old checked-in splits are not
silently overwritten.

## 3. A finite teacher workflow

1. Train gold supervision on the rebuilt train split. Freeze a selected checkpoint.
2. Use that same checkpoint in two teacher views: `direct` reasoning and `verify`
   reasoning that checks the candidates. Start with two sampled chains per view.
3. After each chain, ask a separate binary membership question for **each** option,
   retaining the question, all candidates, task rule, and chain in context.
4. Read the logits of tokenizer-verified single-token `Yes` and `No` labels:

   $$b_{vri}=\sigma(z_{vri,\mathrm{Yes}}-z_{vri,\mathrm{No}}).$$

5. Build a constrained set distribution for each valid chain, then mix them.
6. Initialize the student from the same declared checkpoint; train the no-thinking
   readout with gold-only, marginal-only, or full set-target supervision.
7. Select on dev; evaluate the frozen checkpoint and decoder once on test.

Two views times two samples means **two self-teacher policies and four reasoning
samples**, not four independent models. Shared errors can remain. Candidate yes/no
readout also does not guarantee calibration: teacher quality must be measured on
dev before investing in full generation/training.

The teacher has no gold answer or gold count in its prompt. Missing/invalid
readouts produce `q=null`, never a vector of negatives. Truncated chains are
excluded by default; including them is an explicit, recorded ablation. Each
cache records view, checkpoint identity, prompt/input digest, token IDs, generation
configuration, and chain completion status. Fresh checkpoint identities must be
used after changing local weights.

Offline teacher cost includes reasoning generation and K binary readouts per
chain. The trained student still requires one Transformer forward pass. No GPU
latency or throughput claim has been measured for this implementation.

## 4. Teacher set law: mix before discarding chain information

For one chain, define a **conditional-independence approximation**:

$$B_{vr}(S)=\prod_{i\in S}b_{vri}\prod_{i\notin S}(1-b_{vri}).$$

This product is a modeling choice, not a measured full teacher joint distribution.
With the permitted counts $\mathcal M_x$, condition each component first:

$$Q_{vr}(S)=\frac{\mathbf1[|S|\in\mathcal M_x]B_{vr}(S)}
 {\sum_{A:|A|\in\mathcal M_x}B_{vr}(A)},\qquad
 Q_T(S)=\sum_v w_v\frac1{R_v}\sum_r Q_{vr}(S).$$

The default is equal weight per view, then equal weight per valid chain within
the view. A view with more surviving chains does not receive more weight merely
because of its sample count. Weights are fixed in the manifest, nonnegative, and
normalized. This patch does not learn weights from student gradients or entropy.

Conditioning the mixture afterward would reweight views by their original mass
on the allowed event. That is a different method. Taking the product of averaged
memberships would also destroy some chain-to-chain count information.

From the **same** $Q_T$, compute:

$$q_i^T=\mathbb E_{Q_T}[\mathbf1(i\in S)],\qquad
  \rho_m^T=Q_T(|S|=m),\qquad
  \sum_iq_i^T=\sum_m m\rho_m^T.$$

The implementation uses a log-domain forward/backward Poisson-binomial dynamic
program, handles probabilities 0/1 exactly, and rejects zero-probability protocol
events. It validates the expectation identity and marginal/cardinality feasibility.
Do not reuse unconditioned $b_i$ as $q_i^T$ after imposing a count constraint.

Example: two deterministic teachers select `{A}` and `{A,B,C}` with equal weight.
The target has $q=(1,0.5,0.5)$ and $\rho=(0,0.5,0,0.5)$. A product built from
the mean $q$ instead assigns counts $(0,0.25,0.5,0.25)$, inventing count-two mass.

## 5. Student distribution and loss

The same answer-slot hidden state supplies the existing frozen-label-head scores
$s_1,...,s_K$ and a new learned linear count head with logits $c_0,...,c_{255}$.
Mask counts outside $\mathcal M_x$ and compute $\rho_\theta(m)=\mathrm{softmax}(c)_m$.
For hidden dimension $d$, the count head adds $(d+1)\times256$ trainable parameters.

Define

$$Z_m(s)=\sum_{A:|A|=m}\exp\!\left(\sum_{i\in A}s_i\right),\qquad
P_\theta(S)=\rho_\theta(|S|)\frac{\exp(\sum_{i\in S}s_i)}{Z_{|S|}(s)}.$$

The teacher-to-student cross-entropy is exactly

$$\boxed{\mathcal L_{\rm set}=
 -\sum_m\rho_m^T\log\rho_\theta(m)
 -\sum_iq_i^Ts_i
 +\sum_m\rho_m^T\log Z_m(s).}$$

Code divides this by K to match the existing per-question mean-BCE scale. Thus the
single-choice reduction is candidate CE/K, not unscaled CE. In mixed tasks this
is an explicit per-question weighting choice that must be held fixed in controls.

For a hard gold set Y, replace $q^T$ by its multi-hot vector and $\rho^T$ by
$\delta_{|Y|}$. For fixed gold weight $\beta$ use

$$q^{\rm mix}=(1-\beta)q^T+\beta y,\qquad
\rho^{\rm mix}=(1-\beta)\rho^T+\beta\delta_{|Y|}.$$

This is exactly the CE to a mixture of the teacher law and the gold delta law;
there is no unspecified extra loss. `--gold-weight` is the gold fraction, and
has no effect on `--target gold`. Targets and weights are frozen JSON constants.

The unnormalized gradients are

$$\frac{\partial\mathcal L}{\partial s_i}
 =\sum_m\rho_m^{\rm mix}P_\theta(i\in S\mid |S|=m)-q_i^{\rm mix},\qquad
\frac{\partial\mathcal L}{\partial c_m}=\rho_\theta(m)-\rho_m^{\rm mix}.$$

This differs from BCE's $\sigma(s_i)-q_i$. Fixed teacher weights and offline
sampling have no student-parameter derivative terms. An adaptive or on-policy
extension would require a new gradient analysis, rather than deleting terms.

### Information-matched objective controls

`--objective set` now supports three `--set-loss` choices. They use the same
candidate scores, independently learned count head, teacher targets, and set-MAP
decoder. Define the student's **actual** inclusion marginal

$$\mu_i=\sum_m\rho_\theta(m)P_s(i\in S\mid |S|=m).$$

| `--set-loss` | Objective, with the existing per-K normalization |
|---|---|
| `set_ce` (default) | $[-q^\top s+\sum_m t_m\log Z_m(s)+H(t,\rho_\theta)]/K$ |
| `independent_marginal` | $[\sum_i\mathrm{BCE}(q_i,\sigma(s_i))+\lambda H(t,\rho_\theta)]/K$ |
| `induced_marginal` | $[\sum_i\mathrm{BCE}(q_i,\mu_i)+\lambda H(t,\rho_\theta)]/K$ |

Here t is the coherent teacher/gold mixture count law and
`--count-weight` sets lambda (default 1). Exact `set_ce` requires lambda=1;
changing it would no longer be the stated joint cross-entropy. The moment
controls require `--count-target teacher`; `--count-target gold` remains the
distinct legacy raw-BCE plus hard-gold-count control.

The induced-marginal control is the stronger comparison. Supervising raw
sigmoid(s) and the count head separately does not guarantee that the structured
decoder's actual marginals match q. The induced loss backpropagates through both
the scores and count probabilities, using a log-domain dynamic program for
log(mu) and log(1-mu); it does not detach, clip probabilities, or take the log of
an already rounded sigmoid.

For feasible moments realizable by this student family, exact set CE and the
induced-marginal/count control can share a population optimum. A win over
marginal-only BCE can reflect extra count information; a reproducible win over
this induced control would support an optimization or inductive-bias benefit.
Neither outcome is guaranteed. In single-choice tasks induced BCE is generally
different from categorical CE (at K=2 it is twice the normalized set CE).
Give competing losses equal learning-rate selection budgets on dev, in addition
to matching model, data, examples, update counts and inference. Do not attribute
a loss-scale or step-size effect to joint modeling.

For fixed feasible targets, this objective is convex in the free score/count
logits: the score Hessian is a nonnegative weighted sum of conditional indicator
covariances, and the count Hessian is `diag(rho_theta)-rho_theta*rho_theta^T`.
This is a property of the readout objective, not a claim that optimizing the
Transformer parameters is convex. Common shifts remain unidentifiable.

### Important reductions and limits

* Single-choice protocol: $\rho(1)=1$, $Z_1=\sum_i e^{s_i}$, giving candidate CE.
* With unrestricted counts, if $\rho_\theta(m)$ is forced to
  $Z_m/\prod_i(1+e^{s_i})$, the objective reduces exactly to independent BCE.
  A **separate learned count head** prevents that algebraic cancellation.
* Averaging teacher BCE always equals BCE against the average teacher marginal.
  It carries no extra cardinality-distribution signal.
* This family preserves membership and cardinality statistics, not arbitrary
  pairwise correlations. The mixtures `{AB,CD}` and `{AC,BD}` with equal weights
  have identical q and rho and are indistinguishable under this loss.
* Uniformly shifting all scores leaves the structured distribution unchanged.
  The implementation centers scores to improve numerical stability.

The last statement assumes the independent count law is held fixed. A future PB
mixture would depend on the absolute score shift and must restore that shift when
computing count probabilities. The free-logit convexity statement above is also
specific to the independent-head objective; it does not transfer to a mixture.

### Information retained, and a corrected loss-gap statement

For an arbitrary teacher Q and an independent BCE student with probabilities b,

$$L_{\mathrm{BCE}}(q;b)-L_{\mathrm{set}}(Q;P)
=I(Q)+\sum_iD_{\mathrm{KL}}(\operatorname{Ber}(q_i)\|\operatorname{Ber}(b_i))
-D_{\mathrm{KL}}(Q\|P),$$

where $I(Q)=\sum_i h(q_i)-H(Q)$ is total correlation. Even when b=q, matching
first moments does not remove the residual joint KL. The set objective reads
only q and rho_T and therefore cannot transfer every teacher co-occurrence rule.

Under per-example free optimization of the independent-head family, feasible
targets, and any needed boundary limits, the optimum P* is the maximum-entropy
joint matching q and rho_T. Consequently the optimal summed-BCE minus set-CE
gap is I(P*), not generally I(Q). This characterization does not guarantee a
shared Transformer reaches that optimum, and it is not an accuracy theorem.
See the linked review corrections for a four-option counterexample and proof.

For a proposed PB-based count law, alpha=0 gives ordinary BCE only when all
counts 0,...,K are permitted. Conditioning on nonempty sets gives conditional
BCE; conditioning on count one gives categorical CE. Preserve the same K scaling
when comparing these endpoints. Do not combine unrelated q and rho targets:
feasibility is required for the expression to remain a teacher joint CE.

CE is equivalent to forward $D_{\rm KL}(Q_T\|P_\theta)$ up to teacher entropy.
No additional reference-model KL is mathematically required. A base-model anchor
would be a separately motivated continual-learning ablation, not a prerequisite.

## 6. Exact inference without enumerating 2^K sets

The log partition recurrence is

$$\log Z_m^{(j)}=\mathrm{logaddexp}
 (\log Z_m^{(j-1)},s_j+\log Z_{m-1}^{(j-1)}).$$

Normalization and all membership marginals use O(K²) dynamic programming. The
autograd implementation avoids undefined gradients at unreachable -infinity
states. Final membership probabilities are

$$p_i=\sum_m\rho_\theta(m)
 \frac{e^{s_i}Z_{m-1}(s_{-i})}{Z_m(s)};$$

they are generally **not** sigmoid(s_i). MAP finds the best top-m set over permitted
counts using $\log\rho_\theta(m)+\sum_{i\in\mathrm{Top}_m(s)}s_i-\log Z_m(s)$.
Selecting argmax count followed by top-m is not generally set MAP. Ties have fixed
index/count ordering, never gold-based tie breaking.

Structured exports include `decision_readout.json` and `decision_count_head.pt`.
The evaluator requires both, validates tokenizer label IDs/dimensions, and refuses
to silently evaluate a structured checkpoint with independent sigmoid scores.

## 7. Commands and bounded first experiment

Dependencies for model work are the same native Qwen3.5-capable PyTorch,
Transformers, vLLM and safetensors environment used by the original harness.
Data conversion also requires pyarrow. Pure target building and saved-prediction
evaluation need only Python's standard library. This change does not pin a new
GPU dependency stack without testing it.

Rebuild without overwriting the archival data (raw filenames are checked by the builder):

```bash
python -I harness/multiselect_to_unified.py \
  --raw-dir /path/to/raw_multiselect --out-dir data/multiselect_grouped_mixed \
  --cmb-question-mode mixed --seed 42
```

Run the original gold BCE recipe against these new train/dev files, using a new
output directory, and freeze the selected checkpoint. Then generate a **small,
explicit train/dev pilot** in two views. Use a fresh checkpoint ID and run root:

```bash
MODEL=/path/to/frozen-gold-checkpoint CHECKPOINT_ID=gold-grouped-seed43 \
TEACHER_ID=direct TEACHER_VIEW=direct SAMPLES=2 SHARDS=1 LIMIT=200 \
SPLITS="train dev" DATA_DIR=data/multiselect_grouped_mixed \
RUN_ROOT=runs/teachers_pilot MAX_TOKENS=4096 MAX_MODEL_LEN=16384 \
READOUT_MAX_INPUT_TOKENS=16384 Q_TRANSFORM=membership TRUNCATED_POLICY=exclude \
bash harness/run_sii_style_teacher.sh

MODEL=/path/to/frozen-gold-checkpoint CHECKPOINT_ID=gold-grouped-seed43 \
TEACHER_ID=verify TEACHER_VIEW=verify SAMPLES=2 SHARDS=1 LIMIT=200 \
SPLITS="train dev" DATA_DIR=data/multiselect_grouped_mixed \
RUN_ROOT=runs/teachers_pilot MAX_TOKENS=4096 MAX_MODEL_LEN=16384 \
READOUT_MAX_INPUT_TOKENS=16384 Q_TRANSFORM=membership TRUNCATED_POLICY=exclude \
bash harness/run_sii_style_teacher.sh
```

`LIMIT` is a plumbing pilot, not the final stratified scientific sample. Compare
source/mode/K/count groups before selecting a full experiment slice. Inspect
coverage and completion first; do not treat a forced `</think>` as completed
reasoning. Increase the finite reasoning budget or regenerate failed items if
the exclusion rate is high. Neither above command has been run on a GPU here.

Example train manifest, saved beside `runs/teachers_pilot/teachers_train.json`:

```json
{
  "teachers": [
    {"id": "direct", "path": "direct/q/membership/exclude/merged/sii_teacher_train.jsonl", "weight": 1.0},
    {"id": "verify", "path": "verify/q/membership/exclude/merged/sii_teacher_train.jsonl", "weight": 1.0}
  ]
}
```

Build targets against the **same pilot input subset** (not the full train file
when only 200 rows were generated). Full runs must have full teacher coverage.
Missing teachers stop the build by default; `--missing drop` explicitly creates
a smaller documented subset and must be applied identically to all comparisons.

```bash
python -I harness/build_multiteacher_targets.py \
  --input /path/to/exact_pilot_train.jsonl \
  --manifest runs/teachers_pilot/teachers_train.json \
  --output runs/targets/pilot_train.jsonl
```

For a full covered target file, the new training path is:

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 8 harness/train_multiselect.py \
  --train data/multiselect_grouped_mixed/train.jsonl \
  --dev data/multiselect_grouped_mixed/dev.jsonl \
  --model /path/to/frozen-gold-checkpoint \
  --modules innerjev/configs/4B/modules.json \
  --objective set --target file --target-file runs/targets/full_train.jsonl \
  --gold-weight 0.5 --count-target teacher \
  --output runs/set_distill_seed43 --seed 43 \
  --lr 2e-6 --global-batch 64 --epochs 1 --lambda-g 0 --graph none

python -I harness/eval_multiselect.py \
  --model runs/set_distill_seed43/final \
  --input data/multiselect_grouped_mixed/test.jsonl \
  --output runs/eval/set_distill_seed43_test
```

`0.5` is an initial gold/teacher ablation setting, not an optimized value. Choose
among preregistered values using dev only. The student starts from the same
checkpoint in matched second-stage controls; those controls also receive the
same number of additional updates.

Minimum informative comparisons:

| Arm | Flags / training signal | Confound resolved |
|---|---|---|
| Gold BCE | `--objective bce --target gold` | Existing method reference, with matched additional training budget |
| Gold set model | `--objective set --target gold` | Count head, structured likelihood and decoder alone |
| Mean membership KD | `--objective bce --target file` | Ordinary multi-teacher BCE |
| Same-head marginal-only KD | `--objective set --count-target gold --target file` | Same architecture/decoder; BCE on mixed membership plus gold count CE |
| Independent marginal + soft count KD | `--objective set --set-loss independent_marginal --target file` | Same teacher q,t and count head, auxiliary sigmoid marginals |
| Induced marginal + soft count KD | `--objective set --set-loss induced_marginal --target file` | Same q,t, head and decoder; supervises the actual set-model marginals |
| Full set KD | `--objective set --set-loss set_ce --count-target teacher --target file` | Exact coherent set likelihood; compare against both matched-information controls |
| Single view × 4 chains | Full set KD with one manifest teacher | Same total sampling budget as two views × 2 chains |

The marginal-only control has a separately stated objective, BCE(q_mix,s) plus
gold count CE/K. It does not combine incompatible soft q and hard rho and claim
they are the moments of one teacher law.

Report overall and per-source/per-mode exact-set accuracy, micro-F1, membership
Brier/BCE, cardinality NLL/Brier/mode accuracy, count error, and teacher coverage.
Structured exports also report exact gold-set NLL and the Brier score for the
binary event that the model's predicted complete set is correct. The latter is
not the Brier score of the complete distribution over all subsets. Stable count
log probabilities preserve very large NLLs instead of hiding them behind a
probability floor; legacy prediction files explicitly report a clipped origin.
The evaluator separately labels gold-count top-k as an oracle ranking diagnostic.
No oracle count is supplied to the model or operational decoder. Independent-BCE
thresholds may be chosen on dev and frozen before test; set MAP has no threshold.

**Falsifier:** if teacher count distributions are poor on dev, or full set KD
does not reproducibly improve over matched marginal KD / gold set controls,
do not claim useful extra teacher count information. Repeat promising effects
across seeds and use paired question/group-level uncertainty for comparisons.

The separate [teacher-generation exploration](decision_teacher_generations.md)
tests whether a stronger direct student remains a useful next-generation
reasoning teacher. It has planned fixed/refreshed trajectories, matched-init
teaching probes and result audits; it is not evidence that recursive degradation
already occurs in this implementation.

## 8. Verification and remaining work

The standard-library tests compare DP probabilities, marginals, MAP, and loss
gradients to exact enumeration or finite differences on small problems. They
also cover teacher identity, option order, missing/truncated chains, grouped
splitting, and evaluation without gold-count leakage. Run:

```bash
python -m unittest discover -s tests -v
python -I harness/eval_multiselect.py --help
python -I harness/build_multiteacher_targets.py --help
```

PyTorch-specific tests exercise autograd, mixed-protocol batches, extreme logits,
and count-head save/load. They explicitly skip if PyTorch is absent. Passing
scalar tests is not a substitute for these tests, native Qwen forward checks,
FSDP export/reload, or actual teacher/student GPU runs. The development host for
this patch has no PyTorch/model weights/GPU, so it did not execute those native checks.
An external follow-up review reports passing all 85 tests on a PyTorch host and
real tokenizer validation; it was not rerun on this development host. See the
[verification record](verification_2026-10-22.md) for attribution and scope.
The original `--resume` path is weight/step continuation with a fresh optimizer,
not an exact optimizer-state resume; this patch does not change that behavior.
