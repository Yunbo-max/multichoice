# Paper 1 — Three-regime causal research design (2026-10-30)

**Status: RESEARCH DESIGN + CONTROLLED-DATA GENERATOR + RESTRICTED EXACT-DP REFERENCE.**
No new Qwen GPU training/inference, no held-out paper-scale evidence, and no
arXiv-ready superiority claim is implied. Previous independent 4B TEST
(2,140 questions, 3 seeds) showed Count Anchor Set CE ~61.73% versus
BCE Continue ~61.82% EM. 24 warm-start arms from a shared BCE S44 model
were already completed; they do not prove a BCE win.

## Scientific claim and falsifiability

**Question:** Which structure is needed for reliable set-valued decision
models: independent Bernoulli outputs, cardinality potentials, or pairwise
dependencies? Separate (i) **representational necessity** from
(ii) **learnability/optimization** and (iii) **real-world generalization**.

For visible input x, binary choice vector Y, public permissible
counts M(x), candidate scores s(x), count residual a(x), and pair weights
J(x):

    P(Y|x) = exp(sum_i s_i Y_i + a_|Y|(x)
                  + sum_(i,j) in E(x) J_ij(x) Y_i Y_j)
             / sum_{A: |A| in M(x)} exp(E(A,x)).

Nested specializations:
- **I / BCE**: a=J=0, all counts 0..K permitted => exact product Bernoulli.
  If empty forbidden, this becomes BCE **conditioned** on legal counts.
- **C / Count Anchor**: J=0, flexible a_m. Relative to a frozen BCE
  candidate-score law, log P*/P_BCE is constant *within each cardinality*.
  It cannot express arbitrary pair co-occurrence.
- **R / Pairwise**: add J. Different equally sized sets may have genuinely
  different nonadditive interactions. This family includes C and I.
  A matching-restricted implementation is NOT a universal dense CRF.

### The main algebraic hypothesis for A

For a ground-truth count-only teacher
    Q_A(Y|x) proportional to exp(sum_i s_i Y_i + a*_|Y|),
its within-count conditional Q_A(Y ||Y|=m,x) equals that of the
BCE distribution from the *same frozen s*. The best correction can match
the true count law rho*, and its conditional expected full-set NLL gain is

    KL(rho*(.|x) || rho_BCE(.|x))

assuming the teacher shares the same within-m distribution and the
cardinality head has sufficient capacity. This is an identity about an
oracle probability law, not a guarantee of finite-sample EM or learned-head
benefits, and it is not a new CRF result.

### The main algebraic hypothesis for B

For |Y|=|Y'|, Count Anchor gives
    log P(Y|x) / P(Y'|x) = sum_i s_i (Y_i-Y'_i).
The a_m term cancels. Pairwise adds
    sum_(i,j) J_ij (Y_iY_j - Y'_iY'_j).

A particularly strong counterexample is a sequence of public matchings
on 8 candidate options. Give ALL option scores s_i=0, equal J>0 on
each of four disjoint pair edges, and a common soft count preference
a_m=-0.9(m-2)^2. Rewire matching pairs between inputs. Symmetry makes
q_i and rho_m invariant under rewiring, even though pair co-occurrence
changes. No independent or count-only distribution can match all the
different conditional joint distributions; a pairwise model with the
correct learned edges can. Compare against a rule parser + exact matching
decoder, not just weak BCE thresholds.

**Critical label identifiability:** a single hard set for one unique x
cannot identify a true multimodal joint law. Sample multiple independent
Y realizations per input context in the controlled benchmark, and retain
the exact teacher joint distribution separately for oracle evaluation.
Group by original context to prevent train/dev/test leakage. For real
tasks, require a reliable verifier or multiple valid sets. In a 50/50
AB/CD world the maximum deterministic Exact-Set EM is 50%, not 100%;
report proper joint NLL and mass on valid sets as primary metrics.

## Evidence sequence and decision gates

### E0 — Existing 4B checkpoint integrity and retrospective diagnosis

Use the six **already staged** QZ DEV-only jobs
`qz_static_count_dev/paper1-dev-{bce_continue,set_ce_anchor}-s{43,44,45}.json`.
They have not been submitted. Must execute on actual remote H100 installation
before analysis. They export raw option scores, exact structured count logs,
full input-dependent count residual logits and checkpoint provenance.

Then run:
    python -m paper_static.paired_count_audit \
      --gold data/multiselect_grouped_multi/dev.jsonl \
      --predictions-root runs/paper_static_dev_eval/qwen3.5-4b-bce-s44 \
      --seeds 43 44 45 --bootstrap 1000 \
      --output runs/paper1_count_audit/paired_count_dev.json \
      --per-question runs/paper1_count_audit/paired_count_dev.jsonl

Pre-registered exploratory readout: BCE per-item count NLL quartiles Q1--Q4;
paired EM and count NLL; candidate-ranking separability; zero-residual
a(x)=0 ablation with SAME anchor option logits; count head actual weight norms,
bias norms and gradients (weights must not be inferred from bias alone).
Paired passage/group bootstrap per seed; do not pool 3 seeds as independent
question-level samples. Q4 high BCE count NLL is **label-dependent**:
never market it as deployable gating or infer a causal benefit from it.

If all Count Head weights are unchanged/zero, fix optimizer/FSDP export/load
and rerun affected training BEFORE interpreting the negative count results.

### E1 — Pure probability-family identification (CPU and full oracle)

Use code:
    python -m paper_static.controlled_regimes --output-dir runs/regimes_v1 \
      --seed 20261030 --k 8 --ntrain 240 --ndev 60 --ntest 60 --repeats 4
    python -m paper_static.block_pair_dp
    python -m paper_static.regime_oracle_check
    python -m unittest discover -s tests -p 'test_paper1_controlled_regimes.py' -v

Generator supplies three regimes with all counts 0..K LEGAL (avoids unfair
empty-set truncation in reduction tests):
- **I:** independent Bernoulli; visible per-candidate utility cues; J=0,a=0.
- **C:** same additive score family, dynamic count residual determined by
  visible workload and collective-pressure cues; strength in {0,0.5,1,1.8} (zero is an essential negative control).
  Count support is not leaked from gold. Every workload strength, including zero, is an observable *public cue*, not an oracle label.
- **R:** candidate groups sharing a visible compatibility code, with
  changing perfect matching across contexts, J in {0,1.5,3,4.5},
  same cardinality potential. All s_i=0, hence q/rho invariant under
  matching rewiring at fixed strength; only pair co-occurrences change.

**Files and leakage:** `runs/regimes_v1/student/{train,dev,test}.jsonl`
contains prompts and stochastic sampled labels. Teacher scores, residuals,
pair coefficients and true probabilities exist ONLY in
`runs/regimes_v1/oracle_DO_NOT_USE_AS_MODEL_INPUT/`. They are
for oracle scoring/audit, never training features. Split by entire
context BEFORE repeated draws. For R also split perfect matching
**structures** by deterministic SHA; prevent graph topology leakage.
Public matching tags are an intentionally easy symbolic cue:
a rule parser/solver MUST be a strong baseline.

For each context, estimate from repeated labels and compare against
full known teacher law; do not call realized label count "a hidden rule".
Oracle evaluations measure joint KL, conditional count KL,
pair co-occurrence covariance, proper set NLL, coverage and validity.
Test independent to cardinality strength sweep gamma in {0,.5,1,1.8}
and pair coupling in {0,1.5,3,4.5}. Use K=6/8/10 initially.
A hard exactly-two diagnostic may be used ONLY if the same public support
is passed to all models.

## Executable preparation and CPU/4B limits

Generate the fixed controlled regimes and audit the joint-law ceilings
**before any GPU job**:

~~~bash
python -m paper_static.controlled_regimes --output-dir runs/regimes_v1 \
  --seed 20261030 --k 8 --ntrain 240 --ndev 60 --ntest 60 --repeats 4
python -m paper_static.regime_oracle_check
python -m paper_static.block_pair_dp
python -m unittest discover -s tests -p 'test_paper1_*' -v
~~~

For a meaningful 4B LoRA fit rather than the small CPU smoke, generate a
separate, predeclared dataset directory with
`--ntrain 2400 --ndev 500 --ntest 500 --repeats 4`. The
**same** train/dev files and CPU/GPU budgets must be used for every arm.

Create 13 *plan-only* command arrays (one BCE warm prerequisite + 4 methods
times 3 seeds), no QZ submission and no fabricated checkpoints:

~~~bash
python -m paper_static.plan_regime_pilots \
  --data-dir runs/regimes_v1 \
  --model /path/to/Qwen3.5-4B \
  --output runs/paper1_plan/controlled_4b.json \
  --seeds 43 44 45 --warm-steps 450 --continuation-steps 100
~~~

Review available GPU memory and tune LoRA LR using the **same DEV budget**
for every arm. A single suggested LR in a plan is not evidence of a
fair hyperparameter search. Run BCE warm start, inspect/verify its saved
full checkpoint, and only then submit matched continuations. Qwen Pairwise
**cannot** be added to the plan until a trained edge detector and
pairwise loss have been integrated and audited.

Once a compatible model has emitted predictions for every repeated draw,
evaluate true expected joint KL from the separately sealed synthetic
teacher moments, with option scores and *raw* count/pair residuals:

~~~bash
python -m paper_static.regime_eval \
  --data-root runs/regimes_v1 \
  --predictions runs/<arm>/dev_predictions.jsonl \
  --mode bce --split dev \
  --output runs/<arm>/regime_dev_metrics.json
# Mode can be bce, anchor, or pair. Pair mode requires documented
# edge_source=learned/public_rule_parser/oracle_diagnostic, NEVER unlabeled.
~~~

Use the oracle benchmark solely as a diagnostic label provider, not an
input to the student or a way to select answer sets at inference. The
`regime_eval` scorer will flag oracle-diagnostic pair identities rather
than letting them appear as a learned-model result. Evaluate synthetically
generated TEST only after method/threshold selection is locked.

### E2 — Same-model causal training matrix (Qwen3.5 4B)

One base checkpoint, option readout, prompt format, candidate permutations,
source groups, token/update budget, and training data per seed. Use 3 matched
seeds as pilot, then 5 if positive and affordable. First smoke on 0.8B/2B
or 4B LoRA on 1 H100; full-rank Qwen4B only after readout correctness.
The existing S44 warm start should only be used for continuation controls;
a clean initialization study must start ALL arms from the same base.

**Minimal six baselines (must not omit strong alternatives)**:
1. BCE + fixed 0.5; same-budget dev-tuned global threshold separately.
2. BCE + learned count head + Top-k; allocate matching extra params/updates.
3. BCE + public hard-count conditioning when explicitly allowed.
4. Independent-count Set CE and Count Anchor Set CE; detach count-gradient.
5. Block Pairwise Set CE with known public pairing (ORACLE EDGE ABLATION ONLY)
   versus learned content-dependent pairing; zero-J and random-edge controls.
6. Rule-parser / graph-constraint solver and BCE + structured reranker,
   for relational tasks with publicly observable compatibility constraints.

**Implementation boundary:** controlled data generation, a mathematically exact
small-K oracle and matching-block logZ/MAP are implemented. The learned
content-dependent matching head, full Qwen Pairwise training path, gradient
routing through edge selection and end-to-end 255-option VLM/LLM inference
are NOT implemented; completing them is E2/E4 engineering, not a result.

For fair inference comparisons, define **one common legal-set protocol**,
include a BCE MAP under it, and separately show threshold policy metrics.
Use the same number of student Transformer forwards. Count pairwise edge
prediction, DP, optimizer steps and GPU-hours; distinguish computational
efficiency from accuracy, and report loss-normalization/LR sweeps.

**Statistical design:** measure on context-disjoint DEV, freeze threshold/
model-selection rules before one independently generated TEST. Use
bootstrap of CONTEXT units (and passage units for real MultiRC), not draws
or candidate labels as IID observations. Report 3/5 seed paired means,
per-seed deltas, confidence intervals, regression by true effect strength
(in controlled data only), and compute cost. One-size-fits-all mean EM
alone is inadequate when multiple sets are legitimate.

### E3 — Held-out real tasks (separate from synthetic proof)

- **MultiRC/CMB**: existing old control only, not proof of conditional
  independence. Include exact set, F1, count NLL and ranking by gold count.
  Old 2,140-question TEST is already opened; do not tune a new loss on it.
- **MetaTool multi-tool**: exact multiselect and zero-action protocol.
- **ECtHR / LexGLUE**: potentially empty/multiple violated articles, source-
  separated proper multi-label evaluation (verify legal label mapping).
- **HotpotQA**: complementary evidence passages, with Top-2 as a powerful
  public-count baseline where the benchmark reliably contains two supports.
- **SATA-BENCH** and **EURLEX**: independent external confirmation if
  label licensing, candidate retrieval and train/test identities are audited.

A real dataset with one Gold Set per question does not itself validate
pairwise dependence. Verify alternative valid sets using a solver/annotators,
or explicitly narrow the real-world claim to observed set EM and count bias.

### E4 — Structured scaling to 255 options

Reference implementation: `paper_static/block_pair_dp.py`.
If each option belongs to **at most one pair**, the local block polynomial is

    F_(ij)(t) = 1 + (exp(s_i)+exp(s_j)) t
                    + exp(s_i+s_j+J_ij) t^2.
    F_i(t) = 1 + exp(s_i) t for singleton blocks.
    Z_m = [t^m] product_blocks F_b(t).
    P(Y|x) = exp(E(Y)) / sum_m exp(a_m) Z_m.

The exact log-domain polynomial count DP and max-sum MAP are O(K^2)
for disjoint pairs, even at K=255 (plus Transformer readout).
A differentiable PyTorch logZ reference is implemented for score,
residual and edge-weight gradients. NO learned matching policy
is provided yet; oracle/public edges are an explicit ablation.
Learning a matching from candidate content and calibrating its errors
is a separate task, with possible O(K^3) matching assignment costs.
Do not claim the result solves arbitrary sparse graphs, dense pairwise
interactions, 3-way dependencies or 255-option Qwen fine-tuning.
Extend to bounded blocks (size <=3/4) or bounded treewidth only if
required, with explicit complexity and oracle tests.

Regressions must test:
1. J=a=0 => unrestricted BCE EXACTLY; with legal nonempty support =>
   conditional BCE, not unrestricted BCE.
2. J=0 => Count Anchor EXACTLY.
3. Pairwise DP partition, MAP, NLL and autodiff equal exhaustive
   enumeration at K<=8/12; numerical stability at huge positive logits.
4. K=255 O(K^2) memory/time measurement (include model tokenization and
   matching inference as separate costs).
5. Option permutation equivariance, unseen pairing topology, varying K
   and edge perturbation/noisy relation descriptions.

## Primary metrics and paper rejection rules

| Subquestion | Primary metric | Required positive evidence |
|---|---|---|
| I independent | joint NLL + EM | Anchor/Pairwise should *not* systematically win; near BCE |
| C count effect | count NLL, joint NLL | Anchor improves over BCE **and BCE+learned-count** on independent holdout; if not, limit to optimization stability |
| R within-count pairing | joint NLL, invalid-set probability, valid-set mass | Learned-pair model improves over Anchor **and parser+solver**, not only oracle-edge toy |
| Cross-regime | regret vs per-regime oracle, calibration | Single nested system matches BCE in I and improves in C/R without manual labels selecting submodels |
| Scaling K | wall clock, VRAM, throughput | Correct 255-option inference with measured end-to-end costs, not just exact DP correctness |

**Fallback publication outcomes:**
- If C matches BCE+learned-count, Count Anchor is an initialization/
  gradient geometry result, not cardinality expressivity novelty.
- If R wins only with oracle-known pair identities but learned pair
  detector fails on unseen matching structures, it is a mathematical
  control, not a general decision model.
- If parsing explicit compatibility constraints + a conventional solver
  is as good, report that strong negative result rather than claiming
  a new relational architecture advantage.
- If only simulated oracle KL improves, no general empirical claim
  is made about 4B/255; further real transfer evidence is required.

**arXiv gate:** do not post a positive superiority paper before E0
checkpoint validation, E1 proof/control, at least one E2 independently
held-out condition with meaningful paired gains and strong baselines,
plus clear ablations and compute reporting. A careful negative/boundary
study is still possible with a suitably limited abstract.
