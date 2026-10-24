# Recurrent SetJev: Decision-Native Latent Reasoning (v0 implementation)

**Branch:** research/recurrent-set-decision-20261024  
**Upstream base:** codex/multiteacher-set-distillation at 98d2cc6  
**Paper reviewed:** Engdahl et al., [BDH-CQ: In-Context Learning with Recurrent Latent Reasoning](https://arxiv.org/abs/2608.09888), arXiv:2608.09888v1, 10 Aug 2026.

## 0. Scope and attribution (read before running)

This branch implements an **original, BDH-CQ-inspired recurrent set decision
pilot**, NOT an exact implementation or reproduction of Pathway BDH-CQ. The
paper explicitly withholds the dimensions, exact memory / latent update rules
(section 3.3), and the internal training recipe (section 4.1). Its published
29.5% ARC-AGI-1 pass@2 is **not reproducible from the disclosed implementation
alone**. The [pathwaycom/bdh](https://github.com/pathwaycom/bdh) repository is
an earlier BDH architecture with a byte-level language-model toy trainer; the
[arc-task-gen](https://github.com/pathwaycom/arc-task-gen) repository generates
new ARC-style tasks, not solver checkpoints.

The two published BDH-CQ interfaces **are** useful and distinguishable:

1. Demonstration-conditional **memory**:
   S_t = U(S_(t-1), D_t). Updated when new evidence is ingested.
2. Query-time **latent workspace**:
   H_(r+1) = F(H_r, S_K). Updated without emitting natural-language CoT.

BDH-CQ evaluates an ARC pipeline including candidate construction and ranking;
a set classifier is not equivalent to that pipeline. We separately implement
**sequential action execution** for a toy environment. Static multi-select
classification, intra-query latent recurrence, and environment time steps
must never be conflated.

## 1. What is implemented

Files:

- harness/recurrent_set_decision.py: small shared-step recurrent head, optional
  sequential demo memory, exact set probability and MAP decode.
- harness/run_recurrent_set.py: offline Qwen answer-slot feature extraction,
  frozen-feature supervised training, depth sweep evaluation, CPU smoke.
- harness/toy_sequential_set.py: supervised synthetic task, online environment
  transitions conditioned on the policy's actual selected actions.
- tests/test_recurrent_set_decision.py: DP normalization, exhaustive enumeration,
  gradients, mask invariance, memory response, ablation sanity.
- qz/run_recurrent_set_pilot.sh: QZ/local shell entrypoint.

No existing BCE/SetJev trainer or checkpoint is overwritten. Qwen token
generation is never called during latent recurrence. It remains legal to use
a pretrained Qwen as a **frozen encoder**. The exported gold-BCE checkpoint
continues to provide the original native answer-slot logits.

### 1.1 Exact probability law

For K candidates and permitted cardinalities M(x):

    P_r(S|x) = exp(sum_{i in S} s_i^(r) + a_{|S|}^(r)) / Z_r,
    Z_r = sum_{m in M(x)} exp(a_m^(r)) Z_m(s^(r)),
    Z_m(s) = sum_{|T|=m} exp(sum_{j in T} s_j).

The dynamic-programming partition is O(K^2). We do NOT enumerate 2^K sets.
The exact set-MAP prediction chooses the top m logits for each m, then selects
the m maximizing their sum plus a_m (NOT simply the most probable count).

At r=0, a_m=0, and s_i are the pretrained native letter logits. This exactly
matches *independent Bernoulli probabilities conditional on the permitted
cardinalities*. When nonempty answers are mandatory, this is not exactly
the original independent-BCE threshold decoder; use the same decoder for
matched comparisons.

### 1.2 Recurrent update (our design, NOT published BDH-CQ equation)

    q = ContextProj(h_Qwen(x))
    memory_t = GRU(DemoProj(demo_t), memory_(t-1))
    z_0 = Init(q, memory_K)

    c_i^r = CandidateProjection(s_i^r, sigmoid(s_i^r),
                                variance_i^r, s_i^r - s_i^0, position_i)
    summary_r = AttentionPool(c_1^r, ..., c_K^r; query=z_r)
    z_(r+1) = GRU(q + memory_K + summary_r, z_r)
    s_i^(r+1) = s_i^r + DeltaScore(c_i^r, z_(r+1))
    a_m^(r+1) = CountCorrection(z_(r+1))

The score-delta/count-correction outputs are zero-initialized, so the first
untrained step is no worse than the frozen native-set distribution. In later
training, candidate belief feedback can couple options across repeated steps.
No gold label or true gold cardinality is passed to the forward / MAP methods.

One mathematical caveat: this **cardinality-potential family cannot represent
arbitrary pairwise interactions**. The recurrence changes its parameters, but
does not by itself make the final family a general distribution over 2^K.
Pairwise energies are a candidate v1 extension only if diagnostics justify them.

### 1.3 Three feedback controls

- set: candidates read updated scores and belief at every recurrent step.
- latent_only: latent workspace recurs, but candidate input is always initial
  native logits; separates latent-depth benefit from belief feedback.
- static: re-evaluates the SAME initial state and native logits each step; each
  step adds the same score delta. This is the score-amplification negative control.
  A recurrent vs static win is necessary but not sufficient to claim reasoning.

Use identical model widths, heads, data, optimizer updates, training seeds,
and evaluation sets. Sweep inference depth on the same selected checkpoint.
When comparing 0 vs 4 steps, note that FLOPs differ by design.

## 2. First run: offline, no GPU

From the repository root (PyTorch must be installed):

    python -m unittest discover -s tests -p test_recurrent_set_decision.py -v
    python harness/run_recurrent_set.py smoke
    python harness/toy_sequential_set.py --train-episodes 512 \
       --eval-episodes 256 --horizon 3 --epochs 10 --steps 4

The toy task chooses all actions specified by a demonstration-conditioned
cyclic rule. The model's selected set becomes the next actual environment
state. Report both per-step exact set EM and full-trajectory exact success.
It is **not** an ARC-AGI experiment, and oracle-trajectory supervised training
is not RL, not RLCD².

## 3. Real Qwen3.5-4B experiment with existing SetJev checkpoints

Use the exported **gold-BCE** Qwen checkpoint, e.g.:

    MODEL=runs/qz/grouped_bce_4gpu_s44/v1/final

Confirm the path includes a complete Hugging Face model and tokenizer. Feature
extraction must use the SAME checkpoint for train and dev; we record an explicit
model identity but currently **do not hash all large checkpoint weight files**.

    python harness/run_recurrent_set.py cache \
        --dataset data/multiselect_grouped_multi_5k/train.jsonl \
        --model "$MODEL" --model-identity grouped_bce_s44_final \
        --output runs/recurrent_cache/train5k.pt \
        --device cuda --dtype bf16 --batch-size 1

    python harness/run_recurrent_set.py cache \
        --dataset data/multiselect_grouped_multi/dev.jsonl \
        --model "$MODEL" --model-identity grouped_bce_s44_final \
        --output runs/recurrent_cache/dev2042.pt \
        --device cuda --dtype bf16 --batch-size 1

    python harness/run_recurrent_set.py train \
        --train-cache runs/recurrent_cache/train5k.pt \
        --dev-cache runs/recurrent_cache/dev2042.pt \
        --output runs/recurrent/set_s43 --feedback set \
        --steps 4 --epochs 5 --latent-dim 128 --seed 43 \
        --device cuda

    python harness/run_recurrent_set.py evaluate \
        --cache runs/recurrent_cache/dev2042.pt \
        --checkpoint runs/recurrent/set_s43/best_head.pt \
        --depths 0,1,2,4,8 --device cuda \
        --output runs/recurrent/set_s43/evaluation.json

Feature extraction runs the Qwen backbone **once per input**, caching answer-slot
vectors and native logits. All head training is thereafter from these fixed
features, not Qwen full-rank fine-tuning. Evaluate all 2042 held-out dev questions,
not the 1000-row teacher subset. The selected checkpoint is the best *dev* EM:
reporting the selected dev metric is diagnostic, **not an untouched test score**.
Save official test once the protocol is frozen.

### 3.1 Matched experiments

Train at least seeds 43/44/45 for feedback in {set, latent_only, static},
each at R=4; evaluate the same selected checkpoint at R in {0,1,2,4,8}.
Report exact-set EM (primary), candidate F1, count accuracy/MAE, exact set NLL,
runtime and memory. Also run a **gold continuation baseline** from an equivalent
checkpoint / data / budget before claiming improvement over BCE training.

A positive recurrence hypothesis requires (a) improvement at held-out depth 4
over zero-depth native readout, (b) gain over static repeated-score control,
(c) ideally improvement over latent_only to show belief feedback mattered,
and (d) generalization to a different task family or dynamic environment.
Synthetic environment success alone is not enough.

### 3.2 Extension to ARC is a separate project stage

ARC outputs arbitrary grids, not multi-answer labels. To test this as ARC
reasoning, first define a legal action/operation vocabulary, a state transition
executor, and an unambiguous construction stop action, then report solution
coverage restrictions. Do NOT relabel multi-select EM as ARC pass@2 or use
generated evaluation tasks in training.

## 4. Reproducibility caveats and planned work

- The Qwen weights are frozen in this pilot. Replacing the existing
  gold-BCE training code or claiming end-to-end 4B optimization is incorrect.
- Demonstration memory is a real optional module and exercised by the toy
  benchmark. The static CMB/MultiRC feature-cache pipeline does NOT expose
  sequential demonstrations and thus does NOT train demonstration memory.
- Teacher generation, model-produced data, hard-to-soft KD, active environment
  interactions, and StartLux's RLCD² are **not yet implemented**. StartLux's
  full training release and BDH-CQ's internal recipe cannot be assumed open.
- These files are source-level prototypes. Qwen loading and numerical behavior
  must be checked on the target CUDA environment before any accuracy claims.
- The training objective is gold exact-set NLL, optionally every step:

      L = mean_r -log P_r(S_gold|x)/K.

  This is a supervised, differentiable pilot. Introduce sequence-level reward
  and policy optimization only after the online environment is defined.
- Version 1 could add a pairwise residual b_ij and training with a
  supervised belief-update advantage, *provided* matched controls show a
  real representational or optimization gap.

## References

1. Engdahl et al., BDH-CQ, arXiv:2608.09888 (2026), sections 3.2–4.2.
2. [Pathway BDH implementation](https://github.com/pathwaycom/bdh).
3. [Pathway ARC Task Generator](https://github.com/pathwaycom/arc-task-gen).
4. [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).
5. Existing SetJev v2 results in docs/setjev_v2_gate3_warm_results.md.
