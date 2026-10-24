# SetJev v2 plan: from "harder set loss" to "useful, calibrated teacher supervision"

Date: 2026-10-24. Status: plan only, nothing below is implemented or validated yet.

## 1. What the latest results showed (negative signal)

On the 5k subset (`data/multiselect_grouped_multi_5k`, train 5,000 / dev 1,000),
single seed 43, same 4xH100 recipe:

| Arm | Question EM |
|---|---:|
| Gold BCE | 50.15% |
| Gold Set CE (same data) | 39.37% |
| Best soft-teacher set model | 24.39% |

Soft-teacher-trained students collapse to over-selection: recall saturates at
~97%–99.6% while precision drops to ~64%–70%. The student marks too many
candidates as correct.

This does **not** disprove self-distillation for decision models. It first shows
that the current soft-teacher supervision is mismatched with the real multi-answer
task.

## 2. Three root causes

### A. The current run is soft-only distillation, not hard-then-soft

The latest training config shows:

- the teacher comes from the **full Gold-BCE model**;
- the 5k soft-distillation student restarts from the **Qwen3.5-4B base**;
- `gold_weight = 0`;
- `planned_steps = 79`.

So the current experiment is "same-architecture supervised teacher -> fresh student,
pure soft distillation", i.e. $\theta_{\mathrm{hard}} \to \theta_{\mathrm{soft}}$,
**not** the intended $\theta_{\mathrm{hard}} \to \theta_{\mathrm{hard+soft}}$ where a
hard-label model improves itself with its own reasoning knowledge. The actual
hard-then-soft hypothesis has not been tested yet.

### B. Teacher probabilities are extreme and multi-select-biased

Re-computed over the uploaded 5,000-question teacher marginal targets:

| Teacher target statistic | Value |
|---|---:|
| candidate inclusion probability, mean | 0.685 |
| candidate inclusion probability, median | 0.993 |
| fraction of probabilities > 0.9 | 58.7% |
| expected selected per question | 3.47 |

For reference, the 1,000-question dev subset has a true mean of ~3.19 correct
answers. These are different samples so they are not a per-question error, but
they suggest the teacher systematically over-predicts the selected count.

More importantly, teacher probabilities are concentrated near 0 or 1. That is
over-confident hard judgement, not useful soft uncertainty. No independent BCE /
Brier / ECE diagnostic against gold labels has been computed for the teacher, so
these scores must not be treated as calibrated probabilities.

### C. Count is the bottleneck, not option ranking

From the latest raw validation files (oracle-count diagnostic: re-rank the
selected candidates given the true count):

| 5k model | Actual question EM | Oracle-count EM |
|---|---:|---:|
| Gold BCE | 50.15% | 73.36% |
| Gold Set CE | 39.37% | 76.54% |
| Soft Teacher Set CE | 24.39% | 70.32% |

Soft Teacher Set CE has a 45.93-point gap between actual and oracle-count EM.

Moreover, in the full 2,042-question dev set, 31.34% of questions have exactly 3
correct answers, and the 5k Gold Set CE count-head mode accuracy is also exactly
31.34%. The count head is not learning per-question counts; it is predicting the
mode. It has not moved beyond a constant predictor after only 79 steps.

So the next version should **not** stack more complex set losses. It should first
fix (1) teacher target semantics/calibration, (2) student count-head cold-start,
and (3) the hard-then-soft training-flow mismatch.

Note: the 5k students are all validated on the full 2,042-question dev set; the
1,000-question dev subset is used for teacher generation and diagnostics only.
The next design must distinguish these two evaluation sets explicitly.

## 3. First hypothesis to verify: Teacher–Readout Mismatch

The original hope was:

```
model reasons better -> better soft supervision -> better student decisions
```

This chain does not necessarily hold. The teacher was BCE-trained on candidate
**letter logits**, but supervision is extracted from **Yes/No token logits** after
reasoning. These two readouts are not calibrated to the same probability
semantics, so the reasoning-then-Yes/No readout may be a worse teacher than the
native decision the model already learned.

### Experiment A: fixed model, only vary the teacher readout

Using the same Gold-BCE checkpoint:

| Teacher mode | Purpose |
|---|---|
| no reasoning, native option logits | trained direct-decision baseline |
| no reasoning, per-option Yes/No | does the readout interface itself bias the teacher? |
| reasoning, per-option Yes/No | does reasoning add value? |
| Direct + Verify, multiple reasoning chains | do multiple views beat same-budget repeated sampling? |

Measure on a few hundred stratified questions: membership BCE/Brier, count MAE,
exact-set EM, truncation rate and coverage.

If the no-reasoning native logits are best and reasoning-then-Yes/No is worse,
fix the readout mismatch before scaling teacher count.

## 4. Proposed method: Count-Anchored Self-Distillation (SetJev v2)

### Modification 1: the teacher predicts "how many" independently

Collect, per reasoning chain, both candidate scores $u_i$ and an explicit count
distribution $t_m$:

$$
Q_T(S \mid x) = t_{|S|} \frac{\exp(\sum_{i \in S} u_i)}{Z_{|S|}(u)}.
$$

The teacher can now say "A, C, D are all likely correct, and three is the most
likely count", and the per-option inclusion probabilities are derived from the
same joint distribution so the targets stay consistent.

### Modification 2: cold-start the student count head from BCE

Instead of randomly learning the whole count distribution, use a residual count
correction:

$$
\rho_\theta(m \mid x) = \operatorname{softmax}_m \left[ \log Z_m(s) + a_\phi(m, x) \right]
$$

with a lightweight learnable correction $a_\phi$ initialized to zero. The new model
starts from the count implied by its own candidate scores and only then learns the
extra count signal from the teacher, avoiding a randomly-initialized count head
dominating output over short training.

### Modification 3: strict hard-then-soft training

Use a common hard-training checkpoint and branch:

$$
\theta_H \longrightarrow
\begin{cases}
\theta_{\mathrm{gold}} \\
\theta_{\mathrm{directKD}} \\
\theta_{\mathrm{reasonKD}} \\
\theta_{\mathrm{setKD}}
\end{cases}
$$

All branches share the same initialization, extra training data and update budget.
The soft stage mixes in real labels:

$$
L = \beta L_{\mathrm{gold}} + (1-\beta) L_{\mathrm{teacher}},
\qquad \beta \in \{1, 0.5, 0\}.
$$

Compare $\beta = 1, 0.5, 0$ rather than assuming pure soft supervision. To separate
reasoning-knowledge transfer from teacher memorization, isolate hard-training data
$A$ from distillation data $B$ by question group: the teacher trains on $A$ and
generates supervision for $B$ (never trained on in the hard stage); $B$ also gets a
matched-budget gold-continuation control.

## 5. Experiment sequence

| Gate | Experiment | Success criterion |
|---|---|---|
| 1 | no-reasoning vs reasoning teacher readout; calibration; truncation audit | reasoning demonstrably adds useful information |
| 2 | Gold-only vs Direct KD vs Reasoning KD vs Count-Anchored KD | improves full-set EM from a common start without hurting probability quality |
| 3 | residual count head vs full count head | cold-start correction improves learning |
| 4 | frozen teacher vs refreshed teacher | verify direct-decision gains do not degrade teaching utility |

Use small-scale single-seed diagnostics to localize the problem first; only run
3-seed, full-data and official test once a positive signal is seen. Do not spend
GPU on large multi-teacher runs while the soft labels are visibly broken.

The generational-teacher question is still worth keeping. In particular, if the
same model's native direct decision is good but its post-reasoning probabilities
are bad, that is a concrete entry point for the Decision–Teaching trade-off.
StartLux has already reported decision-model reasoning-termination and
reasoning-length issues, so coordination between the two modes is worth studying,
but the "next-generation teaching utility degrades" claim still needs independent
verification.

## 6. Repo status note

The `main` branch gained a full InnerJev/SII reproduction and JEVal results on
2026-10-24:

| Model | JEVal choice+noul accuracy |
|---|---:|
| SII-4B Full | 67.20% |
| SII-27B Full | 73.13% |

These are not SetJev multi-select scores and are not directly comparable, but they
serve as an auxiliary benchmark for checking whether native decision-model
capability is preserved.

`main` and the research branch have diverged. The next experiments should first
unify the evaluation-code version while preserving the old experiment configs and
checkpoint identities.

## Final recommendation

Shift the research focus from "design a more complex Set CE" to "make the reasoning
teacher produce genuinely useful, calibrated set supervision". The latest results
already point at the problem: candidate ranking still works, but teacher
probabilities are over-confident, the student over-selects, and the count readout is
under-trained. Solve that before asking why stronger direct decision-making might
not imply stronger next-generation teaching.
