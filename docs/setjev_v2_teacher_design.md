# SetJev v2 teacher design: structured reasoning for multi-select decisions

Date: 2026-10-24. Status: design only, nothing implemented yet. To be reviewed.

## 1. Motivation

The task is multi-select: the model must output a *subset* of the K candidates,
not a single argmax. Single-choice reasoning (StartLux AutoThink, a linear chain
ending in one letter) does not fit. The decision space is the power set 2^K, and
the natural structure over subsets is a DAG (the Boolean lattice), not a chain.

The teacher's job is to produce a coherent set distribution

$$Q_T(S|x) = t_{|S|}\frac{\exp(\sum_{i\in S}u_i)}{Z_{|S|}(u)}$$

from which per-option membership and count targets are derived. The reasoning
structure exists only to make `u_i` and `t_m` better.

## 2. Core decomposition: count, then membership

Multi-select factorises into two coupled sub-decisions:

1. **count** — how many options are correct (`t_m`),
2. **membership** — which options, given the count (`u_i`).

The structure should mirror this, so a two-phase design is the default:

- **Phase 1 (count):** refine the count distribution `t_m` (native count prior +
  a little exploration).
- **Phase 2 (membership):** given a target count `m`, decide which options are in.

Both phases write back to `t` and `u`, keeping the student-facing `Q_T` inside
the Count-Anchored (cardinality-potential) family unchanged.

## 3. Search space: the subset DAG (Boolean lattice)

- Nodes are subsets `S ⊆ {1..K}`; edges are `S → S∪{i}`.
- The tree of include/exclude branches is an *unrolled* DAG that repeats states
  (e.g. `{A,B}` reached twice); the DAG is the minimal encoding.
- The subset score is the Count-Anchored log-probability
  `log t_{|S|} + Σ_{i∈S} u_i − log Z_{|S|}(u)`.

Only the **uncertain** options (`|σ(u_i) − 0.5| ≤ τ`) need branching; the
confident options are fixed by the native readout. Count pruning keeps the
explored depth plausible (`t_m > ε`). With ≤8 options and usually 1–3 uncertain
ones, the effective search is tiny.

### Gate: when to reason at all

- If no option is uncertain → **short thinking**: return `Q` from the native
  readout directly (no reasoning). This is the flat baseline (Gate 1: native
  logits already beat Yes/No).
- Otherwise → structured reasoning (below), with a fixed budget.

## 4. Search strategies: beam and MCTS (both, to compare)

- **Beam search:** keep the top-W subsets per level; branch only on uncertain
  options. Deterministic, cheap, easy to reproduce.
- **MCTS:** UCB over include/exclude edges, guided by the native score as the
  prior and the count distribution as the prior over depth. Better suited if
  option dependencies make the value non-monotone.

Both are implemented and compared; the experiments decide which one the teacher
uses (they may differ by question difficulty).

## 5. Exploration content: fast subset scoring

Each explored node is evaluated by a **fast scalar score** for the candidate
subset `S` (single readout / computed from `u,t`), *not* by a per-option
reasoning chain. "Fast" means one pass per node, not a long trace.

The refined `u'` and `t'` come from the scored leaves: the teacher renormalises
the explored subsets' scores into membership marginals and a count distribution,
then returns

$$Q_T(S|x) = t'_{|S|}\frac{\exp(\sum_{i\in S}u'_i)}{Z'_{|S|}(u')}.$$

## 6. Option-dependency graph (retained, later)

As a separate dimension, the options themselves have semantic relations
(contradiction, implication, equivalence, clustering). A graph over *options*
captures these and can prune the subset DAG (e.g. "A and B are mutually
exclusive" forbids `{A,B}`). This is retained for a later version (v3) and is
not required for the first implementation of the two-phase DAG.

## 7. Differences from StartLux (why this is not a copy)

| | StartLux | SetJev v2 teacher |
|---|---|---|
| model | AutoThink transformer, bidirectional prompt attention | hybrid linear-attention (SSM), causal |
| decision object | single choice (argmax) | multi-select subset (DAG) |
| structure | linear chain → re-read | two-phase count→membership over a subset DAG |
| gate | `max(p0)` | set-level: `min_i|σ(u_i)−0.5|` + count entropy |
| search | none | beam and MCTS |
| evaluation | re-read letter probs | fast subset scoring |
| aggregation | linear mixture p0/p1 | refine u,t → cardinality-potential Q_T |

## 8. Experiment plan

1. **Ablate the gate**: confirm the set-level uncertainty criterion (min margin /
   count entropy) selects the questions that actually need reasoning.
2. **Beam vs MCTS**: same budget, compare refined-teacher calibration (membership
   BCE/Brier, count MAE) against the flat native readout.
3. **Fast scoring**: verify one-pass subset scoring does not need a reasoning
   trace to improve `u,t`.
4. **Two-phase**: confirm count-then-membership beats flat membership on the
   count-calibration metric (the known failure mode of the current teacher).
5. **Student distillation**: plug the structured teacher into the Count-Anchored
   targets and run the hard→soft KD arms from the BCE checkpoint.

## 9. Open items for review

- Definition of "fast subset scoring" readout: computed from `u,t` (free) vs a
  one-pass "is S correct?" readout (one extra read).
- The `τ` threshold and the count-entropy gate need calibration on real data.
- Whether the teacher refines `u` and `t` jointly (one update) or sequentially
  (count first, then membership) — the two-phase doc assumes sequential.
