# Episodic SetJev: identifiability + action-shortcut audit (2026-10-25)

Base: research/episodic-set-debug-20261025 at c34ef0982f5eecf8c02400a18813a9d76f46d6f7.
Fix branch: research/episodic-identifiability-debug-20261025.

## 1. What latest experiments actually show

Source: docs/episodic_online_eval_anomaly.md, 256 test episodes / 8 steps.

| seed | carry later EM | reset later EM | carry full trajectory | reset full trajectory |
|------|---------------:|---------------:|----------------------:|----------------------:|
| 43   | 0.628 | 0.347 | 0.316 | 0.297 |
| 44   | 0.648 | 0.884 | 0.449 | 0.297 |
| 45   | 0.638 | 0.799 | 0.422 | 0.297 |

Carry does NOT consistently win later EM. These are old **4-cell** results,
not valid results for the repaired protocol below. The record reports set
NLL too; NLL on the new constrained action space is not directly comparable.

## 2. Exact root-cause evidence: the old reset 0.297 cohort

The old generator sampled 2 active cells from a ring of 4. There are
C(4,2)=6 legal two-active states. Two states are opposite pairs {0,2} and
{1,3}, on which left and right rotations coincide:

    R_+({0,2}) = R_-({0,2}) = {1,3}

Opposite pairs stay opposite under any sequence of +/-1 shifts. Thus, for
these episodes, the rule direction is irrelevant at every step and carrying
the demo is unnecessary.

We independently replayed the original Python RNG call sequence with test
seed 9043 / 256 episodes / horizon 8 / switch p=.25. Exactly **76 of 256**
initial pairs were opposite: 76/256 = 0.296875, exactly matching the 0.297
reset full-trajectory rate in ALL THREE seeds.

This match is a strong diagnostic, NOT definitive per-episode attribution:
the uploaded aggregate report lacks predicted-set per-episode traces needed to
prove these exact 76 IDs are those the models solved. Such traces should be
recorded for any follow-up causal analysis.

Importantly, reset later EM > first EM is not itself a causal contradiction.
Later states can simply be easier, even in memory-free evaluation.

## 3. A second shortcut survived naive 5-cell replacement

With 5 cells, nonempty proper sets distinguish left from right, but a model
that is allowed to choose all 5 actions can execute the full set. Afterwards
R_+(FULL) == R_-(FULL) == FULL, so the environment is an absorbing trivial
state. An empty-set action has the same problem if permitted.

A concrete deterministic counterexample on the old environment:
execute ALL four actions at t=0, then always execute ALL four again.
With an initially two-active state, t=0 is wrong; because FULL is absorbing
under both rotations, the subsequent 7 decisions are all correct. This
constructs first-step EM=0, later EM=1 and strict trajectory EM=0 without
learning any hidden rule or memory. It demonstrates a protocol-level metric
failure, **not** that a specific trained checkpoint actually used this policy.
Per-episode predicted-set cardinalities must be inspected to confirm that.

**Fix both issues, at once:**

- Ring size K=5, two simultaneous active cells/actions.
- The PUBLIC selection protocol explicitly declares allowed_cardinalities=[2].
- Generator, frozen-Qwen prompt, feature-cache builder, exact set NLL and
  on-policy set-MAP decoder all agree on that permitted set size.
- Native SetJev calls lacking the optional field retain their exact old
  semantics and prompt bytes. input_digest records this public field only
  when present to avoid conflating different action protocols.
- Demo now supplies only a one-hot input/output example, not explicit words
  "left shift" or "right shift".
- New episode schema: cyclic5_fixed2_v3. Older synthetic episode formats are
  rejected before expensive Qwen extraction / real online evaluation.

Proof of identifiability for K=5: If R_+(S)=R_-(S), then S is invariant under
a shift of 2. Because gcd(2,5)=1, the orbit visits every position, so S is
either empty or full. Neither is legal under |S|=2. Therefore EVERY one of
C(5,2)=10 legal states has distinguishable left and right outcomes.

Independent local combinatorial test: 5-cell/2-hot has **0 / 10** ambiguous
states, compared with **2 / 6** for the old four-cell environment. A
256-episode / 2048-decision / seed-9043 local protocol simulation found
458 rule switches and **zero** ambiguous states, with each target calculated
from its causally known pre-decision rule. This validates mathematical
logic, NOT production GPU inference accuracy.

## 4. Stronger data integrity

The generator validates ALL of the following:
- schema, stable action IDs, exact-two gold and logged expert selections;
- every state's exact visible text, oracle action derived from current rule;
- initial demonstration agrees with the first hidden rule;
- post-action feedback matches the executed action;
- if hidden rule changes, the NEXT rule is only revealed in post-action
  feedback, never before a current decision;
- no symmetry or absorbing all-selected states.

build_episodic_features.encode_episode validates the synthetic schema and
chronology before any frozen-Qwen encoding. eval_episodic_text_online does
a dataset-wide preflight BEFORE loading the 4B checkpoint.
New tests verify the old 76/256 symmetry cohort, enumeration, fixed support,
unchanged legacy protocol, prompt whitelist, corrupted-feedback rejection,
and on-policy actor selection.

## 5. New experiment paths and protocol

All staged QZ payloads now point at fresh, isolated paths; submitted job
names and log names are also uniquely tagged v3 (payload filenames remain
stable for the documented CLI and repository links):

    runs/episodic_data_identifiable_v3/
    runs/episodic_qwen_identifiable_v3_s43/
    runs/episodic_qwen_identifiable_v3_s44/
    runs/episodic_qwen_identifiable_v3_s45/

Cache job -> three head-training jobs -> three actual-action online jobs.
No existing v2 checkpoints should be loaded or renamed as v3.

Run local / worker smoke:

    git fetch origin
    git switch research/episodic-identifiability-debug-20261025
    bash qz/run_episodic_set.sh smoke

Run staged QZ jobs only after smoke, in this order:
1. qz_episodic/episodic-cache-bce-s44.json
2. qz_episodic/episodic-train-s43.json, s44, s45
3. qz_episodic/episodic-online-s43.json, s44, s45

Only dev may select checkpoints; hold out the longer-horizon test. Report
first/later set EM, strict trajectory accuracy, exact-set NLL, invalid
actions count, oracle-minus-blind controls, run time and full provenance.

## 6. What this does NOT establish

- There are no new GPU/real-Qwen scores for the v3 protocol yet.
- Test code committed here has not been executed on a full repository
  checkout / CUDA QZ worker in this audit.
- A five-cell cyclic shift is still an extremely narrow synthetic family
  (10 two-hot states and two directions). Train/test seeds only split examples,
  not transformation families; a positive result is a *mechanism probe*,
  not ARC or generalized latent reasoning.
- The optional cardinality field removes the learned count-selection
  difficulty in this probe deliberately, so this pilot isolates MEMORY
  rather than the broader Multi-select cardinality learning problem.
- Carry-vs-reset with identical weights isolates access to hidden recurrent
  state, not necessarily all alternative ways of storing history. Current
  state may externalize some prior decisions. Future work needs paired
  opposite-rule counterfactuals under matched visible states, and
  compute-matched nonrecurrent / explicit-history baselines.
- BDH-CQ's proprietary exact update mechanism has not been reproduced.
