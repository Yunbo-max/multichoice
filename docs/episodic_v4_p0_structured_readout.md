# Episodic v4 P0: structured readout — the low accuracy is a memory problem, not a readout problem

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Post-hoc readout on the frozen v4 checkpoints (no retraining). Same option scores,
two readouts: raw set-MAP over all C(7,2)=21 subsets vs structured softmax over
the 6 reachable successors `E_k = sum_{i in shift_k(S)} s_i`.

## 1. Results

| seed | raw step_em | struct step_em | raw set_nll | struct shift_nll |
|---|---|---:|---:|---:|
| s43 | 0.172 | 0.189 | 3.565 | 2.844 |
| s44 | 0.160 | 0.183 | 3.482 | 2.750 |
| s45 | 0.118 | 0.149 | 3.565 | 2.760 |

Baselines: random legal shift EM = 1/6 = 0.1667; uniform-6 NLL = log 6 = 1.792;
uniform-21 NLL = log 21 = 3.045.

## 2. Findings

1. **Constraining the readout to the 6 successors fixes the NLL** (3.5 -> 2.75)
   by removing the mass wasted on the 15 impossible sets, but the constrained NLL
   (mean 2.785) is still ~1 nat above the uniform-6 baseline (1.792).

2. **The structured readout barely moves accuracy**: step EM 0.15 -> 0.17, which
   is essentially the random-legal-shift level (0.1667). The model's six shift
   energies `E_k` carry almost no information about the true `k`.

## 3. Conclusion

The low on-policy accuracy is **not mainly a readout problem**. Even restricted
to the 6 reachable successors, the model is at chance, so the memory is not
encoding the rule `k` (or the decision head is not reading it). This directs the
next step to P1: a linear probe on the memory state to test whether `k` is
stored, plus a same-state counterfactual (same observable state, two histories
`h^(k1)` vs `h^(k2)`, must select different successors).

## Cross-reference

- `docs/episodic_shift_v4_results.md`: the v4 carry-vs-reset results.
- `docs/episodic_set_memory_status.md`: the overall episodic plan.
