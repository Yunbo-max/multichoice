# Episodic v6 shift-head (rule_weight 0.2): still a dead end

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Retry of the shift-head fix with a milder rule supervision (`--rule-weight 0.2`)
and the corrected independent on-policy eval.

## 1. Offline carry across attempts

| attempt | offline carry (dev) |
|---|---:|
| v4 (no shift head) | ~0.42 |
| v5 (shift head, weight 1.0) | ~0.08 |
| v6 (shift head, weight 0.2) | 0.09–0.13 |

## 2. v6 online (independent on-policy, 3 seeds partial)

| seed | raw step_em | score-struct | shift_head | shift_head nll |
|---|---|---:|---:|---:|
| s44 | 0.095 | 0.182 | 0.166 | 1.82 |
| s45 | 0.102 | 0.167 | 0.172 | 1.82 |

Reference: random legal shift EM = 0.1667; uniform-6 NLL = 1.792.

## 3. Conclusion

1. **The shift head still learns nothing online**: step_em 0.166–0.172 ≈ chance and
   its NLL 1.82 ≈ uniform-6, regardless of `rule_weight` (1.0 or 0.2).
2. **Adding the shift head regresses the offline set prediction** (0.42 -> 0.09–0.13),
   a gradient-interference effect: the parallel shift head on the shared latent
   competes with the set decoder for the representation.
3. The corrected probe still shows weak k encoding (test ~0.18–0.26 vs chance
   0.167), slightly higher right after a rule switch.

## 4. What this rules out

The low on-policy accuracy is NOT fixed by: (P0) constraining the readout, nor
(P1) a parallel shift head with rule supervision. The remaining root cause is the
persistent memory update (`observe`) not retaining the rule across steps — the
rule only enters via `prime` and the occasional rule-change feedback, and the
memory GRU does not keep it.

## 5. Next candidates

1. **Direct rule supervision on the memory** (an auxiliary probe-style loss on the
   memory vector itself, not on a parallel head), so the gradient forces the
   memory update to store k.
2. **Decouple the shift head from the shared latent** (stop-gradient on the head's
   input) to avoid the gradient interference, then re-test.
3. **DAgger-style on-policy aggregation** to close the offline->online gap.

## Cross-reference

- `docs/episodic_v5_shift_head_debug.md`: the weight-1.0 attempt.
- `docs/episodic_v4_p1_memory_probe.md`: the diagnosis.
