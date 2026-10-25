# Episodic v5 shift-head debug: rule supervision did not fix the memory pathway

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Attempt to fix the P1 diagnosis ("memory weakly encodes k, decision head does not
read it") by adding a shift head on the memory-conditioned latent, trained with
rule supervision (`--rule-weight 1.0`), and reading it via the structured
(6-successor) readout.

## 1. Results (v5, rule_weight 1.0, vs v4 baseline)

| seed | raw step_em (v4 -> v5) | score-struct | shift_head | shift_head shift_nll | probe (v4 -> v5) |
|---|---|---|---|---:|---:|---:|
| s43 | 0.172 -> 0.064 | 0.158 | 0.155 | 1.813 | 0.271 -> 0.202 |
| s44 | 0.160 -> 0.126 | 0.174 | 0.177 | 1.799 | 0.271 -> 0.209 |
| s45 | 0.118 -> 0.033 | 0.163 | 0.181 | 1.797 | 0.258 -> 0.194 |

Reference: random legal shift EM = 0.1667; uniform-6 NLL = log 6 = 1.792.

## 2. Findings

1. **The shift head learned nothing online.** Its per-step shift NLL (1.80–1.81)
   is essentially the uniform-6 baseline (1.792), i.e. `P(k)` is uniform. Despite
   direct `cross_entropy(shift_logits, k)` supervision at weight 1.0, the shift
   head degenerates to chance on the on-policy rollout.

2. **Rule supervision hurt the base task.** Raw set EM dropped (0.15 -> 0.03–0.13)
   and the memory probe dropped (0.27 -> 0.20). `rule_weight=1.0` crowds out the
   set NLL and, more importantly, the supervised shift signal does not survive the
   offline->online shift.

## 3. Interpretation

This is a second, stronger symptom of the same offline->online gap that already
showed up as "offline carry 0.54 vs online 0.15" in v4. The shift head is trained
on teacher-forced memory and evaluated on drifted on-policy memory; the mapping
learned offline does not transfer, collapsing to uniform. The memory-update
pathway (`observe`) only receives the demo at `prime` (t=0) and does not retain
the rule across later steps.

## 4. Next candidates

1. Lower `--rule-weight` to 0.1–0.2 (mild auxiliary) and re-check.
2. Feed the rule/demo evidence into `observe` across steps, so the persistent
   memory keeps updating the rule, not only at prime.
3. DAgger-style on-policy data aggregation to close the offline->online gap.

## Cross-reference

- `docs/episodic_v4_p1_memory_probe.md`: the diagnosis this attempted to fix.
- `docs/episodic_v4_p0_structured_readout.md`: the readout is not the bottleneck.
