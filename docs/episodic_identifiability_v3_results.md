# Episodic SetJev v3 (identifiable environment): memory helps trajectories, not uniformly per-step

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Real frozen-Qwen episodic results after the v3 identifiability fix (5 cells,
fixed 2 actions, no symmetric / absorbing shortcuts).

## 1. On-policy results (256 held-out test episodes, horizon 8, seeds 43/44/45)

| seed | arm | step_em | later_em | trajectory | set_nll |
|---|---|---:|---:|---:|---:|
| s43 | carry | 0.499 | 0.504 | **0.266** | 2.535 |
| s43 | reset | 0.496 | 0.501 | **0.000** | 1.371 |
| s44 | carry | 0.502 | 0.521 | **0.250** | 1.973 |
| s44 | reset | 0.409 | 0.415 | **0.000** | 1.180 |
| s45 | carry | 0.533 | 0.552 | **0.250** | 2.333 |
| s45 | reset | 0.380 | 0.377 | **0.000** | 1.333 |

## 2. What the v3 fix removed (v2 -> v3)

| v2 anomaly | v3 outcome |
|---|---|
| reset trajectory constant 0.297 (symmetric opposite-pair states) | reset trajectory **0.000** on all seeds |
| reset later_em (0.88 / 0.80) > first_em (select-all absorbing shortcut) | reset later_em 0.38–0.50, below the primed first step |
| carry not stably winning later_em | carry wins later_em on 2/3 seeds |

The 0.297 constant was exactly the 76/256 opposite-pair cohort (29.7%), and the
"select all" policy reproduced a fake perfect later_em; both are now impossible
by construction (5 cells, exactly 2 active, left/right always distinguishable,
no absorbing set).

## 3. Findings

1. **Trajectory success is decisive and clean: carry 0.25–0.27 vs reset 0.000
   on all seeds.** Memory is necessary to keep the rule alive across 8 steps; a
   memory-less policy never completes a single trajectory.

2. **Per-step EM is a partial win, not a clean one.** Carry beats reset on
   step_em/later_em for s44 (+0.09/+0.11) and s45 (+0.15/+0.18), but is tied on
   s43 (0.499 vs 0.496). The exact 2-of-5 set is intrinsically hard (C(5,2)=10
   pairs, so ~50% step EM is already far above the 10% chance level), and the
   memory benefit shows up most clearly at the trajectory level.

3. **Carry has higher (worse) set_nll** (1.97–2.54 vs 1.18–1.37). This is a
   calibration gap worth noting: carry is more confident but on-policy its
   accumulated errors make the set NLL worse than the memory-less reset. This is
   consistent with teacher-forced training + on-policy drift.

## 4. Honest conclusion

After fixing the environment's identifiability, the memory question has a clean
answer on the **primary** metric: carry beats reset decisively on full-trajectory
success (25% vs 0%). The per-step advantage is real but not uniform (2/3 seeds),
and the set-NLL calibration gap (carry worse) needs attention before any claim
about "calibrated latent reasoning".

This is a mechanism probe on a small synthetic rule environment, not evidence of
general latent reasoning (per `docs/episodic_set_memory_status.md`). The next
gate is a harder environment (new rules / candidate sets / longer horizons)
where carry must also beat a same-budget non-recurrent control.

## Cross-reference

- `docs/episodic_online_eval_anomaly.md`: the v2 anomalies this fix addresses.
- `docs/episodic_set_memory_status.md`: the overall episodic plan and caveats.
