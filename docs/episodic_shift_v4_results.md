# Episodic SetJev v4 (shift-by-k): memory wins per-step on every seed, trajectory becomes uninformative

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Real frozen-Qwen results on the harder shift-by-k environment (7 cells, 6 rules,
C(7,2)=21 pairs, horizon 12 test).

## 1. On-policy results (256 held-out test episodes, horizon 12, seeds 43/44/45)

| seed | arm | step_em | later_em | trajectory | set_nll |
|---|---|---:|---:|---:|---:|
| s43 | carry | **0.172** | **0.180** | 0.000 | 3.565 |
| s43 | reset | 0.138 | 0.143 | 0.000 | 3.338 |
| s44 | carry | **0.160** | **0.169** | 0.000 | 3.482 |
| s44 | reset | 0.047 | 0.045 | 0.000 | 3.208 |
| s45 | carry | **0.118** | **0.120** | 0.000 | 3.108 |
| s45 | reset | 0.038 | 0.033 | 0.000 | 3.565 |

## 2. Findings

1. **Carry beats reset on step_em and later_em for every seed.** Mean step_em
   0.150 vs 0.074, mean later_em 0.156 vs 0.074. Unlike v3 (where s43 was tied),
   the harder environment separates carry from reset uniformly: reset collapses
   to chance (1/21 = 4.8%) on s44/s45, while carry retains ~12–17%.

2. **Trajectory is 0 for both arms.** A 12-step horizon at ~15% per-step accuracy
   makes full-trajectory success ~0.15^12 ≈ 0, so the trajectory metric no longer
   discriminates (in v3 at horizon 8 it was 0.25 vs 0.00). This is a ceiling on
   what the current metric can show at this difficulty.

3. **Carry has slightly worse set_nll** (3.11–3.57 vs 3.11–3.34), the same
   teacher-forced-training + on-policy-drift calibration gap seen in v3.

## 3. Interpretation

The harder environment makes the memory advantage cleaner on the per-step
metrics: carry > reset on all seeds, with reset at chance. But the exact
2-of-7 set task is hard enough that absolute accuracy is low (~15%) and the
trajectory metric saturates at zero. To recover a trajectory-level signal, the
test horizon should be shortened (e.g. 9–10) or the active-count task made easier
while keeping the rule family hard.

This still only demonstrates that persistent memory helps retain a *specific
shift k* in a small synthetic environment — not general latent reasoning (per
`docs/episodic_set_memory_status.md`).

## Cross-reference

- `docs/episodic_identifiability_v3_results.md`: the v3 (5-cell rotation) results.
- `docs/episodic_online_eval_anomaly.md`: the v2 anomalies the identifiability fix addresses.
