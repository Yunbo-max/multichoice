# Episodic v6 probe-fix + prime_only ablation: memory weakly encodes k, observe is not corrupting

Branch: `research/episodic-probe-group-split-20261025`. Date: 2026-10-25.
Episode-disjoint probe (fit on dev episodes, evaluate on held-out test episodes)
plus a third memory mode `prime_only` (freeze the primed memory `S_0`, no observe).

## 1. Held-out probe (dev-fit / test-eval, per-episode split)

| seed | train acc | test acc | after-switch test | no-switch test | chance |
|---|---|---:|---:|---:|---:|
| s43 | 0.385 | 0.202 | 0.277 | 0.182 | 0.167 |
| s44 | 0.311 | 0.234 | 0.175 | 0.251 | 0.167 |
| s45 | 0.225 | 0.172 | 0.216 | 0.159 | 0.167 |

The memory encodes `k` only weakly: held-out test accuracy is barely above chance.
Train accuracy is higher (overfit to the dev fit set), but the held-out signal is
~0.17–0.23.

## 2. prime_only ablation (carry / reset / prime_only, on-policy)

| seed | carry step_em | reset step_em | prime_only step_em |
|---|---:|---:|---:|
| s43 | 0.079 | 0.032 | 0.015 |
| s44 | 0.095 | 0.050 | 0.089 |
| s45 | 0.102 | 0.011 | 0.075 |

`carry >= prime_only` in all three seeds, so the observe update does not corrupt
the initial rule (the "memory forgetting" hypothesis is not supported). `carry`
still beats `reset`, confirming the v4 result.

## 3. Three-mechanism diagnosis

| mechanism | verdict |
|---|---|
| Memory forgetting (observe corrupts rule) | not supported: prime_only <= carry |
| Decision readout failure (probe reads k, head does not) | unresolved: probe itself is barely above chance |
| Offline-to-online shift | to test: expert vs model trajectory probe |

The more fundamental finding is that the memory never encodes `k` strongly
(held-out probe ~0.2 vs chance 0.167), so the first thing to fix is the encoding,
not the readout.

## 4. Next

Expert-trajectory vs model-trajectory probe: fit the probe on oracle/teacher-forced
rollouts and on the model's own on-policy rollouts from the SAME checkpoint, and
compare held-out accuracy. A large drop from expert to model trajectories would
isolate the offline-to-online distribution shift.

## Cross-reference

- `docs/episodic_v6_shift_head_debug.md`: the shift-head attempts (v5/v6).
- `docs/episodic_v4_p1_memory_probe.md`: the original diagnosis.
