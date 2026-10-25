# Episodic v6 action-source + no-switch diagnostic results: observe is neutral, offline->online shift is seed-dependent

Branch: `research/episodic-action-source-diagnostic-20261025`. Date: 2026-10-25.
Reuses the frozen v6 checkpoints (no retraining). Two diagnostics: a paired
expert-vs-policy action-source probe, and a no-rule-switch carry/reset/prime_only
counterfactual.

## 1. No-switch counterfactual (frozen rule; later_em)

| seed | carry | reset | prime_only |
|---|---:|---:|---:|
| s43 | 0.074 | 0.025 | 0.007 |
| s44 | 0.080 | 0.045 | 0.081 |
| s45 | 0.095 | 0.004 | 0.066 |

`carry >= prime_only` in all three seeds. Even with no rule switch, the observe
update is neutral-to-helpful and never corrupts the retained rule. This rules out
the "memory forgetting" hypothesis.

## 2. Action-source probe (expert vs policy actions; dev-fit, test-eval)

| seed | expert_fit | test_expert | test_policy | expert_later | policy_later |
|---|---|---:|---:|---:|---:|
| s43 | 0.624 | **0.552** | **0.182** | 0.592 | 0.188 |
| s44 | 0.321 | 0.236 | 0.242 | 0.216 | 0.222 |
| s45 | 0.235 | 0.165 | 0.161 | 0.169 | 0.165 |

Only s43 shows a clear expert-vs-policy gap (0.55 vs 0.18, a ~37pt drop): under
expert actions the memory encodes `k`, but under the model's own actions it does
not. s44/s45 show expert ~= policy, both low.

## 3. Verdict

- **Memory forgetting**: ruled out (carry >= prime_only everywhere).
- **Offline->online shift**: present in s43, absent in s44/s45 — seed-dependent.
- **Memory encoding**: weak overall (held-out probe ~0.17–0.23 vs chance 0.167).

The seed dependence means a single-seed claim is unreliable. s43 is the
informative case: the memory CAN encode `k` under expert actions, so the failure
is the model's own action drift, not a representation limit.

## 4. Next

DAgger-style on-policy aggregation for s43: add the model's own (policy) rollouts
to the training set with the gold labels, so the memory is trained on the states
it actually visits. This directly targets the offline->online shift that s43
exhibits, before touching the architecture.

## Cross-reference

- `docs/episodic_probe_fix_prime_only.md`: the preceding probe-fix round.
- `docs/episodic_v6_shift_head_debug.md`: the shift-head attempts.
