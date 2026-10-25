# Episodic online eval: mixed result and a suspected environment/evaluator bug

Branch: `research/episodic-set-debug-20261025`. Date: 2026-10-25. Status: the
on-policy evaluation is NOT a clean positive for memory; the anomalies below
must be resolved before any claim about carry-vs-reset.

## 1. On-policy results (256 held-out test episodes, horizon 8)

| seed | arm | step_em | later_em | trajectory | set_nll |
|---|---|---:|---:|---:|---:|
| s43 | carry | 0.606 | 0.628 | 0.316 | 2.418 |
| s43 | reset | 0.361 | 0.347 | 0.297 | 1.453 |
| s44 | carry | 0.639 | 0.648 | 0.449 | 2.262 |
| s44 | reset | 0.845 | 0.884 | 0.297 | 0.438 |
| s45 | carry | 0.625 | 0.638 | 0.422 | 3.221 |
| s45 | reset | 0.766 | 0.799 | 0.297 | 0.505 |

## 2. Three anomalies

1. **Carry does not stably win.** `later_em` (the metric memory is supposed to
   help) is HIGHER for reset in s44 (0.884 vs 0.648) and s45 (0.799 vs 0.638).
   Only `trajectory` favours carry (0.316/0.449/0.422 vs a constant 0.297).

2. **Reset's later_em > first_em** (0.884 > 0.570, 0.799 > 0.535). This violates
   the causal setup: after `empty_memory()` at t>0, reset has no demo and no
   memory, so later steps cannot be more informed than the first (which primes
   the demonstration). A later step beating the primed first step is a red flag.

3. **Reset's trajectory is constant** (0.297 = 76/256 for all three seeds) while
   its first_em differs per seed. A constant trajectory across seeds with a
   varying first_em points to a deterministic path in the evaluator/environment,
   not a real policy.

## 3. Suspected cause: rule ambiguity on symmetric states

The environment rotates a 2-active set over 4 cells left or right. For an
**opposite pair** (e.g. `{Alpha, Gamma}`), a left shift and a right shift produce
the SAME set (`{Beta, Delta}`). On those states the rotation direction is
unidentifiable from the state alone, so a memory-less reset policy that guesses a
direction is still correct — inflating reset's later_em and making carry look
worse than it is.

The demonstration disambiguates the direction via a one-hot input, but reset
discards it after t=0, so its later-step success on symmetric states is
spurious.

## 4. What to do

1. **Remove the symmetry**: generate episodes whose initial and intermediate
   states never form opposite pairs (e.g. use 5 cells, or 3-active states, or
   forbid opposite pairs), so left/right are always distinguishable.
2. **Audit the reset branch**: confirm `eval_episodic_text_online.py` actually
   empties memory at `t>0` and that the demonstration never leaks into later
   decision prompts.
3. Re-run cache -> train -> online with the fixed generator and re-check
   `later_em` (carry must exceed reset) and `trajectory` (must vary, not be a
   constant).

Until the anomalies are explained, the on-policy numbers above are NOT evidence
for or against memory.
