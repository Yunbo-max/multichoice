# Episodic v4 P1: memory weakly encodes k, and the decision head does not read it

Branch: `research/episodic-identifiability-debug-20261025`. Date: 2026-10-25.
Linear probe + same-state counterfactual on the frozen v4 checkpoints.

## 1. Results

| seed | probe train acc | chance | counterfactual correct | responds to history |
|---|---|---:|---:|---|
| s43 | 0.271 | 0.167 | 1/6 | no |
| s44 | 0.271 | 0.167 | 1/6 | no |
| s45 | 0.258 | 0.167 | 0/6 | no |

The probe is a linear classifier decoding the shift `k` from the persistent
memory vector `M_t`; the counterfactual primes the same observable state with six
different demonstrations (k=1..6) and checks whether the selected set follows the
history.

## 2. Findings

1. **Memory encodes k only weakly.** Probe accuracy 0.26–0.27 is above chance
   (0.167) but far from separable, so `M_t` carries a little rule information and
   not much.

2. **The decision head does not read the memory.** The same state under six
   different histories yields essentially one prediction (`responds_to_history =
   false`), and the counterfactual correct rate is at chance (1/6 or 0/6). Even the
   rule information that IS in memory does not change the selected set.

## 3. Conclusion

Combined with P0 (structured readout barely moves accuracy), the low on-policy
accuracy is a **memory-to-decision pathway failure**: the memory weakly stores `k`
and the decision head ignores it. This is not a readout problem (P0) and not a
pure memory-storage problem (probe is above chance), but a coupling problem —
the stored rule does not steer the set decision.

## 4. Next (per the P2 decision table)

- Add rule-state auxiliary supervision to the memory update (so `k` is encoded
  more strongly), and/or
- Make the decision head explicitly condition on the memory (e.g. read the
  memory-conditioned structured readout over the 6 successors).

Both are supervised changes to `episodic_set_decision.py` / `train_episodic_features.py`;
no RL/GRPO needed.

## Cross-reference

- `docs/episodic_v4_p0_structured_readout.md`: the readout is not the bottleneck.
- `docs/episodic_shift_v4_results.md`: the v4 carry-vs-reset numbers.
