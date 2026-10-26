# Stage 1: LoRA Qwen CAN overfit the single-step shift decision

Branch: `research/episodic-lora-reliability-20261026`. Date: 2026-10-26.
Diagnostic: can a LoRA Qwen learn the elementary single-step shift transition at
all? 101 train samples (state + rule demonstration -> shifted 2-of-7 set), 25
held-out combos, BCE objective, 30 epochs.

## Result

| metric | value |
|---|---|
| train BCE loss (final) | **0.00034** (essentially 0) |
| dev question EM (held-out 25) | 0.20 |
| chance (exact 2-of-7) | 0.048 |

The model overfits the 101 training samples to ~0 loss, so the base decision
representation, answer-slot readout, input encoding and optimizer are all
functional. Held-out generalization is weak (20%, ~4x chance): the model
memorises the seen (state, rule) mappings rather than the abstract "apply shift k"
operation.

## Conclusion

The earlier concern ("if LoRA cannot fit a few dozen fixed samples, more data will
not help") is NOT confirmed: the base can overfit the elementary transition. The
episodic memory failure is therefore NOT a base-representation problem — it is in
the memory / sequence training (Stage 2+). This also refines the A/B/C readout:
experiment C (LoRA on-policy episodic) did not learn because the episodic objective
is the problem, not the backbone's ability to do the single-step decision.

## Next (Stage 2)

Memory retention: put the rule demonstration at the episode start only, then check
whether the persistent memory retains it across later steps (carry vs prime_only
vs reset), with no rule switch.

## Cross-reference

- `docs/episodic_abc_conclusion.md`: the A/B/C experiments.
