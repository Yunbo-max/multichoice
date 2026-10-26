# Episodic SetJev A/B/C experiments: the bottleneck is training scale/mechanism, not the representation

Branch: `research/episodic-action-source-diagnostic-20261025`. Date: 2026-10-25.
Three controlled experiments on the shift-by-k environment isolate the failure
mechanism of the episodic memory training.

## Results

| experiment | backbone | memory training | outcome |
|---|---|---|---|
| A | Frozen Qwen | Expert replay | carry ≈ reset ≈ chance (v6 baseline) |
| B | Frozen Qwen | On-policy (DAgger) | carry 0.084 ≈ reset 0.089 (no fix) |
| C | LoRA Qwen | On-policy | runs, but train_nll 0.439 -> 0.429 (barely learns in 2 epochs) |

## What this rules out

1. **Expert-action offset is not the cause** — B (on-policy aggregation) did not
   help, so writing the expert action into memory is not the main failure.
2. **Frozen representation is not the whole story** — C (LoRA-unfrozen Qwen)
   runs but still barely learns in a short smoke, so unfreezing alone does not
   immediately fix it.

## What remains

The failure is in the **training scale / mechanism** itself: 1024 synthetic
episodes and a small 128-dim memory head trained with plain set NLL are not
enough to learn the dynamic decision representation. This matches the StartLux
observation that decision models are built with millions of examples (Stage 1
full-param SFT) before any reasoning/memory training.

## Next

Redesign the training data flow (On-Policy SetJev, per the earlier plan), rather
than continuing to tune the loss weight or the memory head size:

1. Stage A: a decision foundation (LoRA/SFT on single-step rule-conditioned
   choices with counterfactual samples).
2. Stage B: evidence-conditioned memory update (only ingest the rule when new
   evidence actually arrives).
3. Stage C: on-policy aggregation (already prototyped in B/C).

## Cross-reference

- `docs/episodic_action_source_diagnostic_results.md`: the action-source probe.
- `docs/episodic_v6_shift_head_debug.md`: the shift-head attempts.
- `docs/episodic_set_memory_status.md`: the overall episodic plan.
