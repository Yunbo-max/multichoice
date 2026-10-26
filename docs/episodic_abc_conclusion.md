# Episodic A/B/C experiments: conclusion — the failure is training scale/mechanism, not offset or frozen features

> **2026-10-26 reproducibility correction:** A LoRA-aware evaluator WAS added after this report (commit `7be558f`), so the older sentence below stating that no evaluator exists is stale. Its adapter restoration and the C trainer's duplicate 4B load / unselected checkpoint were incomplete; see [the LoRA reliability review](episodic_lora_review_and_roadmap_2026-10-26.md). The A/B/C pilot does not causally prove that data scale rather than representation is the only bottleneck. The original results below are preserved as historical records.

Branch: `research/episodic-action-source-diagnostic-20261025`. Date: 2026-10-25.
Three controlled experiments to isolate why the episodic memory does not encode the
rule. All reuse the shift-by-k environment (7 cells, 6 rules, horizon 8/12).

## 1. Results

| experiment | backbone | memory training | outcome |
|---|---|---|---|
| A (baseline) | frozen Qwen | expert replay | carry ~= reset ~= chance (offline 0.08–0.09) |
| B (DAgger) | frozen Qwen | policy rollout + oracle loss | carry 0.084 ~= reset 0.089 (no gain) |
| C (LoRA) | LoRA Qwen (r=16) | policy rollout + oracle loss | runs cleanly; train_nll 0.439 -> 0.429 in 2 epochs (barely moves) |

## 2. What each result rules out

- **A -> B**: on-policy aggregation (the model's own actions drive the memory)
  does NOT help. The expert-action offset (writing `A_t*` into memory, which the
  model then cannot reconstruct) is NOT the main cause.
- **B -> C**: unfreezing Qwen via LoRA does NOT rescue it in a short run. The
  frozen-feature limit is not the only bottleneck either.

The remaining explanation is that the **training scale / data flow is too small to
learn the dynamic decision representation**, exactly matching the StartLux staging
observation: StartLux Stage 1 uses ~2.7M decisions of full-parameter SFT before
adapter/reasoning stages. Our 1024 synthetic episodes are far below that.

## 3. Engineering status of C

`train_episodic_lora.py` now works (LoRA Qwen + memory head, on-policy replay,
gold only for loss). Two memory fixes were required:
- gradient-checkpointing was replaced by `accumulate_episodes=1` (the reentrant
  checkpoint was pathologically slow on the SSM),
- the SSM runs in bf16 (`mamba_ssm_dtype` overridden from float32).

The C checkpoint (LoRA + memory head) is NOT yet evaluable by
`eval_episodic_text_online.py`, which uses a frozen Qwen encoder; a LoRA-aware
eval is still missing.

## 4. Recommendation

Do not stack more loss terms or a v7 rule head. The next change should be a
training-data-flow redesign ("On-Policy SetJev" per the design notes): decide the
single-step decision foundation first, then the memory, then on-policy improvement,
with enough scale to see the train NLL actually move below uniform. The A/B/C
isolation is done; the useful next question is design, not another GPU run.

## Cross-reference

- `docs/episodic_action_source_diagnostic_results.md`: the expert-vs-policy probe.
- `docs/episodic_v6_shift_head_debug.md`: the shift-head attempts.
