# Episodic SetJev v6: probe leakage audit and next decision gate

**Audited SHA:** `aa793886851dc46c2b542f03c3a4a39e966c02cd`  
**Fix branch:** `research/episodic-probe-group-split-20261025`

## What is new in the upstream branch

- `9757d84`: includes causal rule-switch feedback when updating the probe's memory and evaluates Raw/Structured/Shift readouts through independent on-policy rollouts.
- `338b1fa`: fixes a tensor/list type error in raw set-NLL of the structured evaluator.
- v6: shift-head training with rule CE coefficient **0.2** (lowered from 1.0).

Latest [v6 results](docs/episodic_v6_shift_head_debug.md):

| metric | result |
|---|---|
| v4 offline carry EM | about 0.42 |
| v5 offline carry EM (rule weight 1.0) | about 0.08 |
| v6 offline carry EM (rule weight 0.2) | 0.09–0.13 |
| v6 online shift-head step EM | s44 0.166, s45 0.172 |
| v6 online shift-head NLL | about 1.82 |
| theoretical uniform-6 shift NLL | log(6) = 1.792 |
| theoretical random legal shift EM | 1/6 = 0.1667 |

The latest evidence remains **negative** for improved calibrated rule prediction. The v6 results provided are incomplete (two online seeds are tabulated). Do not claim that the shift head learned the latent rule or that the update `observe` is the uniquely proven failure cause.

## A new concrete diagnosis: temporally correlated probe leakage

Previous `harness/memory_probe.py:fit_probe_split` used random **individual time steps**. The same episode often sends 12 highly correlated memory vectors into this evaluation, so memories from one evolving trajectory could appear in both fitted and held-out groups. The old report labeled this a test-set probe result, overstating independence.

`eval_memory_probe.py` also re-fitted a different probe for `after_switch` and `no_switch` subgroups. Those accuracies did NOT evaluate the same classifier, confounding the effect of a rule switch with retraining.

### Implemented fix

- `fit_probe_split(..., groups=episode_ids)`: split on **entire episode IDs**, keeping all consecutive hidden states together.
- `fit_probe_split(..., return_indices=True)`: return exact train/test step indices and episode counts for audit.
- `eval_memory_probe.py`: use the SAME learned probe to score switch/no-switch held-out subsets; report subgroup counts.
- `--probe-fit-episodes DEV.jsonl`: preferred path, train the diagnostic probe on **separate DEV episodes**, then score on the entire untouched TEST set from `--episodes TEST.jsonl`.
- `qz_memory_probe/episodic-memory-probe-s43/s44/s45.json`: explicit DEV fitting and TEST scoring; versioned QZ task/log/result names to avoid overwriting original reports.
- `tests/test_memory_probe_episode_split.py`: disjoint groups, deterministic split, invalid groups, fixed probe predictions and evaluator import.

The original reported 0.18–0.26 rule-probe results must be **remeasured** with the corrected episode-independent evaluation. No corrected 4B measurements have been executed in this review.

## Scientific caution: v5/v6 attribution is still not identified

The v6 report concludes that `observe` forgets the rule. That is a useful **hypothesis**, but not established by the observed output alone.

1. Added `shift_head` modifies model parameters and gradients. A `rule_weight=0` **matched shift-head control** is needed to isolate architecture changes from auxiliary supervision.
2. Exact-set NLL in the recurrent head is divided by candidate count `K=7`; shift CE is not. Weight `0.2` can still represent a material share of the objective. Compare actual gradient norms/cosines before claiming interference.
3. A test of **prime-only memory** (retain the initially demonstrated rule but skip later `observe` writes) versus `carry` and `reset`, both at zero rule switches and under rule changes, is needed to distinguish forgetting from over-writing and representation failure.
4. Compare a **paired oracle-action replay** and a policy-action online rollout using the same checkpoint and environment transition timing to quantify actual distribution drift.
5. Same-visible-state, different-demonstration counterfactual probabilities are more informative than simply measuring whether the argmax changes.

## Recommended immediate execution

On the QZ worker after reviewing the branch:

    git fetch origin
    git switch research/episodic-probe-group-split-20261025
    python -m unittest discover -s tests -p test_memory_probe_episode_split.py -v

Run staged `qz_memory_probe/episodic-memory-probe-s43/s44/s45.json` after ensuring both `episodic_data_shift_v5/dev.jsonl` and `test.jsonl` exist. These jobs will not modify the training model. Use each saved model's immutable weight identity rather than relying on a user-supplied label when preparing final paper results.

**Validation status:** GitHub source and payload changes were committed and read-back checked. The new CPU regression suite, live CUDA model probe and QZ jobs were NOT executed by this audit. No corrected numerical claim is made.
