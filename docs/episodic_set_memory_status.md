# Episodic SetJev: verification status and experiment plan (analysis)

Branch: `research/episodic-set-memory-20261024` (`6a9fa7e`). Date: 2026-10-24.
This is an analysis/status note written after pulling and exercising the branch;
it is not a claim of new results.

## 1. What the branch adds

Persistent episodic memory on top of the Recurrent SetJev core, following the
BDH-CQ two-state split (contextual memory `S_t` + per-query workspace `H_{t,r}`):

```
S_(e,0)=0; prime(demos) -> for t: decide() (within-query recurrence, no CoT)
  -> execute A_t -> observe(feedback) -> S_(e,t+1)=MemoryGRU(...); reset on done
```

Only `S` crosses environment steps; `H` is re-initialised per decision. The
carry-vs-reset comparison uses the SAME trained weights and compute/decoder
budget — a reset arm that discards `S` before every later decision is the
matched negative control.

Two synthetic mechanisms, both causally verifiable: **rule retention** (one
demonstration at start, must be retained) and **rule change**
(`--rule-switch-prob`, new demonstration arrives in a post-action observation).

## 2. What I verified

- `bash qz/run_episodic_set.sh smoke`: 14 episodic tests pass (causality, memory
  reset, state pass-through, BPTT, chronology, whitelist).
- Full suite: **210 tests pass**.
- Toy training runs and emits the `carry` vs `reset_every_step` eval block with
  `first_em / later_em / step_em / trajectory_em / set_nll_per_option`.
  (2-epoch run shows carry slightly lower set_nll, but that is noise, not evidence.)

## 3. The gap I found and filled

The real-Qwen pipeline (`build_episodic_features.py` -> `train_episodic_features.py`)
requires **real episode JSONL** (state/action/post-observation/reset metadata).
CMB/MultiRC static rows cannot be treated as episodes (per the branch's own
non-claims), and no episode generator existed.

I added:

- `harness/gen_text_episodes.py` — textual rule-retention/rule-change episodes in
  the `build_episodic_features.py` schema (K semantic `action_ids`, multi-hot
  `gold`/`executed_set`, `done` only at the last step, chronological).
- `qz/make_episodic_payloads.py` + `qz_episodic/` — one cache job (gen 1024/256
  episodes + frozen-Qwen feature encoding from `grouped_bce_4gpu_s44/final`) and
  three train jobs (seeds 43/44/45).

## 4. How the experiment answers the core question

The core question (per the branch README): does persistent memory yield a
*stable* benefit over single-shot and over memory-less recurrent decisions,
under equal compute?

Decision rule: `carry` must beat `reset_every_step` (identical weights/compute)
on **later-decision EM** and **trajectory exact** — a first-decision win is
insufficient (memory has nothing to carry at t=0), and a one-seed/tiny-toy win is
not a paper-level result. Seeds 43/44/45 and a separate eval seed with a longer
horizon are required.

## 5. Honest caveats

1. `train_episodic_features.py` reports **offline teacher-forced replay**
   accuracy, NOT on-policy interaction success. The rule-shift toy is the only
   genuinely on-policy harness.
2. The toy runner does not use Qwen and makes no claim of 4B learning.
3. No GPU/H100 results have been run on this branch yet.
4. The frozen-Qwen feature pipeline has not been executed end-to-end; the
   generator and payloads are written but the cache job has not been submitted.

## 6. Next

1. Submit the cache job, verify gen->frozen-Qwen-cache runs clean.
2. Train the three seeds, compare `carry` vs `reset_every_step` on later-decision
   EM and trajectory exact.
3. Only if `carry` wins stably, consider the rule-change variant as the stronger
   (memory *updating*, not just *retention*) probe, and a longer-horizon eval seed.
