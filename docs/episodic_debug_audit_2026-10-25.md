# Episodic SetJev — Debug Audit (2026-10-25)

**Base:** research/episodic-set-memory-20261024 at 823ec93ca6ed3366a7f2ab71ec125031e7a80a3a.
**Fix branch:** research/episodic-set-debug-20261025.
**Validation status:** Python generator stress checks completed locally; new PyTorch regressions and real Qwen GPU forward still require QZ execution.

## Latest analysis / actual completed work

The upstream status note (docs/episodic_set_memory_status.md) records 14 episodic
tests and 210 full tests passed, an inconclusive two-epoch toy pilot, and QZ
payloads for frozen-Qwen feature extraction plus three seed runs. It explicitly
says no Qwen cache job had been submitted and no H100 results exist yet.
Files on GitHub alone are not evidence of finished training.

## P0: hidden rule switches before current decision (FIXED)

Old gen_text_episodes switched hidden direction at t>0 before computing gold_t
and only revealed the rule switch in post_action_feedback_t. Thus a model was
asked to predict a target depending on an unobservable new rule.
At horizon=6, switch p=0.25, expected affected proportion is 5/6*0.25=20.8%.

Fix: compute the current gold using the rule known at the start of t; execute
current action; change rule only afterwards; reveal new example in feedback_t
to use at t+1. Store private oracle_rule_before_decision metadata for audit,
never pass it to current prompt or the memory updater.
Generator now prevents overwriting existing datasets and uses deterministic
collision-free IDs derived from seed/index.

Local stand-alone Python stress check of corrected generator: 1000 episodes,
9000 decisions, 5949 post-action rule changes; all causal invariants passed.
A separate file-write test generated 50 episodes / 400 steps / 85 switches.
This is **not** a full remote integration test.

## P1: redundant frozen-Qwen inference (FIXED)

Original feature builder runs Qwen again for repeated prompt text, action labels
and feedback. Synthetic generator only has four cells, repeated 6-step episodes,
and identical descriptions; tens of thousands of unnecessary forward requests
could dominate the QZ cache job.

Fix: FrozenQwenDecisionEncoder caches exact prompt strings (LRU capacity 8192),
deduplicates each request, batches unseen prompts in groups of 4 and stores
feature vectors on CPU. Only valid with frozen/eval Qwen, as implemented.
Added tests for ordering, deduplication and eviction. Check numeric parity of
cached/uncached features on actual Qwen before a paper claim.

## P1: oracle replay mistaken for online execution (FIXED protocol)

The original Qwen episodic trainer evaluates against cached expert trajectories.
If a policy makes a wrong choice, those cached next states no longer describe
its environment. Offline teacher-forced EM cannot prove online success.

New harness/eval_episodic_text_online.py:
1. Start from the recorded initial visible state and initial demonstration.
2. Compute decision set with actual Qwen + episodic head, without text CoT.
3. Execute the model-selected set, not the logged oracle action.
4. Build next-state prompt and post-action observation from executed actions.
5. Reveal any next-step rule switch only after executing the current action.
6. Score each step using the private oracle rule, never forwarded to the agent.
7. Compare carry vs reset_each_step with identical weights and compute.

Added no-Qwen fake-encoder tests that inspect the actual next prompt and
ensure oracle-trajectory feedback is never reused.

IMPORTANT: model selection uses dev, so on-policy online evaluation must not
use the same dev episodes as an untouched score. Cache stage now generates
a separate 256-episode test set (seed 9043, horizon 8 vs train/dev horizon 6).
Three online QZ jobs evaluate trained heads on test, one per seed.
These test episodes share the same tiny synthetic rule family: this tests
execution/memory, NOT generalization to novel ARC rules.

## P1: QZ payload failure propagation (FIXED)

The QZ cache generator and all staged JSON jobs now use set -euo pipefail.
Prior set -uo pipefail let data generator failures continue into feature cache
steps, potentially using stale intermediate files. The cache job explicitly
refuses source without the causal rule audit marker. Output overwrite
protection ensures a failed prior job is not silently reused.

## New/updated files

- harness/gen_text_episodes.py: causal rule updates, metadata, ID/output safety
- harness/episodic_qwen_bridge.py: exact-prompt frozen feature cache
- harness/eval_episodic_text_online.py: genuine on-policy simulator rollout
- qz/make_episodic_payloads.py: fail-fast and test/online evaluation jobs
- qz_episodic/: refreshed cache/train configs and new online jobs for s43/s44/s45
- tests/test_episodic_generator_cache.py: temporal and cache regressions
- tests/test_episodic_online.py: true executed-action state transition checks
- qz/run_episodic_set.sh: expanded smoke tests

## Explicit validation sequence

    git fetch origin
    git switch research/episodic-set-debug-20261025
    bash qz/run_episodic_set.sh smoke

Run QZ jobs in order:
1. episodic-cache-bce-s44.json (generates new train/dev/test and frozen Qwen train/dev caches).
2. episodic-train-s43.json / s44 / s45 (offline supervised head training, selecting on dev).
3. episodic-online-s43.json / s44 / s45 (policy execution on untouched test).
Compare first/later exact set EM, strict whole-trajectory success and set NLL,
for carry vs reset. Do not claim improvements without repeated seeds and
equal-compute controls.

**Artifact safety:** staged QZ jobs now use fresh run roots `runs/episodic_data_causal_v2/` and `runs/episodic_qwen_causal_v2_s43/44/45/` rather than reusing prior episodic outputs. Jobs fail fast if these fresh paths already exist. Keep the code SHA together with every result.

## Outstanding research limits

1. No actual H100 results or full production-core test suite executed by this
   audit; the 14/210 test counts above are from the previous status note.
2. Four cells / two active cells / left-right shifts give extremely limited
   task diversity, so random-seed train/dev/test are structurally near-duplicate.
3. Qwen model identity is a user-supplied string; full checkpoint weight hashes
   are not verified.
4. The persistent-memory architecture uses independent proposed update rules,
   not proprietary BDH-CQ source code or a replication of its ARC score.
5. Offline training uses logged expert actions, not GRPO/RLCD2 or online RL.
6. The set distribution still has only count potential and candidate scores,
   not an unrestricted joint pairwise distribution.
