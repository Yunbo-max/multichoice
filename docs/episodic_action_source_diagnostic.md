# SetJev v6: Action-as-oracle diagnostic and prime-only interpretation

**Audited branch:** research/episodic-probe-group-split-20261025  
**Audited SHA:** d8e56f03c302d248cd625766f61290b706a92757  
**New diagnostic branch:** research/episodic-action-source-diagnostic-20261025

## 1. Latest recorded evidence, not a new accuracy claim

The prior audit's DEV-fit/TEST probe now reports test rule decode EM:
s43 = 0.202, s44 = 0.234, s45 = 0.172 (mean ≈0.203) versus
chance 1/6 = 0.1667. The prime_only ablation on the v6 online policy:

| Seed | carry | reset | prime_only |
|---|---:|---:|---:|
| 43 | 0.079 | 0.032 | 0.015 |
| 44 | 0.095 | 0.050 | 0.089 |
| 45 | 0.102 | 0.011 | 0.075 |

These are recorded **online Step EM** on a mixture of unchanged and changed
rules, not a controlled no-switch memory-retention test. The statement
"carry >= prime_only therefore observe does not corrupt memory" is
**not logically justified** when the rule can change during the episode:
prime_only necessarily cannot ingest new rule demonstrations.

Under a six-rule Markov process which switches with probability p to
a uniformly chosen *different* rule, a hypothetical perfect classifier
retaining ONLY the initial rule k_0 would match the current rule at step t
with probability

    P(k_t == k_0) = 1/6 + (5/6)*(1-6p/5)^t .

For p=.25 and T=12 decisions, the expected step-average agreement is
about 0.395. This illustrates why stale memory has an intrinsic limitation
on switching episodes; it does not predict SetJev's observed step EM.

A retention diagnostic needs (a) separate p=0 episodes, (b) separate p>0
episodes, (c) prime_only vs carry vs reset with identical checkpoints,
episode seeds, readout and inference budget. Frozen initial memory and
on-policy memory updates have different information at post-switch steps.

## 2. Stronger new mechanism hypothesis: offline expert action is an oracle

See harness/train_episodic_features.py:replay_episode. After every decision,
the offline trainer feeds into memory.update:

    executed_set = step["executed_set"]
    post_observation = step["post_observation"]

In the synthetic training data, executed_set equals the correct oracle
gold set, which by construction is shift_k(current_state).

Because the six nonzero shifts of a legal two-active state on a seven-cell
prime ring are all distinct, the pair

    (current_state, expert_executed_set)

**uniquely identifies the rule k**, even when the original demonstration
is forgotten. That is an additional supervised signal silently present in
the recurrent *future input distribution* during expert replay.

On-policy rollout, by contrast, the model may execute the wrong set, and the
observed post-action state is its *own* action. A mistaken action is not a
reliable observation of the environment's hidden k. This discrepancy can
produce a model that scores well under teacher forcing but fails to read,
retain, or update k under its own actions.

This is a **concrete causal shortcut hypothesis supported by the code**,
not an established empirical explanation for all v5/v6 failures. The new
paired experiment is designed to quantify it.

## 3. New paired diagnostic code

harness/eval_memory_probe.py now accepts:

    --compare-action-sources --probe-fit-episodes DEV.jsonl --episodes TEST.jsonl

The collection path has two causal execution modes:

- expert: executes shift_k(x_t) after each decision (mimics the logged training
  trajectory). The hidden rule is used by the *environment's executor*, not
  inserted into the current model prompt.
- policy: executes the model's predicted set as on-policy evaluation does.

Both share identical test episodes, the same exogenous rule-change schedule,
the same Qwen model and identical trained episodic checkpoint.

The diagnostic **fits a single linear rule probe on expert DEV memories**,
then applies it unchanged to *both* expert TEST and policy TEST memories.
It reports matched test accuracy, later-step accuracy, after-switch and
no-switch later-step accuracy, with sample counts. No TEST features are used
to fit the probe.

Prediction:
  If expert TEST probe is high but policy TEST is low, then action-conditioned
  memory enters a distribution shift under self-execution.
  If both are low, then this particular linear probe cannot recover k,
  even with oracle-action history: do not attribute the failure to drift alone.
  If both are high but decision-set EM is low, diagnose memory-to-decision
  coupling and training gradients.

The matched comparison itself does not prove that observe "forgot" or that
retraining with GRPO would help.

## 4. New tests and jobs

- tests/test_action_source_diagnostic.py: exact recoverability of k from
  expert (current set, next set); fake-agent paired rollout confirms that
  memory is the same at t=0 but diverges after an incorrect self-action.
- qz/run_episodic_set.sh smoke: includes the above regression.
- qz_memory_probe/episodic-action-source-probe-s43/s44/s45.json:
  DEV probe fitting; matched TEST expert vs policy trajectories, with fresh
  output filenames to preserve prior probes.
- harness/eval_episodic_text_online.py --include-no-switch-control:
  paired environment counterfactual (same initial episodes / checkpoints)
  which freezes the initial hidden rule for all subsequent steps and suppresses
  rule-switch demonstrations in post-action feedback. All actions are still
  chosen and executed by the corresponding model policy.
- qz_memory_probe/episodic-memory-mode-noswitch-s43/s44/s45.json:
  compare carry, reset, and prime_only on both the recorded switching
  schedule and its matched no-switch environment. This identifies whether
  retaining S_0 helps when the hidden rule genuinely remains unchanged.

Use:

    git fetch origin
    git switch research/episodic-action-source-diagnostic-20261025
    python -m unittest discover -s tests -p test_action_source_diagnostic.py -v

Then run the three action-source QZ job configs AND the three
memory-mode no-switch QZ job configs (no retraining of the 4B model).
These are frozen-Qwen inference/feature-extraction diagnostics only.
Original switching and counterfactual no-switch episodes are paired using
the same visible initial state and exogenous episode identity. Interpret the
post-hoc counterfactual as a diagnostic, not a fresh untouched benchmark.

## 5. Pending validation and next architectural fork

New GitHub code has been committed, but the full source-level test suite,
4B paired diagnostic, and QZ jobs have not yet run from this review; no
expert-vs-policy accuracy or performance gain is claimed.

Do NOT simply add more rule CE or another parallel shift head yet. If the
expert-vs-policy difference is large, the next training intervention should
be interactive imitation/data aggregation, i.e., train the recurrent head
on policy-executed states with a fresh oracle **label** (for loss only), not
oracle-executed actions fed into future memory inputs. Preserve a matched
expert-replay baseline, evaluate on a disjoint on-policy test and report
both NLL and decision EM.

Independently, p=0 prime_only vs carry is needed to test loss of initial rule
information separately from adaptation to switches.
