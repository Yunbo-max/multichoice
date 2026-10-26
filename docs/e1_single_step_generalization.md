# E1: single-step generalization — the model memorises, it does not learn the shift operation

Branch: `research/episodic-lora-reliability-20261026`. Date: 2026-10-26.
Strict generalization tests on the 126 elementary state-rule combinations
(C(7,2)=21 active 2-sets x 6 cyclic shifts). LoRA Qwen, BCE objective, 30 epochs.

## Results

| split | train | test | dev question EM | interpretation |
|---|---|---|---:|---|
| random | 101 | 25 | 0.20 | overfit, weak generalisation (p~0.406 vs 1/6) |
| unseen_rule | 105 (k=1..5) | 21 (k=6) | **0.048** | = chance; cannot infer a new shift |
| unseen_state | 90 (15 states) | 36 (6 states) | **0.0** | cannot transfer the shift to new states |

Chance for exact 2-of-7 set selection is 1/21 = 0.048.

## Conclusion

The model fits the seen (state, rule) -> set mappings (train loss -> 0.0003) but does
not learn the abstract "shift by k" operation: it fails at chance on a held-out
rule and at 0% on held-out states. Stage 1 therefore demonstrates memorisation
only, not single-step rule generalisation.

This is the primary bottleneck, before any memory or sequence question: the rule
input representation / training diversity does not let the model extract the
elementary transition.

## Next (per the E1 decision rule)

Improve the rule input and training combination diversity, rather than adding
recurrence depth:

1. Make the rule statement explicit (e.g. "shift every active component forward
   by 2 positions") instead of a single input->output example.
2. Counterfactual training: the same state paired with different rule examples,
   forcing a rule->output function rather than a state->output lookup.
3. Wider state x rule combination coverage in the training set.

## Cross-reference

- `docs/stage1_single_step_overfit.md`: the memorisation result.
- `docs/episodic_abc_conclusion.md`: the episodic A/B/C experiments.
