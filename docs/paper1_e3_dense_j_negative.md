# Paper 1 — E3 learned-pair attempt: dense J does not recover the matching

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
E3 tried a learned pair head (dense upper-triangle J, full 2^K enumeration, K=8)
trained with joint set CE on the controlled R regime. Result: the pair head does
not recover the true disjoint matching.

## What was implemented

- `Readout` objective `pair`: a `pair_head` (Linear(hidden, K*(K-1)//2)) emitting
  dense upper-triangle J logits, plus the existing count head.
- `torch_pairwise_log_partition` in `harness/set_distillation.py`: autograd-safe
  full-enumeration log Z with dense pairwise terms (K <= 12), verified against
  exact enumeration.
- `pair_loss_terms`: joint set CE `log Z - gold energy` with BCE and count-CE
  diagnostics; pair-head save/load in `decision_pair_head.pt`.
- Two fixes during the run: (1) the symmetric J matrix was built with in-place
  assignment on a fresh tensor, severing autograd (pair-head weights stuck at
  ~1e-3); fixed with `index_put`. (2) the pair head shared the 2e-5 LR, too small
  for J to reach the 1.5-4.5 target; added `--pair-lr-mult` (E3 used 50).

## Results (R regime, mean over seeds 43/44/45, DEV)

| setting | edge F1 (chance ~0.143) | joint NLL |
|---|---:|---:|
| full I/C/R data, 100 steps, lr 2e-5 | 0.147 | — |
| R-only, 150 steps, lr 2e-5 | 0.126 | 4.884 |
| R-only, 150 steps, pair lr x50 | **0.093** | 4.855 |

Edge F1 stays at or below chance regardless of data isolation or the pair-head
learning rate. Raising the pair LR does move the head weights (~1e-3 -> ~3e-2)
and slightly lowers joint NLL (4.884 -> 4.855), but the inferred matching does
not improve — the dense J learns a smooth coupling, not the sparse disjoint
matching.

## E3b — disjoint pair blocks (greedy + straight-through) also fails

Following the negative dense-J result, the loss was changed to a greedy
top-|w| disjoint matching whose selected weights flow through an autograd-safe
O(K^2) block log partition (`torch_pair_block_log_partition`).

| setting | edge F1 (chance ~0.143) |
|---|---:|
| dense J, full I/C/R, lr 2e-5 | 0.147 |
| dense J, R-only, lr 2e-5 | 0.126 |
| dense J, R-only, pair lr x50 | 0.093 |
| **pair blocks, R-only, pair lr x50** | **0.114** |

Edge F1 remains at chance. The pair-head weights stay near zero except for the
few selected pairs (|W|max ~3e-2, |W|mean ~1e-4), which is the signature of a
straight-through failure: the greedy matching is frozen by the zero
initialization, so only the initially-chosen pairs ever receive gradient and the
model cannot explore alternative matchings.

## E3c — Gumbel-perturbed straight-through matching also fails

Adding Gumbel noise to the greedy selection (so non-selected edges can be
explored, with gradients flowing through the un-perturbed weights) leaves edge
F1 at 0.14 — still chance. This rules out the "frozen initial matching" story as
the sole cause: even with exploration, the joint set CE gradient does not
recover the pairing.

| variant | edge F1 (chance ~0.143) |
|---|---:|
| dense J, full I/C/R, lr 2e-5 | 0.147 |
| dense J, R-only, lr 2e-5 | 0.126 |
| dense J, R-only, pair lr x50 | 0.093 |
| pair blocks, R-only, pair lr x50 | 0.114 |
| pair blocks + Gumbel noise, R-only | 0.140 |

## Root cause

Learning a discrete matching from a joint set objective does not work here under
any of the three selectors tried (dense J, greedy straight-through, Gumbel-
perturbed straight-through). The remaining explanation is that the joint set CE
gradient signal for the pairing is too weak at this data scale (the pairing only
shifts the joint law by ~0.84 nats of KL while the cardinality term dominates),
so the pair head cannot commit to the correct matching. A stronger inductive
bias — supervised matching, an explicit matching parameterization with entropy
regularization, or a pairwise-supervised loss — is the E3/E4 frontier. The exact
O(K^2) block DP (`paper_static/block_pair_dp.py`) is the intended inference once
such a selector exists.

## Interpretation

1. **A dense J head is the wrong inductive bias for disjoint pairings.** The true
   R law uses 4 disjoint pairs (a perfect matching), but a dense 28-parameter J
   can lower the joint CE by spreading small weights across many pairs without
   committing to the matching. Greedy top-disjoint-pairs extraction from a smooth
   dense J is then no better than chance.

2. **This is exactly the design note in the research doc**: the intended E3/E4
   architecture is *non-overlapping pair blocks* (an explicit matching), not a
   dense interaction matrix. The dense J path was tried first and is a controlled
   negative: it does not identify the matching.

3. **Next step (E3b):** parameterize the matching directly (pair blocks), and/or
   supervise the matching with the oracle edges during training, rather than
   recovering it post-hoc from a dense J. The exact O(K^2) disjoint-pair DP in
   `paper_static/block_pair_dp.py` is already the intended inference for that path.

## Cross-reference

- `docs/paper1_three_regime_research_design.md`: the intended pair-block architecture.
- `docs/paper1_e0e1_results.md`, `docs/paper1_e2_results.md`: the earlier stages.
- `harness/train_multiselect.py`, `harness/set_distillation.py`, `paper_static/pair_eval.py`: the dense-J implementation.
