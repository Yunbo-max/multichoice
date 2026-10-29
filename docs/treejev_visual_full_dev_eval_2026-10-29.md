# 2026-10-29 full-DEV reevaluation protocol

The existing 85.94% (seed45) came from first 64 DEV examples
with an unsafe batched answer-cue position. It is NOT a full-DEV score.

This branch now includes:
- treejev/visual_batching.py: token packing and checkpoint-safe cache repacking.
- treejev/eval_visual_action.py: full 3001-row DEV, trained adapter reload,
  optional train-layout overlap rejection, proper soft CE/Brier,
  stratification by number of tied optimal actions.
- Blocking 8-image check: the same adapter must agree between batch=1 and
  mixed-resolution batch=8; otherwise NO accuracy result is saved.
- qz_tree/tree-stage1-g2-s45-full-dev-fixed.json: single H100,
  170min max, fresh output (no deletion), branch preflight and no official test.

Run CPU:
    python -m unittest tests.test_tree_visual_batching tests.test_tree_visual_eval_metrics -v

Submit after checking out research/episodic-tree-sokoban-20261027
at the job's isolated code path and verifying existing adapter.
    qz train CreateJob --data "$(cat qz_tree/tree-stage1-g2-s45-full-dev-fixed.json)"

This payload is STAGED, NOT SUBMITTED in this change.
Uncached older training runs may need to be retrained, since they appended
cue after right padding. Single-image-cached G2 full 30k training was encoded
differently and is not automatically discredited by the specific bug.
