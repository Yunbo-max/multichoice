# TreeJev Stage 0/2/3 CPU end-to-end: symbolic grid baseline fits, does not generalise

Branch: `research/episodic-tree-sokoban-20261027`. Date: 2026-10-27.
CPU-only run (torch 2.8, no CUDA) of the exact-oracle data path, the lightweight
grid policy/value baseline, and fixed-budget beam control. No Qwen4B / LoRA
weights and no GPU are involved.

## Commands

```bash
python -m unittest tests.test_tree_sokoban tests.test_tree_prepare tests.test_tree_grid_model -v
python -m treejev.prepare oracle --maps maps_train.jsonl --output-dir runs/tree/oracle_train
python -m treejev.train_grid_policy --train runs/tree/oracle_train/train_multi.jsonl \
  --dev runs/tree/oracle_train/dev_multi.jsonl --output runs/tree/grid_policy --epochs 40 --seed 43
python -m treejev.eval_grid_beam --checkpoint runs/tree/grid_policy/best.pt \
  --maps maps_heldout.jsonl --output runs/tree/beam_eval.json \
  --widths 1 2 4 --node-budget 64 --horizon 24
```

## Training maps (5 levels, all certified shortest DAG)

| level_id | distance | optimal states | shortest paths |
|---|---:|---:|---:|
| demo | 3 | 4 | 2 |
| line2 | 2 | 2 | 1 |
| corner3 | 5 | 9 | 2 |
| zig4 | 6 | 11 | 3 |
| pushdown | 5 | 9 | 2 |

ASCII sources (JSONL, `level_id` + `ascii` list of rows):

```
demo      ####### / #@    # / #  $. # / #     # / #######
line2     ######  / #@ $.# / #    # / #    # / ######
corner3   #####   / #@  #  / # $ # / #  .#  / #####
zig4      ####### / #@    # / #  $  # / #   . # / #     # / #######
pushdown  ######  / #@   # / # $  # / #  . # / #    # / ######
```

Held-out levels (never seen in training): `held_l4` (distance 5, 101 states),
`held_z5` (distance 7, 237 states).

## Results

### Stage 0 — oracle
`runs/tree/oracle_train/summary.json`: 5/5 levels certified, 26 train states,
9 dev states, 10 total shortest trajectories, zero skipped, `official_test_touched=false`.

### Stage 2 — grid policy/value (40 epochs, CPU)
| dev metric | value |
|---|---:|
| optimal-action accuracy | 0.4444 |
| action CE | 1.132 |
| distance MAE | 1.121 |
| train loss (final) | 0.967 |

`runs/tree/grid_policy/training.json` records the full 40-epoch history. The model
fits the 26 training states but reaches only 4/9 correct dev states.

### Stage 3 — fixed-budget beam control (widths 1/2/4)
| level_id | width | won | steps | expanded |
|---|---|---:|---:|---:|
| held_l4 | 1 | false | 0 | 24 |
| held_l4 | 2 | false | 0 | 47 |
| held_l4 | 4 | false | 0 | 64 |
| held_z5 | 1 | false | 0 | 24 |
| held_z5 | 2 | false | 0 | 47 |
| held_z5 | 4 | false | 0 | 64 |

`runs/tree/beam_eval.json`: `uses_exact_simulator=true`, `valid_arc3_eval=false`.

## Interpretation

The symbolic grid baseline is an audited control, not the proposed 4B latent
world model. It learns tied-optimal action probabilities on the 5 training maps
but fails to guide beam search to a win on either held-out map, regardless of
beam width. This is the expected outcome for a tiny CNN trained on five layouts
and does not represent a latent-branching result: it merely confirms that the
Stage 0 oracle and Stage 2/3 control plumbing run end-to-end and that grid-level
memorisation does not transfer to unseen boards.

The remaining work is Stage 1 (Qwen3.5-4B LoRA action SFT on Valen Sokoban) and
Stage 4 (latent branching RL with independent branch memory), both of which need
GPU and are out of scope for this CPU verification.

## Cross-reference

- `treejev/prepare.py`, `treejev/train_grid_policy.py`, `treejev/eval_grid_beam.py`: the audited control entry points.
- `tests/test_tree_sokoban.py`, `tests/test_tree_prepare.py`, `tests/test_tree_grid_model.py`: the CPU contract tests (all pass).
