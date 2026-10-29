# TreeJev Stage 1 — Valen visual action SFT (G1 smoke + G2 three seeds)

Branch: `research/episodic-tree-sokoban-20261027`. Date: 2026-10-27.
Qwen3.5-4B multimodal LoRA action SFT on the Valen Sokoban tied-optimal action
distributions. Data prepared by `treejev.prepare valen`: 26,999 train / 3,001
dev rows (2,790 / 310 canonical boards, 3,911 multi-optimal train rows), official
`eval_sokoban` never downloaded or used. Each job is single-GPU H100.

## G1 smoke (gate check)

`python -m treejev.train_visual_action --limit 128 --dev-limit 16 --max-steps 8`:

- Nonzero-LoRA-gradient check passed (no "no nonzero gradient" error).
- Adapter saved and reloadable (`runs/tree/stage1_g1_smoke_s43/adapter_best`).
- best_dev `[0.125, 1.5464]` (8 steps only; smoke, not a result).

Two forward-path fixes were required before the smoke could pass: the answer cue
must also extend `mm_token_type_ids` (else Qwen3.5 3D position ids raise a shape
mismatch), and `torchvision` had to be installed for `Qwen3VLVideoProcessor`.

## G2 results (30k train, 250 optimizer steps, seeds 43/44/45)

`--limit 0 --dev-limit 64 --max-steps 250`, soft CE to the Valen tie-uniform
action distribution over up/right/down/left. Dev = held-out boards.

| seed | optimizer steps | action-optimal accuracy | soft CE | mass on optimal |
|---|---:|---:|---:|---:|
| 43 | 250 | 0.7188 | 0.7962 | 0.5906 |
| 44 | 250 | 0.6875 | 0.8596 | 0.5821 |
| 45 | 250 | 0.7188 | 0.8077 | 0.5298 |
| mean | — | 0.7083 | 0.8212 | 0.5675 |

## G2 batched results (batch 8, 2700 steps ≈ 21.6k images, seeds 43/44/45)

After switching from per-image to batch-8 forward (`--batch-size 8`, `padding=True`
for mixed-size Valen images), a single job can cover ~72% of the 30k train set
within the ~3h per-job limit.

| seed | optimizer steps | action-optimal accuracy | soft CE | mass on optimal |
|---|---:|---:|---:|---:|
| 43 | 2700 | 0.8281 | 0.4920 | 0.7397 |
| 44 | 2700 | 0.8438 | 0.6521 | 0.7815 |
| 45 | 2700 | 0.7969 | 0.5203 | 0.7616 |
| mean | — | **0.8229** | 0.5548 | 0.7609 |

## G2 full 30k cached result (batch 8, 3375 steps, seed 45)

Caching the processor features (`treejev.cache_valen_features`) removes the CPU
image-preprocessing bottleneck, so a single job exhausts the full 26,999-row
train set (3375 optimizer steps).

| seed | optimizer steps | action-optimal accuracy | soft CE | mass on optimal |
|---|---:|---:|---:|---:|
| 45 | 3375 | **0.8594** | 0.6178 | 0.7341 |

`action-optimal accuracy` counts a win whenever argmax over the four restricted
native logits is ANY tied optimal move. Random chance is roughly 0.25–0.29 (most
boards have a single optimal action; the multi-optimal fraction is ~15%).

## Interpretation

The LoRA action policy reaches ~86% held-out-board accuracy versus ~25–29%
chance on the full 30k train set, up from ~71% at 250 steps and ~82% at ~72% of
the data. This is a single-action softmax policy trained on all tied-optimal
moves, not the set-readout policy of Stage 3 and not any branching or RL. It
demonstrates that the multimodal processor path and the restricted A/B/C/D
next-token scoring are sound, and that the model learns a strong visual action
policy on unseen layouts.

Remaining Stage 1 questions (per `paper_tree/EXPERIMENTS.md` G2): comparison
against one-random-optimal CE and BCE-multihot-set, and the balanced by-`#optimal`
breakdown — still deferred.

## Cross-reference

- `treejev/train_visual_action.py`: the visual SFT entry point.
- `treejev/prepare.py`, `treejev/data.py`: Valen manifest and group-disjoint split.
- `paper_tree/EXPERIMENTS.md`: the Stage 1 protocol and priority gates.
