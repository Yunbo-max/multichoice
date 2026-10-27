# Paper 2 — Episodic SetJev-Tree: Text/Grid -> Image branching (2026-10-27)

## Scientific question
Can a 4B decision-native policy learn to preserve several viable action branches,
estimate the value of each independently, execute a single real action, ingest its
actual feedback, and stop when the environment verifies completion—under a fixed
node/action/compute budget? A naive BFS is not new. The novel hypothesis is
**adaptive latent branch allocation under action and observation uncertainty**.

This is an EXPERIMENTAL implementation branch. No new VLM training, held-out
game success, or learned latent world model is claimed. Previous Episodic GRU
results near chance and single-step unseen-state 0% must not be labeled solved.

## Public datasets and contamination boundaries
- Valen-Team/Valen-Eval-Game (Apache-2.0): TRAIN 30,000 image-action questions
  from 3,100 layouts; 4,378 have tied optimal next moves. Train has 24k optimal
  states, 4.5k solvable deviations, 1.5k pre-failure corrections. Official
  100 games / 500 action questions are TEST-only, and share the same layouts.
  The 100 games are a selected subset, so also report independent Boxoban.
  Image paths are relative to train_sokoban/ after extracting assets.zip.
  Only visible request and image may reach model; symbolic states/solver
  distances, failures and meta are teacher/audit metadata, never VLM input.
- AlignmentResearch/boxoban-astar-solutions: many A* reference action paths,
  not necessarily ALL equally-optimal successors, and some search failures.
  An unsolved record under a bounded search is NOT a mathematical deadlock.
  Download only train/validation files with matching Boxoban level maps,
  and certify labels via a separate controlled search.
- BabyAI/MiniGrid: continuous partial-observation episodes; one expert action
  trajectory is not an all-optimal-action label.
- PlanBench Blocksworld: text/PDDL held-out transfer; solve under a deterministic
  planner for all-shortest controllable labels.

## Stage 0: exact CPU oracle and interfaces — IMPLEMENTED
treejev/sokoban.py: immutable, verified environment transitions.
treejev/oracle.py: bounded shortest-path BFS DAG, ALL tied optimal next actions
on shortest paths, exact remaining distance for every shortest-path node, and
up to N shortest successful trajectories. Over-budget searches RAISE and
cannot silently produce incorrect labels.
treejev/data.py: Valen target parsing (no privileged meta exposure) and
canonical-board-disjoint train/DEV split.
treejev/beam.py: fixed-beam baseline uses simulator for TRAINING and DEBUG only;
this simulator is NOT permitted as an internal oracle in live ARC-AGI-3.
Every imagined branch requires separate per-branch memory when a learned model
is added. Only one real environment action may be executed at a time.

## Stage 1: one-step multiple-optimal-action SFT — CODED, UNRUN ON GPU
Visual:
    hf download Valen-Team/Valen-Eval-Game --repo-type dataset --local-dir data/Valen-Eval-Game
    unzip -n data/Valen-Eval-Game/train_sokoban/assets.zip -d data/Valen-Eval-Game/train_sokoban
    python -m treejev.prepare valen --train-jsonl data/Valen-Eval-Game/train_sokoban/train.jsonl --output-dir runs/tree/valen_split
    python -m treejev.train_visual_action --train runs/tree/valen_split/train_visual.jsonl --dev runs/tree/valen_split/dev_visual.jsonl --model /path/to/Qwen3.5-4B --modules innerjev/configs/4B/modules.json --output runs/tree/stage1_s43 --limit 128 --dev-limit 16 --max-steps 8

Training objective: CE to Valen tie-uniform distribution over up/right/down/left.
All four restricted native next-token logits form a single-action policy.
Argmax receives credit whenever it is ANY optimal action. Never execute all
predicted optimal actions simultaneously.
Compute: one H100 80GB for 4B VLM LoRA smoke; 24GB text oracle pilot.
The multimodal training path depends on actual Qwen3.5 processor/weights and
must pass one H100 smoke before launch. Stage 1 is neither RL nor branching.

Text grid (runs today using existing Qwen text harness after downloading weights):
    python -m treejev.prepare oracle --demo --output-dir runs/tree/demo
    python -m unittest tests.test_tree_sokoban -v
For actual training use --maps /path/to/independent_train_levels.jsonl with many
different board groups; a one-level demo has NO DEV and is never a generalization
experiment. Run existing harness/train_multiselect.py with --train
runs/tree/oracle/train_multi.jsonl and --dev
runs/tree/oracle/dev_multi.jsonl, --target gold --objective bce or set;
it learns the SET of optimal next moves, not a single-action softmax.
At online execution choose only one action; group by board layout.

## Stage 2: trajectory and search-value targets — ORACLE GENERATED
From an exact DAG, produce per-state optimal actions, remaining shortest
distance, and alternative shortest successful trajectories, with an
independent immutable state per child. For large Boxoban, bounded forward BFS
is often intractable: start tiny and use an audited A*/IDA*/deadlock pruning
solver, never call incomplete bounded search exact.
A conditional value head V(s,b) should predict reachable success at budget b.
Training from optimal-path-only states is insufficient to learn dead ends:
add certified recoverable off-path states and certified failures. Never label
a search timeout as unsolvable.

## Stage 3: learned tree policy and breadth — RESEARCH TODO
P(a|z_n), V(z_n,b), and branching set P(B|z_n) using Count Anchor;
save independent branch memory z_n, not a shared GRU overwritten by siblings.
Compare Greedy, Best-of-N, FIFO BFS, A*, fixed beam 1/2/4/8, same-budget
value-guided beam, and action-input-matched history transformer. For the first
prototype, exact simulator may be used for **offline search labels**, not for
hidden-state inference in the official game evaluation.

## Stage 4: on-policy budget-aware search RL — RESEARCH TODO
Train a branching/stopping controller on real rollouts, with
R = 1[WIN] - lambda_real N_real_actions - lambda_internal N_expanded.
'Stop searching and execute' is different from 'game ended': only environment
WIN can declare success. ARC-AGI-3 may not permit state cloning and actions
are limited; latent rollout must use an independently trained world model.

## Primary experiment matrix and evaluation contracts
Backbone: Qwen3.5-4B (primary), 2B cheap control, 9B scaling after 4B passes.
Same pretrained checkpoint/LoRA budget in all action-policy controls. Same
train/DEV splits, image resolution and language prompts. Seeds 43/44/45.
Stage1: Valen tied-action CE vs one-random-optimal CE, BCE-multihot-set,
direct Qwen, frozen Qwen head, and Valen published pretrained checkpoint
(where comparison protocol permits). Evaluate tied-optimal action hit,
proper action CE, illegal-action rate, balanced by #optimal choices,
unseen BOARD layouts (not random states on same board).
Stage2: trajectory-completion success, steps-to-win / optimal distance ratio,
value ranking for positive and certified negative sibling branches, recovery
after off-path moves and symbolic-state oracle upper bound.
Stage3: Solved@{32,64,128} expanded nodes, GPU-second/action, real game
actions, branching count, search stop calibration; same budget across all
algorithms and no copied hidden states at evaluation.
Stage4: hold-out Boxoban boards and new sizes then BabyAI/ALFWorld;
ARC-AGI-3 is a final interactive external test, not a training oracle.
Report independent frozen TEST only once.

## Priority gates
G0: run treejev CPU unittest, replay demo shortest routes, verify tied labels.
G1: Valen 128-image GPU smoke with nonzero LoRA grads, exact adapter reload.
G2: Valen 30k action SFT three seeds and held-out group DEV; compare all-optimal
soft CE versus one sampled expert. Keep official 100 game test sealed.
G3: Qwen action SFT plus value ranking on *verified* off-path alternatives.
G4: learned branching under matched expanded-node budgets.
G5: only after G4 demonstrates benefit, integrate visual ARC-AGI-3.

## Data safety follow-up (2026-10-27)
Oracle data now uses canonical wall/goal board geometry under dihedral rotations/reflections as the group_id (not merely an arbitrary level ID). Same-layout differently seeded states cannot cross train/DEV. The official Valen group_id remains the authoritative grouping for its image records. The visual SFT path uses the multimodal Qwen3.5 processor for image pixel tensors, then appends the answer cue to input_ids/attention_mask before selecting the native A/B/C/D logits.

## Executable Stage 2/3 symbolic learned-search control
The following scripts are CPU-capable, independent of the 4B visual branch:
    python -m treejev.train_grid_policy --train runs/tree/oracle/train_multi.jsonl --dev runs/tree/oracle/dev_multi.jsonl --output runs/tree/grid_s43 --epochs 10 --seed 43
    python -m treejev.eval_grid_beam --checkpoint runs/tree/grid_s43/best.pt --maps data/heldout_ascii_levels.jsonl --output runs/tree/grid_beam_eval.json --widths 1 2 4 8 --node-budget 128
    python -m unittest tests.test_tree_grid_model -v
This is an initial learned policy and remaining-step regression baseline; it uses the
TRUE Sokoban simulator inside search. It is not a learned world model, a
trained 4B branching controller, or a legal live ARC-AGI-3 search. No evaluation
should mix held-out layouts with train/dev. Value training only on optimal-DAG
states lacks certified off-path negatives; add solver-labeled recovery states
and matched best-first/A* controls before claiming learned branching gains.
