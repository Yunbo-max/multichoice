# Paper 2: Visual action padding/cache bug and verification plan (2026-10-29)

The old uncached batched path appended the answer cue after existing
processor padding, then indexed attention_mask.sum(-1)-1. For shorter
samples, this can select padding-area logits rather than the answer slot.

FIX: treejev/visual_batching.py strips existing padding per sample,
appends the answer cue, re-right-pads the batch, preserves multimodal
token-type IDs, and validates the expected last-token locations.
FeatureCache.get also removes precomputed per-chunk padding when merging
examples, verifies cached image IDs match label rows, and uses a bounded
LRU instead of directional chunk-number eviction.

IMPORTANT: previously cached full-30k TRAIN features were built one
image at a time, so their cue was already before chunk padding. This
does not prove the previously trained adapters are corrupted. Previous
uncached batched DEV evaluation does contain the padding issue, however.
Treat all reported 64-row DEV accuracies as provisional pending re-eval.

G0 CPU: python -m unittest tests.test_tree_visual_batching -v
G1 GPU: pair batch=1 and batch=8 logits on 8 images of different
resolution using exactly the SAME checkpoint and corrected evaluator.
G2 GPU: evaluate saved 250/2700/3375 step adapters on all 3,001
DEV examples; include breakdown by number of optimal moves.
G3 GPU: if corrected confidence/EM changes, retrain affected arms
with the fixed batch path. Preserve old logs/checkpoints.
G4: 3 full-seed results, new held-out game episodes, official TEST sealed.

No H100 job or independent result is claimed by this code patch.
