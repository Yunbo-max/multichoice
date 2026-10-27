# Paper 1 — Count-Anchored SetJev: locked experimental protocol (2026-10-27)

## Claim and scope
Main hypothesis: coupling the native candidate scores to a residual cardinality prior stabilizes continuation fine-tuning and improves set-valued decision calibration. It is NOT a new cardinality-potential family: independent count logits c_m and residual a_m are expressively reparameterizable via a_m=c_m-log Z_m(s). The empirical claim must concern optimization, gradients, and calibrated set decisions, not representational universality.

Current evidence: 4B BCE baseline 0.5615±0.0041 Dev exact-set EM (three seeds); Set CE with count anchor 0.5640±0.0057 versus same-loss independent count head 0.4659±0.0318, on a 5k warm-start continuation; induced-marginal anchor 0.5624±0.0006 versus 0.4977±0.0291. **These results use the same pre-trained BCE-S44 checkpoint for all warm starts** and do not show training from only 5k from scratch. Neither a matched BCE continuation nor final held-out Test has been run in this protocol.
Source: docs/seed_results_2026-10-23.md and docs/setjev_v2_gate3_3seed_results.md.

## Mathematical objective
Let s_i be native option logits, M(x) the publicly permitted counts, and
Z_m(s)=sum_{S:|S|=m} exp(sum_{i in S} s_i).
For any permitted subset S,
P(S|x)=exp(sum_{i in S}s_i+a_|S|)/sum_{m in M(x)} exp(a_m) Z_m(s).
In the independent-head control, P(S|x)=softmax(c)_|S| exp(sum s_i)/Z_|S|(s).
For anchored Set CE with fixed gold/teacher sufficient statistics (q_i, rho_m),
the unnormalized loss is log Z(s,a) - sum_i q_i s_i - sum_m rho_m a_m.
The full-anchor candidate-logit gradient is mu_i-q_i.
For independent count logits, it is sum_m rho_m P(i in S|m)-q_i.
The cross-output Hessian for full anchor equals Cov[1(i in S),1(|S|=m)].
This is an optimization-geometry mechanism, not a proof of superior accuracy.

New control --count-anchor-detach has EXACTLY the same forward law as --count-anchor but computes count logits using s.detach(): it removes the gradient from log Z_m(s) through the count prior. This differs from detaching every candidate term, and must be interpreted as a specific path ablation.

## Data, held-out protocol, and metric definitions
- Gold-source: CMB-Exam and MultiRC group-separated split in data/multiselect_grouped_multi/; train=28,751, DEV=2,042, TEST=2,140; passage groups and exact duplicates audited. Use the checked-in 5k subset only for matched continuation. A development split is not a held-out result.
- Primary all-arm comparison: same checkpoint runs/qz/grouped_bce_4gpu_s44/v1/final, 5k gold continuation, 4xH100 full-rank, 1 epoch / 79 updates, batch64, learning rate 2e-6, terminal-only DEV checkpoint evaluation, paired seeds 43/44/45. No extra Gold data for any arm.
- Report exact-set EM, micro and macro F1, option BCE/Brier/ECE, cardinality NLL/Brier/ECE, cardinality MAE and count-mode accuracy, proper gold-set NLL, calibration by option count/language/source/gold cardinality, cold/warm gradients, wall-clock and GPU-hours. Count-only class accuracy is NOT a calibration score.
- Use one final selected checkpoint per arm; before any Test inference freeze arm definitions, chosen learning rate and readout rule, and persist SHA256 hashes of train/dev/test and model identity. NEVER select on Test.
- Statistical analysis: 3 seeds for initial mechanism work; 5 seeds for main table when budget permits; paired seeds and per-question paired bootstrap confidence intervals at a common model seed. Report the paired difference, not only marginal standard deviations. Separate same-sample bootstrap from across-seed inference. Use grouped bootstrap for MultiRC passage groups.
- Label probabilities under a 2-of-K protocol do not automatically define a calibrated count forecast. For full constrained set NLL use the actual permitted-count law for every method; separately report unconstrained BCE metrics.

## Experiments in execution order
P0 (engineer): audited source/model/modules, same set-MAP decoder for structured arms, correct group-disjoint splits, count-anchor-detach gradient regression, train/dev/test provenance. Do not assume model weights are stored in git.
P1 (must run): all eight arms in paper_static/plan_experiments.py on 4B with identical S44 warm start: BCE continuation, independent count Set CE, anchor Set CE, detached anchor Set CE, independent/anchor/detached induced-marginal, and BCE+gold-count CE. Report both DEV EM and probability quality, not only winner-selected step.
P2 (mechanism): weight- and compute-matched stopgrad, cold vs BCE warm start, different learning rates, count-only head vs full fine-tune, same parameter budget, initial gradient cosine/norm, condition numbers on a small controlled model.
P3 (model size): verify Qwen/Qwen3.5-0.8B, -2B, -4B, -9B; pretrain matching BCE checkpoints and per-size exact module manifests; do NOT reuse 4B modules or 4B checkpoint with a different-size model. Baselines must receive the same hard-label pretraining, same downstream continuation and learning-rate search. 27B optional only after smaller sizes succeed. Full rank vs LoRA is a separate equal-budget question.
P4 (data/semantic generalization): 5k/10k/28k with fixed base and non-overlapping held-out splits; unseen option count K, gold count m, Chinese CMB vs English MultiRC, native single vs multi protocol; candidate permutation consistency; chronological or source OOD split, if a legitimate independent dataset is available. Keep tokenizer/readout and sampling stable across arms.
P5 (self-teacher): native option-logit teacher vs legacy candidate-Yes/No teacher, Gold-only versus hard-then-soft (same starting checkpoint/updates), exact-set KD versus moment-matched KD. Keep teacher quality, count bias, missing chains, and teacher calibration on a separate audit. Prior soft-only KD failed; do not call this confirmed.
P6 (external): locked Test once, plus comparable SII/OpenJev/StartLux-style direct-action baselines *where model weights, answer protocol, and licensing allow*. Compare protocol-matched single-choice with single-choice; not raw headline accuracies on different datasets.

## Mandatory ablations and likely rejection tests
- If anchor fails to beat BCE continuation, claim is at most mitigation of poor independent-count initialization; narrow scope.
- If detached anchor equals full anchor, the covariance/gradient-feedback mechanism is not empirically supported; re-evaluate explanation.
- If advantage disappears under matched LR tuning, attribute improvement to conditioning and optimization hyperparameters, not inherent expressivity.
- If count ECE / NLL / Brier worsen, do not call model calibrated solely because EM improves.
- If only 4B/one mixed-dev split gains, narrow contribution to that setting; collect an external dataset.
- Correlated label interactions not reducible to count cannot be modeled by this family; disclose limitation.

## Run, without automatic GPU submission
From a QZ checkout containing full checkpoint weights:
    python paper_static/plan_experiments.py --output paper_static/runs_4b.jsonl
    python -m unittest tests.test_static_anchor_gradient tests.test_static_plan -v
This creates a manifest with exact argv and SHA hashes if local files exist; submit selected argv on your real compute environment. A 4B module list is required. Generation does not start training and no new GPU results are claimed.
