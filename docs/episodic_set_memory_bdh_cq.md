# Episodic SetJev — persistent memory across changing decisions

**Branch:** research/episodic-set-memory-20261024  
**Parent:** research/recurrent-set-debug-20261024 at 735bc69b83414e12ba80407d2ddd0865a17ca2f2  
**Paper:** Engdahl et al., [BDH-CQ: In-Context Learning with Recurrent Latent Reasoning](https://arxiv.org/abs/2608.09888), 2026, especially Sec. 3.2, 3.3, 4.1, 6.

## 1. What is actually from BDH-CQ?

The paper specifies TWO conceptually distinct states:

- Recurrent contextual memory: S_t = U(S_(t-1), D_t) absorbs demonstrations.
- Query-time reasoning workspace: H_0 = E(x, S_K);
  H_(r+1) = F(H_r, S_K); predicted output = G(H_R).

The exact update rules, state dimensions, and training recipe are expressly
proprietary (Sec. 3.3–4.1). The repo pathwaycom/bdh is not the published
BDH-CQ ARC solver, and pathwaycom/arc-task-gen produces evaluation tasks.

This branch is a **new BDH-CQ-inspired episodic multi-set policy**, not an
official BDH-CQ reproduction and not an attempt to reproduce 29.5% ARC pass@2.

## 2. What was wrong with our previous code?

The old toy rollout called model.forward(context_t, ...) separately for every
environment step. Each call initialized an entirely new latent workspace and
no persistent memory was carried between steps. The env state changed, but
the policy did not keep a trainable summary of prior observations.

That only tests repeated *within-query* computation, not the user's requested
sequence of choices as the prompt/environment evolves.

## 3. New causal state machine

Let e identify an episode; t its environmental decision step and r the
internal reasoning iteration.

    persistent S_(e,0) = 0
    S_(e,0) = ingest_known_demonstrations(S_(e,0), D_e)

    for t=0,...,T-1:
      H_(t,0) = Init(current_state_features_t, S_(e,t))
      H_(t,r+1) = WithinQueryGRU(H_(t,r), S_(e,t), set_belief_(t,r))
      P_(t,r) = CardinalitySetReadout(H_(t,r))
      A_t = MAP(P_(t,R))    # one or multiple selected actions
      feedback_(t+1) = environment.execute(A_t)
      S_(e,t+1) = MemoryGRU(
          S_(e,t), state_features_t, executed_actions_t, feedback_(t+1)
      )
      if done: reset S_(e,t+1) to zero

Only the persistent S crosses the env-step boundary. The per-query H is
freshly initialized at each decision, conditioned on S. Qwen text generation
is NEVER required. The existing Count-Anchored readout / exact set decoder
remains untouched.

**Causal ordering is mandatory:** A_t is chosen before feedback_(t+1) exists.
The gold target, future observations and future demonstrations MUST NOT enter
decide() or observe(). Teacher forcing may execute oracle actions DURING TRAINING
to diversify trajectories, but inference uses policy-selected actions only.

Implementation:

- harness/recurrent_set_decision.py: backwards-compatible memory_state
  argument to core forward(), initial_memory for its demo GRU.
- harness/episodic_set_decision.py: explicit empty_memory(), reset(),
  prime(), decide(), observe(). No module-global episode cache.
- harness/train_episodic_set.py: differentiable whole-episode supervised
  training with scheduled teacher forcing and on-policy evaluation.
- harness/episodic_qwen_bridge.py: frozen Qwen3.5-4B features, logits,
  feedback embeddings, semantic action descriptions.
- tests/test_episodic_set_decision.py: causal, reset and BPTT regression tests.

### Changing option menus

For fixed-index actions (grid cells), memory can consume the selected multi-hot
mask. For changing action descriptions where A/B/C may refer to different
actions at different times, set

    require_semantic_actions=True

and supply option_features [B,K,H] to observe(). It pools the embedding of
the ACTUALLY selected options, not the alphabetical label positions. This
transition is permutation-invariant under matched reordering of options,
features and selected masks. The *within-query* base SetJev still uses
label/position readout and should be tested with order permutation augmentation.

### Episode boundaries

A batch member starting a new episode must call reset(S, episode_start).
After terminal env feedback, observe(..., done=mask) resets those members.
A new decision without reset would be a data-leakage risk.

## 4. Training objectives and what they mean

Each t uses the exact-set NLL:

    L_t = -log P_(t,R)(S_gold_t | current_state_t, memory_t)

The target is computed on the *actual* current state reached by expert or
sampled policy execution. The memory GRU and demo encoder receive gradients
through the complete episode (BPTT); no detach between t steps unless
explicitly chosen. Discrete action selection is not differentiated through.

This is **supervised interactive data aggregation**, NOT GRPO/PPO, RLCD2,
RLVR or a proven Decision Policy Improvement method. It is the prerequisite
for studying longer-horizon reward methods.

The model was deliberately split into two controls using identical trained
weights and decoder/compute budget:

- carry: persist S_t throughout each episode;
- reset_every_step: prime at t=0, then discard S_t before all later decisions.

A positive result on this controlled probe means memory access helped on
that environment, not that we reproduced BDH-CQ or solved ARC.

## 5. Two synthetic mechanisms, both causally verifiable

(1) **Rule retention:** a hidden cyclic rule is shown through ONE demonstration
only at episode start; the correct action is a set of rotated active cells.
Subsequent states do not reveal the original rule. The policy has to retain
the demonstration beyond t=0.

(2) **Rule change:** with --rule-switch-prob, the environment sometimes
replaces its hidden rule *after* the agent has acted, supplies a new
input-output demonstration in the post-action observation, and expects
subsequent decisions to use the updated rule. This tests actual updating
of S_t, not just retention.

In both cases the state at t+1 is the **executed** selected set; it is not
silently replaced with the oracle answer during evaluation.

This is an intentionally small synthetic mechanism probe with a tiny finite
rule family; it cannot establish ARC generalisation by itself.

## 6. Reproduce locally / on a QZ CPU-GPU worker

From repository root, PyTorch must be installed:

    python -m unittest discover -s tests -p 'test_episodic_set_decision.py' -v
    python -m unittest discover -s tests -p 'test_recurrent_debug_regressions.py' -v

Small CPU/CI pilot:

    python harness/train_episodic_set.py train \
      --output runs/episodic/smoke_s43 --device cpu \
      --epochs 2 --train-batches 2 --eval-batches 2 --batch-size 8 \
      --latent-dim 32 --inner-steps 2 --horizon 4 --rule-switch-prob 0.5

Suggested controlled experiment (run seeds 43/44/45 separately):

    python harness/train_episodic_set.py train \
      --output runs/episodic/full_s43 --seed 43 --device cuda \
      --epochs 30 --train-batches 80 --eval-batches 32 \
      --batch-size 32 --latent-dim 128 --inner-steps 4 --horizon 6 \
      --rule-switch-prob 0.25

Evaluate a **separate seed**, including a longer horizon:

    python harness/train_episodic_set.py evaluate \
      --checkpoint runs/episodic/full_s43/best.pt --device cuda \
      --seed 3043 --horizon 10 --eval-batches 64 --batch-size 32 \
      --rule-switch-prob 0.25

Also run --rule-switch-prob 0 and 1. Report first-decision EM,
later-decision EM, full-trajectory exact success and Set NLL.
The trained model should outperform its identical reset_every_step control
in later decisions/trajectory success under equal inference compute.
A one-seed or tiny-toy win is not a paper-level result.

### Live Qwen inference sketch

    encoder = FrozenQwenDecisionEncoder(PATH_TO_GOLD_BCE_EXPORT)
    config = EpisodicSetConfig(
        context_dim=encoder.hidden_size,
        max_options=32, latent_dim=128, inner_steps=4,
        require_semantic_actions=True,
    )
    agent = EpisodicSetDecision(config).to(encoder.device).eval()
    memory = agent.empty_memory(1, device=encoder.device)
    memory = agent.prime(memory, encoder.encode_demonstrations(KNOWN_EXAMPLES))
    # At every environmental time step:
    context, logits, mask = encoder.encode_question(CURRENT_ROW)
    option_features = encoder.encode_option_meanings(CURRENT_ROW)
    history = agent.decide(memory, context, logits, mask)
    selected = decode_MAP(history[-1])              # from set_map()
    feedback = ENVIRONMENT.execute(selected)        # only now observable
    obs = encoder.encode_observation(str(feedback))
    memory = agent.observe(
        memory, context, selected_multi_hot, mask, obs,
        option_features=option_features, done=terminal_mask
    )

This is a **wiring example**, not a trained Qwen episodic checkpoint or a
complete production agent. The episodic head must be trained on real
sequential trajectories before its predictions can be trusted.

## 7. Explicit non-claims / remaining gates

- BDH-CQ exact architecture and weights are unavailable.
- Qwen backbone remains frozen here; the included episodic *toy* runner does
  not run Qwen and makes no claim of 4B learning.
- The live Qwen bridge is not yet an end-to-end logged-trajectory trainer.
  Existing CMB/MultiRC static rows cannot be treated as real episodes by
  arranging random questions in order.
- Real episodes require accurate state, action, post-action observation,
  candidate identity, endpoint and reset metadata.
- Scheduled sampling uses nondifferentiable selected sets; long-horizon RL
  requires a different, explicitly justified training objective.
- No GPU/H100 results have been executed or pushed on this branch.
- The existing unshared MLP control is not parameter/FLOP matched; use the
  same trained model's carry-vs-reset comparison first.

## Citations

- Engdahl et al. (2026) https://arxiv.org/abs/2608.09888
- Pathway BDH source https://github.com/pathwaycom/bdh
- StartLux-Decision https://github.com/StartLuxLabs/StartLux-Decision
- SetJev baseline: docs/setjev_v2_gate3_warm_results.md
