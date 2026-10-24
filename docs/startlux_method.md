# StartLux Decision (AutoThink): method summary for comparison

Source: `decision_compare/startlux-decision` (Apache-2.0), README + `startlux_decide/`.
This is a factual summary of the StartLux method, written so our SetJev v2
teacher design can be compared against it side by side.

## 1. What it is

StartLux-Decision is a family of "decision models" (0.8B–27B) that answer
**decision tasks** through an *answer slot*: the model reads a rendered question
with a list of candidate options and returns a probability distribution over
those options, plus a confidence, instead of free text. Two flavours:

- **Instant**: answers in one pass, never reasons. Prompt attention is
  **bidirectional** (a custom vLLM plugin), so the single read-out attends to the
  whole question and all options at once.
- **AutoThink**: adds a reasoning gate — it reasons (writes a short thought) only
  when its first read-out is not confident.

## 2. Decision formalism (task types)

Every task is reduced to a probability over a closed option set:

| type | output |
|---|---|
| `choice` | softmax over listed options (single argmax answer) |
| `noul` | binary probability (yes/no over one criterion) |
| `score` | softmax over an ordered legend (0..n-1) + a numeric score |

The interface is a letter/logit **answer slot**: the engine reads the log-prob of
each option's letter, not free text.

## 3. The readout

From `readout.py`: a read-out returns one log-probability per option letter.
`probabilities(z, T) = softmax(z / T)`. The first read-out uses the task type's
temperature; the post-thought read-out is re-tempered (`retemper`) and mixed with
the first.

For `choice` lists over 26 options there is a **knock-out** stage: split into
≤25-option groups, read each group in one pass, keep the top finalists, then a
final round. Knocked-out options keep a small residual probability share.

## 4. AutoThink routing (the "route to short thinking" decision)

From `pipeline.py:334–345`, per question:

1. **First read-out (short thinking)**: build the one-pass prompt, read letter
   probabilities `p0` (no reasoning).
2. **Gate**: reason iff
   `think_mode == "always"` or (`think_mode == "auto"` and `max(p0) < gate`),
   and there is context room.
   - `max(p0) ≥ gate` → **route to short thinking**: answer straight from `p0`.
   - `max(p0) < gate` → **route to long thinking**: generate a thought, read
     again `p1`, combine.
3. **Second read-out**: after the thought (its tail characters, marked `[...]`),
   re-tempered with the think temperature, then mixed:
   `p = (1 - mix) * p0 + mix * p1`.

Controls: `think_mode ∈ {auto, never, always}`; `gate` default 0.6 (published
0.9); `think_budget` default 4096 tokens; `mix` blends the two read-outs.

## 5. The thought (reasoning trace)

The thought starts with a **forced prefix** and is generated with a fixed seed /
sampling / stop string / budget. A **loop detector** cuts a repeating thought and
ends the request. A thought that is not closed by the model nor cut as a loop is
cut with a hard line (`\n\nI have to decide now …`). So the reasoning is bounded,
detectably-terminated, and never unbounded.

## 6. Training: RLCD²

The reasoning is trained with **RLCD² — reinforcement learning for calibrated
deliberative decisions**. Instant models are supervised; AutoThink models are
trained so the *deliberation* improves calibrated decision quality (DecisionPivot
pair accuracy is the headline metric: pairs differing in one decisive detail,
counted only when both answers are correct).

## 7. Reported behaviour

- Instant: 25–52 ms median per request (one H200, vLLM).
- AutoThink: 38–59 ms when sure (one read-out); when unsure it writes a thought
  (longer the more it writes) and reads a second time.
- AutoThink adds 11.5–18.4 DecisionPivot pair-accuracy points over Instant; the
  gate trades speed for accuracy (published results use gate 0.9).

## 8. Key facts for our comparison

| axis | StartLux | SetJev v2 |
|---|---|---|
| decision object | single choice / binary noul / score | **multi-select subset** |
| option space | ≤26 (knock-out beyond) | ≤ K, each option an independent Bernoulli |
| model attention | bidirectional prompt (Instant) | **hybrid linear-attention (SSM), causal** |
| readout | softmax over letters | sigmoid over answer-slot labels (255) |
| reasoning trigger | `max(p0) < gate` (single top prob) | set-level uncertainty (min margin / count entropy) |
| reasoning structure | single thought trace → re-read | two-phase count→membership over subset DAG |
| aggregation | linear mix `(1-mix)p0 + mix p1` | refine u,t → cardinality-potential `Q(S|x)` |
| count / set coherence | none (independent per task) | **explicit count head + set partition** |
| training | RLCD² | BCE warm start → set CE / induced / count-anchor |

The decisive differences are (1) multi-select vs single-choice, (2) causal linear
attention vs bidirectional transformer, and (3) an explicit count/set coherence
mechanism, which StartLux does not have.
