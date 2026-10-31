# Paper 1 — E4 K=16: anchor's cardinality advantage grows with K

Branch: `research/static-setjev-paper-20261027`. Date: 2026-10-30.
Extends the E2 cardinality result to K=16 candidates, using an O(K^2) log-partition
generator (`paper_static/controlled_c_large.py`) instead of 2^K enumeration. This
tests whether the anchor's count correction still beats BCE when the count support
widens.

## Setup

- C-regime synthetic data, K=16, cardinality correction
  `a_m = -strength * (m - center)^2`, center in {1.8, k*0.48, k*0.75}.
- Qwen3.5-4B LoRA, batch 32, 100 steps, seeds 43/44/45; anchor (`--count-anchor`)
  vs BCE continuation from a shared warm BCE.
- Metric: mean count KL to the oracle count law (DEV).

## Results (mean count KL over seeds)

| arm | seed 43 | seed 44 | seed 45 | mean |
|---|---:|---:|---:|---:|
| count anchor | 0.805 | 0.798 | 0.775 | **0.793** |
| BCE | 1.158 | 1.040 | 0.998 | 1.065 |

Anchor beats BCE by ~0.27 nats on count KL in every seed.

## Scale trend

| K | anchor count KL | BCE count KL | anchor − BCE |
|---|---:|---:|---:|
| 8 (E2) | 0.278 | 0.333 | −0.055 |
| 16 (E4) | 0.793 | 1.065 | **−0.27** |

The anchor's cardinality advantage widens with K: as the count support grows, the
independent-sigmoid BCE's Poisson-binomial law drifts further from the true count
distribution, while the anchored count head keeps tracking it. This is the
"when to upgrade from independent to count modeling" evidence at larger scale.

## Cross-reference

- `docs/paper1_e2_results.md`: the K=8 controlled result.
- `paper_static/controlled_c_large.py`: the O(K^2) C-regime generator.
- `paper_static/c16_count_kl.py`: the count-KL evaluator.
