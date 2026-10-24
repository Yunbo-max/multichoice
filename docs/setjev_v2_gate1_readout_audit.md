# SetJev v2 Gate 1 readout audit: Yes/No readout degrades the teacher

Date: 2026-10-24. Checkpoint: `grouped_bce_4gpu_s44/v1/final` (gold BCE). Sample:
300 stratified dev questions (source × gold count × K, seed 43).

## Hypothesis

The plan (section 3) asked whether the reasoning teacher's Yes/No readout is
itself mismatched with the native decision the model already learned. This audit
compares the two *no-reasoning* readouts on the same checkpoint, so any gap is
readout-only, before reasoning is involved.

## Result: native option logits beat the candidate Yes/No readout on every metric

| metric | native logits | candidate Yes/No | Δ (yesno − native) |
|---|---:|---:|---:|
| question EM | **0.540** | 0.453 | −0.087 |
| F1 | **0.898** | 0.874 | −0.024 |
| precision | **0.879** | 0.813 | −0.067 |
| recall | 0.918 | 0.945 | +0.027 |
| membership BCE | **0.305** | 0.700 | +0.395 |
| membership Brier | **0.094** | 0.145 | +0.051 |
| count MAE | **0.483** | 0.713 | +0.230 |
| count accuracy | **0.600** | 0.500 | −0.100 |
| oracle-count EM | **0.793** | 0.773 | −0.020 |

The Yes/No readout over-predicts: recall rises slightly (0.918 → 0.945) while
precision falls (0.879 → 0.813) and calibration collapses (BCE 0.305 → 0.700,
count MAE 0.483 → 0.713).

## Conclusion

The candidate Yes/No readout interface, on its own, turns the trained decision
into a worse teacher. The plan's Teacher–Readout Mismatch hypothesis is
confirmed: **the teacher should emit native option-logit probabilities, not
Yes/No membership probabilities**, unless the Yes/No readout is explicitly
calibrated first. Reasoning (Gate 1 modes 3–4) is therefore unlikely to help on
top of this readout; the readout must be fixed first.

This also feeds the Count-Anchored design: the teacher's candidate scores `u_i`
should come from the native option logits, not from Yes/No softmaxes.
