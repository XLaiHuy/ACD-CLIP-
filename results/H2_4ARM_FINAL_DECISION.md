# H2 four-arm factorial decision

Status: `MEDICAL_COMPLETE_MVTEC_COMPLETE`

The frozen exact Medical evaluation completed for H/A/C/AC at E15 and E20.
E15 is primary; E20 is secondary. The metrics are raw exact values from
`benchmark_exact`-equivalent evaluation with `pixel_stride=1`, fixed
`current_shared` prompts, `cls_only` image scores, and no checkpoint
selection or tuning.

## Primary E15 rule application

The preregistered tolerance is `tol=1e-6` on raw metrics. Pareto support on
pixel metrics requires both pixel AUROC and pixel AP to be no lower than the
reference (within tolerance), with at least one strict improvement.

| comparison | pixel AUROC delta | pixel AP delta | Pareto result |
|---|---:|---:|---|
| A - H | 0.4364540284135927 | 3.594116817186361 | PASS |
| C - H | -0.5950622009201112 | -1.3813161418203634 | FAIL |
| AC - H | -0.01404613100419283 | 1.179916205552395 | FAIL |
| AC - A | -0.45050015941778554 | -2.414200611633966 | FAIL |
| AC - C | 0.5810160699159184 | 2.5612323473727585 | PASS |

Primary classifications:

- `ANCHOR_SUPPORT=PASS`
- `CIR_SUPPORT=FAIL`
- `AC_SUPPORT=FAIL` against H (AC improves over C but does not Pareto-beat H or A)
- `INTERACTION=NEUTRAL`: the E15 pixel interaction is positive for pixel AUROC
  (`0.14456204150232566`) and negative for pixel AP
  (`-1.0328844698136024`), so it is not directionally consistent.

The fixed minimal-winner rule therefore selects:

`FINAL=A`

This follows the preregistered case “If A > H and C fails: FINAL = A.” The
E20 extension does not rescue or overturn a failed E15 comparison.

## Secondary E20 evidence

E20 supports the same qualitative mechanism pattern only partially: C fails
against H, AC improves over C, and AC does not beat H or A. A improves pixel
AP but loses pixel AUROC to H, so it is not a Pareto pass at E20. Pixel
interaction remains mixed (AUROC `0.3634074555381801`, AP
`-0.15238149021411118`).

E20 is reported as longer-horizon supporting/contradicting evidence only; it
was not used to alter the E15 decision.

## Source artifacts

- Primary summary: `H2_4ARM_E15_MEDICAL_SUMMARY.csv`
- Secondary summary: `H2_4ARM_E20_MEDICAL_SUMMARY.csv`
- Per-dataset tables: `H2_4ARM_E15_MEDICAL_PER_DATASET.csv` and
  `H2_4ARM_E20_MEDICAL_PER_DATASET.csv`
- Factorial contrasts: `H2_4ARM_FACTORIAL_EFFECTS.csv`
- Evaluation provenance: `H2_4ARM_MEDICAL_EVAL_MANIFEST.json`

The matched industrial transfer check is complete and must not be used to
retune or change this selection.
