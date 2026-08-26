# P31 Fast Objective-Level Scientific Screen

Status: `PASS`

Decision: `KEEP CURRENT`

Terminal: `FAST_SCREEN_COMPLETE_KEEP_CURRENT_NO_NEW_STAGE2`

This is a read-only screen over immutable P27/P29/P30/P30R1 artifacts. It is
not a new scientific execution, prediction, scoring run, or method-selection
sweep. P31 is the already qualified native/zero-adapter control.

## 1. Scope and entry validation

P31 engineering qualification is valid:

- qualification: `research/sabra_v2/region_distill/P31_ENGINEERING_QUALIFICATION.json`
- qualification status: `PASS_TO_SCIENTIFIC_PROTOCOL`
- authoritative preregistration SHA-256: `f42f0add36c0de2e303e6f25b0d48b63c33eda7d4c56d2a7ccb368ca76c865e3`
- focused tests: `15/15` passed
- synthetic cases: `15/15` passed
- production/reference max absolute error: `0.0`
- branch at entry: `research/p29r1-fast-objective-forensic-v1`
- worktree at entry: clean

The inherited causal finding is `TEACHER_DIRECTION_NOT_CAUSAL`, with
`SPARSE_SELECTIVE_CORRECTION` secondary. P31 asks the prior causal control
question: is teacher intervention needed at all? It is not a new attempt to
repair direction or ranking metrics.

Current-turn operations read reports, JSON, and frozen metric CSVs only. No
held GT/mask files, prediction tensors, MVTec/Medical data, model, cache
rebuild, or model forward was used.

## 2. Frozen comparison and primary result

The exact P31 identity output is native by construction:

```text
R_P31 = 0
A_P31 = A_native
Delta_P31 = 0
```

The already frozen candle comparison is:

| method | pAP | pAUROC | source |
|---|---:|---:|---|
| P31 native control | 0.514140304931 | 0.980667143514 | `P30R1/candle/metrics/P30R1_HELD_METRICS.json` |
| P30R1 learned intervention | 0.511513734224 | 0.980534708954 | same frozen metric artifact |
| P31 − P30R1 | +0.002626570707 | +0.000132434559 | derived from frozen scalars |

P31 is exactly neutral versus native (`0.0` pAP and `0.0` pAUROC), and it is
slightly better than the existing P30R1 intervention on both frozen candle
metrics. This supports the scoped null/control claim. It does not claim an
improvement over native and does not authorize another candle run.

## 3. Strongest baseline and cross-category reference

The prior 12-category evidence is retained as context, not recomputed:

| method | macro pAP | macro pAUROC | ΔpAP vs native | ΔpAUROC vs native | pAP up/flat/down | AUROC up/flat/down |
|---|---:|---:|---:|---:|---:|---:|
| native / P31 identity | 0.452521603402 | 0.934565049560 | 0.000000000000 | 0.000000000000 | 0/12/0 | 0/12/0 |
| P27R1 | 0.461387566334 | 0.920341193874 | +0.008865962933 | −0.014223855686 | 8/0/4 | 1/0/11 |
| P29 | 0.455924444190 | 0.919197902935 | +0.003402840788 | −0.015367146625 | 5/0/7 | 1/0/11 |

P31 generated no new 12-category predictions; its identity projection is
exactly the existing native row. The full per-category values and deltas are
in `P31_FAST_OBJECTIVE_SCREEN.json`.

## 4. Direction and ranking gate

Direction/ranking is `NOT_APPLICABLE` for P31 because no student residual or
teacher target exists. Treating this as a numerical zero or inventing a
student agreement would be invalid.

The existing frozen transfer contrast is decisive for the prior target:

| method | directional cosine | sign agreement | Spearman | pAP | pAUROC |
|---|---:|---:|---:|---:|---:|
| P30 | 0.736923574 | 0.569567901 | 0.714233750 | 0.144618064 | 0.972904419 |
| P30R1 | −0.070148225 | 0.119691358 | 0.054565094 | 0.511513734 | 0.980534709 |
| P31 identity | N/A | N/A | N/A | 0.514140305 | 0.980667144 |

P30 has stronger direction metrics but much worse detection; P30R1 has weak
direction/ranking but recovers pAP and pAUROC. The frozen evidence therefore
does not support adding a direction, sign, or ranking objective. This screen
does not establish a new teacher-transfer mechanism.

## 5. Normality and correction gate

P31 has no correction:

- residual mean absolute value: `0`
- residual absolute q99: `0`
- score-delta mean absolute value: `0`
- score-delta q99: `0`
- normal score shift: `0`
- effective correction support: `0`
- inference overhead: `0%`

For context, the existing frozen diagnostics report:

| method | residual abs q99 | normal score-delta q99 | normal score-delta mean | effective support |
|---|---:|---:|---:|---:|
| P30 | 25.929798946 | 0.998690784 | 0.020160845 | 0.109472575 |
| P30R1 | 4.528306532 | 0.000007328 | 0.000675487 | 0.056409298 |
| P31 identity | 0 | 0 | 0 | 0 |

P31 passes the normality gate by identity. P30R1 reduced the normal-tail
problem relative to P30, but its sparse correction is not evidence that the
correction is useful: native remains slightly better on the frozen candle
comparison.

## 6. Cross-category and transfer interpretation

P31 makes no category-specific correction and therefore makes no claim of
cross-category correction transfer. That is scientifically preferable to
calling an identity output a learned transfer result. The prior learned
transfers provide the relevant warning: P27R1 improved pAP in 8/12 categories
but regressed AUROC in 11/12; P29 improved pAP in 5/12 but regressed AUROC in
11/12. Gains were not broad safety improvements.

No feature-level consistency was tested or added. No objective-level rescue
loss was tested or added. The zero-objective control is sufficient for the
question “is intervention necessary?” but not evidence that a future learned
effect/rank objective would work.

## 7. Runtime, memory, and cost

P31 is evaluation-only:

- objective count: `0`
- new hyperparameters: `0`
- optimizer steps: `0`
- training overhead: `0%`
- inference overhead: `0%`
- new model memory: `0` bytes
- warmed offline control median: `1.0229647e-05` seconds
- warmed offline control p90: `1.0454049e-05` seconds
- warmed RSS maximum: `155475968` bytes

The P31 profile is an offline resident-array control profile, not a claim
that a native CLIP forward costs microseconds. Existing end-to-end references
are retained separately: P27R1 scientific wall time `25124.04` seconds and
P30R1 training wall time `1215.84` seconds. P31 adds no model or inference
branch.

## 8. Gate result and decision

| gate | result | reason |
|---|---|---|
| Engineering | PASS | P31 qualification and parity artifacts pass |
| Direction | N/A for null | no student/teacher residual path exists |
| Normality | PASS | exact native identity, zero shift |
| Scientific proxy | PASS for null control | native/P31 is non-inferior to P30R1 on both frozen candle metrics |
| Scale | NOT AUTHORIZED | P31 is a control, not a learned mechanism to scale |

**Decision: `KEEP CURRENT`.** The strongest defensible result is to keep the
native detector as the default for this scope and stop teacher-imitation
expansion. A future learned downstream-effect or ranking hypothesis would need
its own research decision, exact formulation, preregistration, and
engineering qualification. It must not be inferred from this control.

The pre-existing P30R1 candle validation already answers the locked native
counterfactual. Launching a new P31 Stage-2 attempt would duplicate that
comparison and is not scientifically necessary.

## 9. Exact artifact evidence

- Fast-screen machine-readable record:
  `research/sabra_v2/region_distill/P31_FAST_OBJECTIVE_SCREEN.json`
- P31 qualification:
  `research/sabra_v2/region_distill/P31_ENGINEERING_QUALIFICATION.json`
- P31 frozen preregistration:
  `research/sabra_v2/region_distill/P31_PREREGISTRATION.md`
- P31 speed profile:
  `research/sabra_v2/region_distill/P31_SPEED_PROFILE.json`
- P30R1 causal forensic:
  `research/sabra_v2/region_distill/P30R1_FORENSIC/P30R1_CAUSAL_FORENSIC.json`
- P30R1 transfer and stability diagnostics:
  `research/sabra_v2/region_distill/P30R1/P30R1_TRANSFER_DIAGNOSTIC.json`,
  `research/sabra_v2/region_distill/P30R1/P30R1_STABILITY_DIAGNOSTIC.json`
- P27R1 category reference:
  `research/sabra_v2/region_distill/P27R1_SCIENTIFIC_METRICS.csv`
- P29 category reference:
  `research/sabra_v2/region_distill/P29_CLASS_TABLE.csv`

## 10. Required counts

```text
new scientific Stage 2 attempts = 0
new Stage 3 attempts = 0
full runs = 0
held-result tuning iterations = 0
new CLIP forwards = 0
new Phase2B forwards = 0
cache rebuilds = 0
```

No scientific UUID, execution marker, new prediction, or new held score was
created. No MVTec or Medical data was touched.
