# P30R1 Causal Forensic Report

## 1. Executive finding

The frozen candle result is best explained primarily by `TEACHER_DIRECTION_NOT_CAUSAL`, with `SPARSE_SELECTIVE_CORRECTION` as a secondary mechanism candidate. P30 has higher teacher-direction fidelity but far worse pAP, while P30R1 has collapsed direction and recovers the learned-adapter detection metrics; this makes teacher direction a poor validated causal proxy on this class. P30R1 also makes sparse, anomaly-enriched changes while staying close to the native detector on most pixels. This is post-hoc exploratory evidence only: P30R1 remains `STAGE2_SCIENTIFIC_STOP`, and no Stage 3 or full run was authorized.

## 2. Frozen-artifact inventory

The inventory below distinguishes stored tensors from deterministic reconstructions. P29 stored no region residual, so P29 residual-space analysis is `UNAVAILABLE_WITHOUT_NEW_FORWARD`; no adapter checkpoint was run. P30 and P30R1 stored region residuals. Shared Tier-A native logits, source Tier-B teacher tensors, all three held prediction maps, and post-freeze masks were available.

| Artifact | P29 | P30 | P30R1 | Native | Frozen? | Allowed? |
|---|---|---|---|---|---|---|
| frozen native logits | Tier-A shared cache | Tier-A shared cache | Tier-A shared cache | yes | True | True |
| frozen corrected logits | absent | absent | absent | no | False | False |
| anomaly probability maps | prediction tensor | prediction tensor | prediction tensor | yes | True | True |
| student residual tensors | absent | present [3,9,9] | present [3,9,9] | no | True | True |
| held teacher residual tensors | reconstructable from frozen native logits + masks | reconstructable from frozen native logits + masks | reconstructable from frozen native logits + masks | no | True | True |
| region-grid residuals | absent | present | present | no | True | True |
| upsampled residuals | absent | deterministically reconstructable | deterministically reconstructable | no | True | True |
| final prediction maps | present | present | present | yes | True | True |
| held masks | present post-freeze | present post-freeze | present post-freeze | yes | True | True |
| frozen prediction hashes | completion JSON | completion JSON | completion JSON | yes | True | True |
| source tensors | Tier-A/Tier-B cache | Tier-A/Tier-B cache | Tier-A/Tier-B cache | yes | True | True |
| held tensors | Tier-A features/native logits | Tier-A features/native logits | Tier-A features/native logits | yes | True | True |
| per-sample identifiers | image_path | image_path | image_path | yes | True | True |
| per-pixel identifiers | image_path + array coordinates | image_path + array coordinates | image_path + array coordinates | derived | True | True |
| correction-scale metadata | frozen code/protocol | frozen code/protocol | frozen code/protocol | yes | True | True |

## 3. Native / zero-adapter comparison

The zero-residual prediction reconstructed from frozen Tier-A native logits through the unchanged deterministic deployment operator is `EXACT_FROZEN_COUNTERFACTUAL`. Its metrics are pAP `0.514140301610` and pAUROC `0.980667144217`, matching the frozen native reference within the recorded reconstruction tolerance. Native pAP is slightly above P30R1 (`0.002626567385`), so the evidence supports preservation / damage avoidance more strongly than a claim that the learned correction improves the frozen detector. This is exploratory evidence and does not alter the P30R1 gate.

Native score statistics: global mean `0.00044657`, normal-pixel q99 `0.00000038`, anomaly-pixel q99 `0.99158327`.

## 4. Prediction similarity

All similarity descriptors are unlabeled except where explicitly stated; fixed pixel fractions and fixed absolute score-difference thresholds were chosen before inspecting the result.

| Method vs native | Pearson | mean | q99 abs diff | top-1% overlap |
|---|---:|---:|---:|---:|
| P29 | 0.785339941 | 0.000837845 | 0.000002201 | 0.887697 |
| P30 | 0.187372506 | 0.021002682 | 0.998811186 | 0.803834 |
| P30R1 | 0.731909455 | 0.001114394 | 0.000013708 | 0.956680 |

P30R1 score-delta fixed-threshold fractions are `{"0.0001": 0.007153907216648529, "0.001": 0.0049976707264352055, "0.01": 0.003436330704670473, "0.05": 0.002562759946929831, "1e-06": 0.017428519252843576}`. The global Pearson value is not near-perfect because a small set of anomaly-region changes carries most absolute delta mass; per-image Pearson is `0.947845739` and top-1% overlap is `0.956680`. Its gamma sensitivity (`0.0, 0.5, 1.0`) is recorded in the JSON using only unlabeled score changes.

## 5. Correction magnitude and sparsity

| Method | residual mean | residual q99 | score-delta mean abs | score-delta q99 abs |
|---|---:|---:|---:|---:|
| P29 | UNAVAILABLE | UNAVAILABLE | 0.000837845 | 0.000002201 |
| P30 | 1.542490326 | 25.929798946 | 0.021002682 | 0.998811186 |
| P30R1 | 0.178743865 | 4.528306532 | 0.001114394 | 0.000013708 |

P30R1 residual effective support fraction is `0.056409`; score-delta top-mass and Gini descriptors are in the machine-readable artifact. P29 residual sparsity is `UNAVAILABLE_WITHOUT_NEW_FORWARD`.

## 6. Directional collapse diagnosis

P30R1 staged directional cosine mean is `-0.070148225` and sign agreement mean is `0.119691358`. Teacher residuals were reconstructed from frozen native logits and post-freeze masks with the exact deterministic R0 utility; no teacher neural forward occurred. The result therefore cannot be dismissed as a missing teacher tensor, but direction may still be ill-conditioned when the student norm is small.

P30R1 student norm median is `2.645800829` and student/teacher norm-ratio median is `0.061280576`. Norm/cosine Pearson/Spearman correlations are `0.738152` / `0.795999`, while norm/sign correlations are `0.147057` / `0.339461`.

## 7. Student-norm-conditioned direction analysis

The four bins are descriptive held student-norm quartiles, not label-selected thresholds.

| Method / norm bin | n | student norm median | cosine mean | sign mean | score-delta abs mean |
|---|---:|---:|---:|---:|---:|
| P30 q00_q25 | 50 | 12.906026927 | 0.850333276 | 0.877283951 | 0.000000005 |
| P30 q25_q50 | 50 | 13.182304784 | 0.948903164 | 0.959506173 | 0.000000002 |
| P30 q50_q75 | 50 | 38.299201345 | 0.505879899 | 0.314567901 | 0.007033083 |
| P30 q75_q100 | 50 | 126.880558764 | 0.642577957 | 0.126913580 | 0.076977637 |
| P30R1 q00_q25 | 50 | 0.117977138 | -0.774313622 | 0.092592593 | 0.000000000 |
| P30R1 q25_q50 | 50 | 0.129700595 | -0.533927528 | 0.130617284 | 0.000000000 |
| P30R1 q50_q75 | 50 | 12.210726602 | 0.405624213 | 0.123209877 | 0.000535841 |
| P30R1 q75_q100 | 50 | 20.182981551 | 0.622024038 | 0.132345679 | 0.003921733 |

The inherited raw coordinate epsilon threshold is `0.049601097` and the corresponding 243-coordinate L2 threshold is `0.773204583`. Exact and thresholded near-zero fractions are reported without an outcome-tuned cutoff.
The lowest-norm cosine is `-0.774313622` and the highest-norm cosine is `0.622024038`, but the highest-norm sign agreement remains `0.119691358` overall; low-norm abstention explains part of the collapse, not all of it.

## 8. Teacher-scale normalization reweighting

Across the unique source-cache union (`2162` samples), normalized teacher RMS q01/q50/q99 is `0.039694438` / `0.414681322` / `1.000000385`. The corresponding `1/a_t` q01/q50/q99 is `0.999949619` / `2.410789514` / `24.560375061`.

This means small-teacher samples receive larger bounded gradient coefficients and large-teacher samples receive smaller coefficients; the observed q99/q01 inverse-weight ratio is `24.561612`. The relationship is analytic and monotone (teacher-RMS/inverse-weight Spearman `-1.000000`), so P30R1 behaves as scale-balanced regression rather than direct unweighted residual imitation. This could explain residual shrinkage, but does not by itself prove downstream causality.

## 9. Sparse/selective correction analysis

For P30R1, mean absolute score delta is `0.000675487` on normal pixels and `0.325263709` on anomaly pixels. Absolute delta mass in anomaly pixels is `0.394671371`, an enrichment of `291.875700371` over anomaly area. This supports spatial selectivity, but aggregate pAP remains slightly below native, so “useful” correction is a hypothesis rather than an established causal result; the statistic is descriptive, not a tuning rule.

## 10. P29 → P30 → P30R1 mechanism table

| Property | P29 | P30 | P30R1 |
|---|---|---|---|
| Objective count | 3 | 1 | 1 |
| Student self-normalized? | no | yes | no |
| Teacher-relative reweighting? | no | directional only | yes, via `1/a_t` |
| Exact zero teacher treatment | restoring term active | excluded from directional mean | retained by normalized SmoothL1 |
| Mean residual | `2.036682759` | `1.542490326` | `0.178743865` |
| Residual absolute q99 | `4.321676936` | `25.929798946` | `4.528306532` |
| Directional cosine | `0.708549174` | `0.736923574` | `-0.070148225` |
| pAP | `0.490503231` | `0.144618064` | `0.511513734` |
| pAUROC | `0.970040297` | `0.972904419` | `0.980534709` |
| Native top-1% overlap | `0.887697` | `0.803834` | `0.956680` |
| Correction sparsity | residual unavailable | `0.109473` residual support | `0.056409` residual support |
| Main mechanism | mixed-objective conflict | radial non-identifiability | see ranked forensic hypotheses |

## 11. Causal hypothesis ranking

1. `TEACHER_DIRECTION_NOT_CAUSAL` — P30 has higher held directional cosine but far worse pAP, whereas P30R1 has collapsed direction and recovers pAP; this is a direct frozen cross-method proxy contrast, not a causal proof.
2. `SPARSE_SELECTIVE_CORRECTION` — P30R1 score-change mass is strongly anomaly-enriched and its residual effective support is small; aggregate usefulness remains exploratory because native pAP is slightly higher.
3. `DO_NO_HARM_NATIVE_PRESERVATION` — The exact native counterfactual is strong; P30R1 has high per-image similarity and top-1% overlap, but its global score correlation and anomaly-region deltas show that preservation is incomplete.
4. `DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION` — The lowest-norm quartile has cosine -0.774 and the highest has 0.622, while 46% of P30R1 vectors are below the inherited vector epsilon scale; high-norm sign agreement remains poor, so this is only partial.
5. `TEACHER_SCALE_REWEIGHTING` — The frozen objective implies a 24.56x q99/q01 inverse-weight spread over the source cache; the mechanism is analytically clear, but individual training-step causality was not reconstructed.

## 12. Falsification evidence

Each hypothesis has an explicit falsifier. The observed evidence is descriptive and does not convert any hypothesis into a training gate.

- `H1_DO_NO_HARM_NATIVE_PRESERVATION` — falsifier: Strongly non-native score/ranking changes despite small mean residual would falsify native preservation.
- `H2_SPARSE_SELECTIVE_CORRECTION` — falsifier: No anomaly enrichment and diffuse correction mass would falsify useful selectivity.
- `H3_TEACHER_DIRECTION_PROXY_FAILURE` — falsifier: Consistently better downstream score behavior among high-direction samples after norm conditioning would weaken this hypothesis.
- `H4_DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION` — falsifier: Persistently poor direction among substantial-correction bins would falsify low-norm ill-conditioning as the main explanation.
- `H5_TEACHER_SCALE_REWEIGHTING` — falsifier: Nearly constant weights unrelated to scale would falsify a meaningful reweighting mechanism.

## 13. SABRA-old overconstraint risk

Yes, the old failure pattern is a live risk. Adding cosine, sign, ranking, or gating terms solely to repair P30R1's failed internal fidelity metric would repeat P29's mixed-objective temptation without evidence that those terms improve detection. The current candle result instead says teacher fidelity must earn its place as a causal target; no rescue objective is implemented or recommended here.

## 14. Scientific interpretation

P29's multiple objectives created conflict/starvation. P30 isolated direction and exposed radial non-identifiability. P30R1 restored radial control and learned-adapter detection while direction collapsed. On this one class, downstream utility and teacher imitation quality are decoupled: the cross-method contrast supports `TEACHER_DIRECTION_NOT_CAUSAL` as the primary forensic mechanism, with sparse anomaly-enriched intervention as a secondary candidate. Low-norm abstention and teacher-scale reweighting plausibly contribute, but neither is isolated as the sole cause. The exact native counterfactual remains slightly better than P30R1, so no superiority over the frozen detector is claimed and no result is generalized across classes.

## 15. Recommended next research question

What downstream-relevant invariant should be transferred instead of raw teacher correction direction?

## Required terminal state

`FORENSIC_COMPLETE` — primary mechanism: `TEACHER_DIRECTION_NOT_CAUSAL`; secondary mechanism: `SPARSE_SELECTIVE_CORRECTION`. New training runs: `0`; optimizer steps: `0`; new CLIP/Phase2B forwards: `0`; cache rebuilds: `0`; new scientific marker: `false`.
