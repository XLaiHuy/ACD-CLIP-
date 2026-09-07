# H2 Stagewise Causal Localization Audit R1 — Diagnosis-Conditional Literature R&D

## Research status

`RESEARCH_ACCESS=AVAILABLE` on 2026-09-07. The diagnosis was frozen before
searching. The evidence points to `CONTEXTUAL_MIXING_WITH_PATCH_AMBIGUITY`:
Stage 2 has unique oracle support, zero-footprint near scores are elevated,
partial footprints carry much larger scores, and the strongest directional
families suppress Stage 2 while Conv-LoRA also changes Stage 3. The candidate
replay is invalid, so no trajectory result is used to rank literature.

The sources below are scholarly primary papers or official open-access paper
pages. They motivate candidate mechanisms only; none is implemented here.

## Retained papers

### PointRend: Image Segmentation As Rendering

* Venue/year: CVPR 2020, Kirillov, Wu, He, Girshick.
* Verified sources: [CVPR open-access paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf), [DOI 10.1109/CVPR42600.2020.00982](https://doi.org/10.1109/CVPR42600.2020.00982), [arXiv:1912.08193](https://arxiv.org/abs/1912.08193).
* Mechanism/evidence: treat segmentation as rendering; iteratively sample uncertain points and predict them with fine and coarse features. The paper reports crisper boundaries and gains on COCO and Cityscapes while avoiding dense high-resolution computation.
* Similarity/difference to ACD-CLIP: similarity is the need to recover fine spatial detail from a coarse regular grid; difference is that ACD-CLIP currently Gaussian-smooths and bilinearly expands 37x37 stage logits, whereas PointRend adds an adaptive point-wise prediction path.
* Novelty conflict: `MEDIUM`; an anomaly-specific, stagewise logit-margin renderer with footprint diagnostics would need a distinct contribution beyond PointRend.
* Adaptability: `HIGH`; it can sit after the frozen equal-fusion map or be attached to native stage features.
* Complexity: `MEDIUM`.
* Overhead: training `MEDIUM`; inference `LOW-to-MEDIUM` if only uncertain/boundary points are rendered.

### Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former)

* Venue/year: CVPR 2022, Cheng, Misra, Schwing, Kirillov, Girdhar.
* Verified sources: [CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html), [DOI 10.1109/CVPR52688.2022.00135](https://doi.org/10.1109/CVPR52688.2022.00135), [arXiv:2112.01527](https://arxiv.org/abs/2112.01527).
* Mechanism/evidence: masked cross-attention constrains feature extraction to predicted mask regions; the paper reports strong semantic, instance, and panoptic segmentation results and emphasizes localized feature extraction.
* Similarity/difference to ACD-CLIP: similarity is explicit spatial restriction of attention; difference is that the current DFG Q/K path mixes stage/text features without a predicted-mask support constraint and is not a mask-query decoder.
* Novelty conflict: `HIGH`; replacing or constraining DFG attention would overlap a well-established masked-attention mechanism and would be a larger architecture change.
* Adaptability: `MEDIUM`; it would require a decoder/query interface or a carefully scoped mask-conditioned attention adapter.
* Complexity: `HIGH`.
* Overhead: training `HIGH`; inference `MEDIUM-to-HIGH` due to mask-query attention and decoder state.

### Glancing at the Patch: Anomaly Localization With Global and Local Feature Comparison

* Venue/year: CVPR 2021, Wang, Wu, Cui, Shen.
* Verified source: [CVPR open-access paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.html).
* Mechanism/evidence: separates local patch detection from surrounding global context, trains a Global-Net to mimic local features, and uses inconsistency/distortion heads to detect local-context discrepancy. The paper reports improved anomaly localization on industrial benchmarks.
* Similarity/difference to ACD-CLIP: similarity is explicit local-versus-context comparison; difference is that ACD-CLIP’s S2-LOCR acts directly on near-background logits and its directional audit shows the image-side/Conv-LoRA pathway can suppress both inside and outside margins, rather than providing a separately parameterized local/context residual.
* Novelty conflict: `MEDIUM`; a stagewise residual comparison could be distinct, but generic local/global discrepancy is established.
* Adaptability: `HIGH`; a residual head can consume native stage tokens without changing the frozen endpoint fusion rule during diagnosis.
* Complexity: `MEDIUM`.
* Overhead: training `MEDIUM`; inference `MEDIUM` for extra local/context features and heads.

## Candidate directions (not implemented)

### CANDIDATE_1 — Uncertainty-guided native-footprint refinement with stagewise margin gating

Mechanism: retain the existing 37x37 stage logits and equal fusion, identify
uncertain/boundary points from the fused map, and render only those points with
fine image features plus a gated native Stage-1/2/3 margin. The gate must be
stagewise and calibrated against exact patch occupancy, so zero-footprint
near-background pixels cannot inherit a partial-footprint score solely through
bilinear expansion.

Diagnosed failure addressed: patch-footprint aliasing plus contextual
near-background spillover. Why it is better targeted than Functional Anchor,
GradBudget, Late Freeze, THBR, and S2-LOCR R1: those interventions constrain
feature drift, update budgets, freeze timing, boundary loss, or Stage-2 local
logits; none supplies a subpatch observation path that resolves the verified
14x14 footprint ambiguity. Expected benefit is improved boundary/interior
separation with lower dense inference cost than full-resolution refinement.

Risks: it may become a post-hoc score repair, may overfit the VisA footprint,
and can still inherit cross-stage responsibility coupling unless the gate is
audited independently. Novelty is `MEDIUM`; generalization is `MEDIUM-HIGH` if
the renderer is trained and tested across image sizes. Complexity `MEDIUM`;
training overhead `MEDIUM`; inference overhead `LOW-to-MEDIUM`.

Preflight required: exact A/B occupancy and near/far red-team replay, native
versus resized ranking, stagewise gradient directional audit, and a frozen
cross-domain holdout before any endpoint decision. Implementation is not
authorized by this report.

### CANDIDATE_2 — Mask-conditioned stage responsibility routing

Mechanism: add a lightweight mask-conditioned attention support so each stage’s
text/image interaction is restricted to its predicted support, inspired by
Mask2Former masked attention. The aim is to prevent context tokens from
receiving the same responsibility as anomaly-support tokens.

Diagnosed failure addressed: contextual mixing and Conv-LoRA cross-stage
responsibility compensation. It is potentially more direct than Functional
Anchor, GradBudget, Late Freeze, THBR, and S2-LOCR R1 because it changes the
spatial support of interaction rather than only the update direction.

Risks: high architecture/novelty conflict with existing DFG attention, query
collapse, dependence on imperfect predicted masks, and substantial compute.
Novelty `LOW-to-MEDIUM`; generalization `MEDIUM`; complexity `HIGH`; training
overhead `HIGH`; inference overhead `MEDIUM-to-HIGH`.

Preflight required: attention-support ablation, stagewise responsibility
conservation, zero-footprint and partial-footprint counterfactuals, and strict
no-target cross-domain evaluation. Implementation is not authorized.

### CANDIDATE_3 — Explicit local-context residual branch with stage-isolated heads

Mechanism: add separate local-patch and context encoders/heads and score their
disagreement as an anomaly residual, drawing on Glancing at the Patch. Stage
heads would expose a local residual without forcing the same trainable family
to suppress Stage 2 and alter Stage 3.

Diagnosed failure addressed: coupled suppression and cross-stage compensation.
It is better targeted than Functional Anchor, GradBudget, Late Freeze, THBR,
and S2-LOCR R1 when the failure is responsibility sharing, but it is less
direct than CANDIDATE_1 for the verified 14x14 footprint alias.

Risks: extra local/global feature capacity can amplify contextual leakage,
duplicate the existing multistage representation, and introduce a second
score-calibration problem. Novelty `MEDIUM`; generalization `MEDIUM`; complexity
`MEDIUM`; training overhead `MEDIUM`; inference overhead `MEDIUM`.

Preflight required: local/context swap tests, stage-isolated family
directional derivatives, occupancy-conditioned ranking, and frozen A/B plus
cross-domain validation. Implementation is not authorized.

## Ranking and recommendation

| Rank | Direction | Fit | Causal evidence | Novelty | Simplicity | Adaptability | Conflict | Compute | Survival probability |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CANDIDATE_1 | High | High | Medium | Medium | High | Medium | Medium | High |
| 2 | CANDIDATE_3 | Medium-High | Medium | Medium | Medium | High | Medium | Medium | Medium |
| 3 | CANDIDATE_2 | Medium | Low-Medium | Low-Medium | Low | Medium | High | High | Low-Medium |

`RECOMMENDED_NEXT_DIRECTION=UNCERTAINTY_GUIDED_NATIVE_FOOTPRINT_REFINEMENT_HEAD_WITH_STAGEWISE_MARGIN_GATING`.

CANDIDATE_2 is not selected because its mechanism overlaps the existing DFG
attention path and has the highest architecture/compute risk. CANDIDATE_3 is
not selected because it may reinforce the same local/context mixing that the
audit found and does not directly resolve zero-footprint aliasing. The selected
direction is the smallest hypothesis that directly tests the strongest frozen
diagnosis, but it remains advisory only.

`IMPLEMENTATION_AUTHORIZED=NO`.
