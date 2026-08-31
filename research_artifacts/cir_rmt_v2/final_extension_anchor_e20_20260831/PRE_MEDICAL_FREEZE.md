# Pre-Medical freeze

Status: FROZEN. Target evaluation is authorized only after this record.

This is a target-blind freeze. No Medical metric, target label, target
checkpoint choice, or target-domain diagnostic was used to create it. MVTec
remains out of scope.

## Frozen identity

- Architecture: `CIR_DFG_RMT_V2`, architecture version 2.
- Config SHA256: `064e8acd4369645f631030b5d60abf8615e878b50e9caff6a4a8b2439b64f81c`.
- Architecture freeze SHA256: `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86`.
- Source: VisA, seed 0, FP32, effective batch 6.
- CLIP asset SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
- Image anchor: Phase2B E14, lambda `0.001`, SHA256 `3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2`.
- Anchor E20 checkpoint SHA256: `91f121852b6c20b8dddf31225518db10f79219b61e8b326e3afe37d06b4dbde6`.
- Frozen source sample: 96 images, sample seed 9014, 8 images/category, identity from the pre-full root-cause lock.
- Source matrix SHA256: `f4f730829e0aa087a17e5d9e2198ef5dcc12b9c576cdb93babb71528990ad43e`.
- Representation closure SHA256: `07e3cf00c87dd0c4696feaf1e3619610cf4e08711f0f2b17c3ef9bd1cf618c13` for parameter drift and `0f687e8c97eaf8bc670b49d1fd233e399c08da8949115627f7d4cc75b2b2895f` for feature drift.

## Frozen evaluation set

- Candidate epochs: E10, E12, E14, E16, E18, E20.
- Methods: `P`, `C_OLD_0`, `C_OLD_05`, `A0`, `A05`.
- Medical targets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir.
- Metrics: pixel AUROC, pixel AP, image AUROC, and image AP where the evaluator defines them.
- Target tuning: prohibited.
- MVTec evaluation: prohibited in this experiment.

## Primary checkpoint rule

The primary anchored deployment candidate is selected before Medical from the
source-only matrix by the lexicographic rule:

1. highest `A05` deterministic-source pixel AUROC;
2. highest `A05` deterministic-source pixel AP as a tie-break;
3. earliest epoch as a final tie-break.

Under that frozen rule, `PRIMARY_CHECKPOINT=A05_E20` because its source pixel
AUROC is `0.9711176169`, the highest A05 value across the six candidate
epochs. This is a weak source-training-sample criterion, not target tuning;
the complete six-epoch target matrix remains mandatory and no target result
may revise the primary epoch.

## Causal labels frozen for reporting

- `ANCHOR_TRAIN_EFFECT = A0 - P`.
- `RMT_INFERENCE_EFFECT = A05 - A0`, conditional on the anchored representation.
- `TOTAL_ANCHORED_CIR_EFFECT = A05 - P`.
- `A0`/`A05` are not to be collapsed into the old CIR labels.
- Same-epoch representation distances are descriptive and cannot establish
  transfer causality.

The scientific question after this freeze is whether the anchored training
trajectory improves target behavior relative to P and C_OLD, and whether the
conditional A05-minus-A0 inference effect is directionally useful across
domains. No decision has been made from Medical yet.
