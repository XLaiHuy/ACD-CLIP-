# P26 Agent Context

## Current scientific state

SABRA/CURE studied whether source-trained GT-free evidence could safely improve
a frozen zero-shot anomaly detector by signed patch intervention. P26 freezes
`SABRA-FINAL-NATIVE-PHASE2B-V1`: the frozen Phase2B detector with native
postprocessing, no sidecar correction, and `KEEP` for every patch.

**DEPLOYABLE:** frozen Phase2B checkpoint, CLIP backbone, canonical preprocessing
and native postprocessing. **SUPPORTED BUT DISABLED:** signed direction and
harm-risk evidence. **DROPPED:** magnitude/uncertainty correction, R2/R2-v2
selectors, P14/P21 image-context controllers, and P25 patch-benefit RankNet.
**DIAGNOSTIC/ORACLE ONLY:** P13 rejected-sign-correct opportunity. Do not retry
closed branches automatically.

## Critical historical evidence

- **OBSERVED (R0):** fixed signed alpha 0.25 had source headroom.
- **OBSERVED (R1):** sign/rank transferred broadly, but exact magnitude failed
  the frozen MAE gate.
- **OBSERVED (R2/R2-v2):** risk/harm filtering reduced wrong-sign harm; it did
  not establish stable positive downstream pAP.
- **POST-HOC (P13):** over-abstention discarded useful sign-correct patches;
  the target-bearing cohort is not deployable evidence.
- **OBSERVED (P20):** contextual image value was weak and the valid recovered
  P14 study stopped scientifically.
- **OBSERVED (P23):** NATIVE fallback enlarged coarse headroom and all 12
  classes improved, but both A0/A1 failed the required macro-pAP magnitude;
  SAFE30 added negligible headroom.
- **OBSERVED (P25R3):** exact numerical recovery yielded 12 valid Q1 folds,
  median Spearman -0.08424, positive Spearman 2/12, macro sign AUC 0.55615,
  macro BC20 0.05882; G2-G6 failed and Q2 was not entered.
- **INTERPRETATION:** benefit/actionability is not GT-free identifiable using
  the frozen tested formulations. Safety evidence alone cannot activate an
  intervention. Native-only is the only complete deployable path.

Engineering-stop phases are provenance only and have no scientific outcome.

## Governance

Medical is forbidden. MVTec remains untouched by P26 until explicit user
authorization. No new CLIP forward or Phase2B training occurred. There is no
target adaptation, external threshold tuning, or architecture revision after
an MVTec result. `--run` is locked.

## Next step

Explicit user review. If later explicitly authorized, restore the exact tagged
P26 state and run one untouched MVTec external industrial validation. No P27
model search is automatically allowed.
