# P25R — Patch-Level Benefit / Actionability Identifiability Recovery V1

P25R starts from `c0dd9ee86806346276f07bad8a7d1ea56327590d`, the pre-marker
P25 panel-capacity no-go. It is a clean new preregistration: no previous P25
target, fold, attempt, or scientific result is reused.

P25R tests whether exact native-anchored source patch advantage can be ranked
from GT-free patch/action features across strict 12-class LOCO, and, only on
Q1 pass, whether a source-calibrated low-capacity selector yields safe, broad,
meaningful held pAP improvement. It is not a final architecture or external
validation.

Frozen components are Phase2B, CLIP, signed direction, harm risk, alpha=.25,
deployment transform, exact pixel pAP/pAUROC, 12 VisA classes, and canonical
GT-free caches. MVTec/Medical reads, new CLIP forwards, Phase2B training,
prompt adaptation, alpha sweeps, and image-level SAFE controller revival are
forbidden. There is exactly one post-execution-base attempt marker.

Q1 failure is `P25_PATCH_BENEFIT_NOT_IDENTIFIABLE`; Q1 pass/Q2 failure is
`P25_PATCH_BENEFIT_NOT_POLICY_TRANSFERABLE`; all Q1/Q2 gates passing is
`P25_PATCH_ACTIONABILITY_IDENTIFIED`, which only permits future user review.
