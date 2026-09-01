# Source factorial interpretation

Status: PASS. The fixed 96-image VisA sample is used only to characterize the 2x2 training factorial before Medical access.

P = native Phase2B without anchor; C_OLD_0 = CIR training without anchor; PA = native Phase2B with the frozen P_E14 image anchor; A0 = CIR training with that same anchor.

Primary contrast: CIR_WITH_ANCHOR = A0 - PA. Interaction = A0 - C_OLD_0 - PA + P. These source effects are diagnostic associations, not target-domain evidence and not a permission to tune on Medical.

See SOURCE_FACTORIAL_2X2.csv for every epoch and metric, and PRE_PA_MEDICAL_FREEZE.json for the target-blind freeze created after this source stage.
