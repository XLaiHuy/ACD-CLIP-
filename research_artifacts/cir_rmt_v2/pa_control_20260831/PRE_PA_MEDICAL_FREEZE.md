# Pre-PA Medical freeze

Status: PASS. Source-only freeze completed before Medical access.

PA source evaluation covered E10/E12/E14/E16/E18/E20 on the fixed 96-image VisA sample. The source-only reporting epoch is E20 under the rule: highest PA pixel AUROC, tie by PA pixel AP, then earliest epoch. This does not cherry-pick the Medical benchmark: all six PA checkpoints must be evaluated.

Primary Medical factorial: P vs C_OLD_0 vs PA vs A0, with CIR-with-anchor = A0 - PA and interaction = A0 - C_OLD_0 - PA + P. Existing P/C_OLD/A Medical rows are frozen and will be reused; only the 36 PA cells are authorized for new evaluation.

Identity: VisA source, seed 0, ViT-L/14@336, image size 518, FP32, AMP=false, TF32=false, effective batch 6, Adam/StepLR canonical schedule, exact P_E14 image anchor lambda=0.001. Target tuning: NO. MVTec: NOT RUN.

The full machine-readable freeze is PRE_PA_MEDICAL_FREEZE.json. No Medical or MVTec data were accessed before this freeze.
