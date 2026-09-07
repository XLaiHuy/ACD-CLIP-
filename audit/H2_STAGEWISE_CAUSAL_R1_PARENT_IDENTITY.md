# H2 Stagewise Causal Localization Audit R1 — Parent Identity

* branch: `research/h2-stagewise-causal-localization-audit-r1`
* exact R1 parent HEAD: `f44cca2e163585dba3bbffc99c45518501db4852`
* branch HEAD at identity capture: `f44cca2e163585dba3bbffc99c45518501db4852`
* parent identity: `PASS`
* Safe Anchor E10: `/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth`
* Safe Anchor SHA256: `64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7`
* R1 control: `/workspace/h2_s2_locr_r1/A_S2_LOCR_R1_CONTROL/final.pth` (2d4e8d43b2b47edb05ebcc6e3faf25bfd8f7ffd93517d42eb05e8441683e5ba5)
* R1 candidate: `/workspace/h2_s2_locr_r1/A_S2_LOCR_R1_CANDIDATE/final.pth` (bc79ac7ed75de405a592458563e3a98d742aaa4ab12b4e7a2d5d9adeb21938a3)

The committed S2-LOCR R1 decision is preserved unchanged: `BOUNDED_SCREEN=FAIL`, `CASE_B`, and `S2_LOCR_MECHANISM=NOT_SUPPORTED`. This branch performs stagewise causal localization, patch-footprint, calibration-invariant, trajectory, and parameter-family diagnostics only.

No Medical inference, MVTec inference, target tuning, new training objective, architecture change, optimizer/fusion/interpolation change, hyperparameter sweep, S2-LOCR-v2, or post-hoc R1 gate modification is permitted or performed.

Cohort A is the exact committed 96-image spillover cohort (`a07ee8ac03ee49ca0fa11cf0e018b7861214212b361c5235219730747465c05e`). Cohort B is a new disjoint 96-image VisA test cohort selected only by the preregistered canonical manifest rule (`68cf348efbb31b1b4e7460e227f93cc47653f9bca9f5795b0af6314b8b4ed1c6`).
