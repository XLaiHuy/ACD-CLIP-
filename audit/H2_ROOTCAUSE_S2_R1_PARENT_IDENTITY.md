# H2 Root-Cause / S2-LOCR R1 Parent Identity

* branch: `research/h2-rootcause-s2-strength-audit-r1`
* exact parent HEAD: `3c97eed761df4669d4143aa7b5d7b586f97d001c`
* parent identity: `PASS`
* Safe Anchor: `/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth` (64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7)
* retained Cohort A: `audit/H2_STAGEWISE_CAUSAL_R1_COHORT_A.json` (a07ee8ac03ee49ca0fa11cf0e018b7861214212b361c5235219730747465c05e)
* retained Cohort B: `audit/H2_STAGEWISE_CAUSAL_R1_COHORT_B.json` (68cf348efbb31b1b4e7460e227f93cc47653f9bca9f5795b0af6314b8b4ed1c6)
* retained endpoints: control `2d4e8d43b2b47edb05ebcc6e3faf25bfd8f7ffd93517d42eb05e8441683e5ba5`, candidate `bc79ac7ed75de405a592458563e3a98d742aaa4ab12b4e7a2d5d9adeb21938a3`
* target inference: `NO`
* training / lambda sweep: `NO`
* committed S2-LOCR R1 decision: preserved unchanged (`BOUNDED_SCREEN=FAIL`, `CASE_B`, `NOT_SUPPORTED`)

Cohorts are reused byte-for-byte from the previous source-only stagewise audit. No new category, image, target, score, or metric-based selection is performed.
