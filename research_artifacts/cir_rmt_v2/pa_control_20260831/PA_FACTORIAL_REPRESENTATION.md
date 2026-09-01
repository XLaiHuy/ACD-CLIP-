# PA factorial representation and source analysis

Status: PASS for the source-only stage.

The four primary source cells are P (native/no anchor), C_OLD_0 (CIR/native/no anchor), PA (native/image anchor), and A0 (CIR/native/image anchor). The primary CIR-with-anchor contrast is A0 - PA. The 2x2 interaction is A0 - C_OLD_0 - PA + P.

PA was forwarded with the canonical native Phase2B path only. It has no CIR/RMT transport, peer search, delta, or alpha inference. The PA feature rows therefore measure the native representation change associated with the train-only image anchor relative to the matched P checkpoint; they do not establish a target-domain causal result.

Frozen P/C_OLD/A representation context was reused from `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/research_artifacts/cir_rmt_v2/final_extension_anchor_e20_20260831/SAME_EPOCH_FEATURE_DRIFT.csv`. New P-vs-PA compact feature rows are in `PA_SOURCE_FEATURE_DRIFT.csv` and combined in `PA_FACTORIAL_DRIFT.csv`. Parameter rows are in `PA_PARAMETER_DRIFT.csv`. Frozen source matrix SHA256: `f4f730829e0aa087a17e5d9e2198ef5dcc12b9c576cdb93babb71528990ad43e`.

Source-only Medical freeze selection rule: highest PA pixel AUROC, tie-break by PA pixel AP, then earliest epoch. This selected E20 only as a preregistered reporting anchor; all six PA epochs remain required for Medical evaluation.

No Medical or MVTec data were accessed by this source stage.
