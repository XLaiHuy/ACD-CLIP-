# EXTERNAL_VALIDATION_FAILURE

`VALID=false` for the MVTec external-validation stage.

The candidate freeze was already remotely verified, but the required MVTec
image root was absent at every audited location:

- `/workspace/data/mvtec_ad`
- `/workspace/data/data/mvtec_ad`
- `/workspace/data/MVTec`
- `/workspace/data/data/MVTec`

No MVTec image or mask was opened. The metadata probe count is 1 and the
image/mask read counts are 0. No external metrics or confirmatory evidence
were generated. This is an external-data gate failure, not a failure of the
finalized cache or the VisA analysis.

The VisA result, frozen candidate, GT-free cache, OOF-derived quantities,
Trust-v2/Need/Authority VisA statuses, and medical firewall remain usable and
unchanged. Full-20e training remains unauthorized. If an authorized MVTec
image root is supplied later, rerun only the frozen MVTec evaluation.

The complete machine-readable record is in `FAILURE_STATE.json`. Existing
MVTec-unavailable artifacts were preserved in place and are not relabeled as
valid external results.
