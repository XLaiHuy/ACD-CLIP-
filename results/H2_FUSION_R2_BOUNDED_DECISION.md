# H2 Fixed-Fusion R2 Bounded Decision

`PROTOCOL_ID=H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2`

`BOUNDED_SCREEN=FAIL`

`MECHANISM_STATUS=NOT_SUPPORTED_BY_R2`

`NUMERICAL_VALIDITY=PASS`

`FULL_E15_STARTED=NO`

## Execution identity

- Branch: `research/h2-fixed-fusion-bounded-r2`
- Start checkpoint: `runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth`
- Start SHA256: `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`
- Decision head: `06e2107823fadc29ffdf0337c9acb7b8055b889a`
- Implementation parity: `PASS`
- 16-batch preflight: `PASS`
- Endpoint split: calibration `96`, endpoint `96`, intersection `0`
- Exact 500-attempt batch identity match: `PASS`

## Arm validity

- `A_FUSE_SHORT_R2_CONTROL`: attempts=500, successful=499, nonfinite_loss_skips=0, nonfinite_grad_skips=1, max_consecutive_grad_skips=1.
- `A_FUSE_SHORT_R2_CANDIDATE`: attempts=500, successful=499, nonfinite_loss_skips=0, nonfinite_grad_skips=1, max_consecutive_grad_skips=1.

## Mechanism and endpoint results

- R_late control mean: `0.657385408878`
- R_late candidate mean: `0.626247644424`
- Candidate minus control: `-0.031137764454` (lower: `PASS`)
- Final AP delta: `-0.005394889087`
- Final AUROC delta: `0.010204822524`
- Positive mean/median deltas: `0.009969443083` / `0.000001602694`
- Interior mean/median deltas: `0.012264832854` / `0.000001711181`
- Stage-1 AP/AUROC deltas: `-0.012786217261` / `-0.000171130026`
- Stage-2 AP delta: `-0.017633093426`; stage-3 AP delta: `0.012361037277`

## Preregistered gates

- `numerical_validity`: PASS
- `exact_batch_identity_match`: PASS
- `R_late_lower`: PASS
- `final_ap_noninferior`: FAIL
- `final_auroc_noninferior`: PASS
- `positive_mean_noninferior`: PASS
- `positive_median_noninferior`: PASS
- `interior_mean_noninferior`: PASS
- `interior_median_noninferior`: PASS
- `stage1_ap_noninferior`: FAIL
- `stage1_auroc_noninferior`: FAIL

The candidate lowered the direct late-stage contribution ratio, but the final
AP and stage-1 noninferiority gates failed. Geometry remains descriptive and
does not override those failures.

Calibration was not used for endpoint pass/fail. Medical, MVTec, target data,
and full E15 were not evaluated.

Next full E15: not started; a new explicit user approval is required.
