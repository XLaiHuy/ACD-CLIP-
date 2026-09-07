# H2 S2-LOCR R1 Oracle Decision

* `LOCAL_ORACLE_CAUSAL_SUPPORT=PASS`
* `GLOBAL_STAGE2_REPLACEMENT_REPORTED=YES`
* `STRONGEST_DESIRABLE_SIGNATURE=PRESENT`

The local oracle is a GT-assisted source-only diagnostic, not a deployable method. It changes only production-resized Stage-2 logits on exact near-background pixels, then recomputes unchanged equal pre-softmax fusion.

Gate checks:
* `local_ap_ge_baseline=PASS`
* `local_auroc_ge_baseline=PASS`
* `local_near_p95_le_baseline=PASS`
* `local_near_p99_le_baseline=PASS`
* `local_positive_mean_ge_baseline_minus_1e6=PASS`
* `local_positive_median_ge_baseline_minus_1e6=PASS`
* `local_interior_mean_ge_baseline_minus_1e6=PASS`
* `local_interior_median_ge_baseline_minus_1e6=PASS`

If the local oracle gate fails, S2-LOCR formulation, gradient preflight, and training are not authorized by this protocol.
