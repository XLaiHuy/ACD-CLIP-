# H2 Root-Cause / S2-LOCR R1 Pareto Audit

* Hypothesis: `NOT_SUPPORTED`
* Curve: `EARLY_USEFUL_THEN_OVERSUPPRESSED`
* Best source Pareto alpha: `NONE`

The five alpha values are fixed endpoint weight interpolations, not lambda interpolation and not a training trajectory.

| alpha | A AP | B AP | A AUROC | B AUROC | A final near p95 | B final near p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.368366529 | 0.615166781 | 0.810918086 | 0.989935066 | 0.528762907 | 0.972146574 |
| 0.25 | 0.392368517 | 0.631777668 | 0.816551015 | 0.990939523 | 0.443752626 | 0.927755541 |
| 0.5 | 0.405315935 | 0.637359569 | 0.827686166 | 0.992760082 | 0.330584455 | 0.780031574 |
| 0.75 | 0.401407295 | 0.633655214 | 0.841941768 | 0.993724566 | 0.221653324 | 0.607655641 |
| 1 | 0.385816946 | 0.626174194 | 0.858161976 | 0.994415133 | 0.143470044 | 0.478725855 |

No alpha is called a source Pareto point unless every preregistered preservation check passes on both cohorts. Raw positive/interior means are retained as diagnostics but are not used as the sole gate.

`R2_TRAINING_AUTHORIZED=NO`; this report only authorizes a source-only formulation direction after final diagnosis.
