# Source-only selected-solution gate

## Scope

The gate used exactly the preregistered deterministic 96-image VisA sample:
8 images per category, 4 normal and 4 anomalous, with the recorded
`SOURCE_SAMPLE_IDENTITY.json` and held-out assessment categories. It used the
repository's unchanged `forward_cir`, `deploy_native_logits`, `image_score`,
and `binary_metrics` paths. Medical, MVTec, target tuning, and new full
evaluation were not run.

## Result

Implementation smoke: `PASS`.

Scientific source gate: `INCONCLUSIVE`; authorization gate: `FAIL`.

The frozen baselines are E14 P/C0/C05. The selected anchor checkpoint is a
short E02/5-step smoke checkpoint. Its low source scores are expected from the
untrained short horizon and must not be interpreted as evidence against the
anchor. The detailed rows are in:

- `SOURCE_SOLUTION_GATE_RESULTS.csv`
- `SOURCE_SOLUTION_GATE_HELDOUT.csv`
- `SOURCE_SOLUTION_GATE_AP_TAIL.csv`
- `SOURCE_SOLUTION_GATE_DEPLOYMENT.csv`
- `SOURCE_SOLUTION_GATE_BRANCH.csv`
- `SOURCE_SOLUTION_GATE_STATUS.json`

At E14, the frozen source comparison is:

| method | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|
| P0 | 0.961153 | 0.475602 | 0.985243 | 0.986690 |
| P05 | 0.961103 | 0.475607 | 0.985243 | 0.986690 |
| C0 | 0.937895 | 0.513409 | 0.982205 | 0.984496 |
| C05 | 0.939245 | 0.513512 | 0.982205 | 0.984496 |

The selected E02 smoke rows are recorded, not compared as a matched result:

| method | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|
| anchor smoke C0 | 0.454248 | 0.001601 | 0.518663 | 0.531785 |
| anchor smoke C05 | 0.443515 | 0.001473 | 0.515191 | 0.529611 |

## Required gate dimensions

- Seen/held-out: data are present for the smoke, but the horizon mismatch makes
  the intervention comparison `INCONCLUSIVE_HORIZON_MISMATCH`.
- Representation drift: `INCONCLUSIVE_HORIZON_MISMATCH`; no selected-solution
  E14 representation exists.
- Culprit image drift: `INCONCLUSIVE_HORIZON_MISMATCH`; the implementation
  contract is verified, but a five-step state cannot test cross-epoch drift.
- AP tails: computed for the smoke and preserved, but `INCONCLUSIVE_HORIZON_MISMATCH`.
- Deployment: computed for the smoke and preserved, but
  `INCONCLUSIVE_HORIZON_MISMATCH`.
- Solution hyperparameters: `lambda_image_anchor=1e-3`, image-adapter-only,
  frozen parent E14 reference; selected without Medical/MVTec data.

Because the source gate did not pass at a matched horizon, no full run is
authorized in this task.
