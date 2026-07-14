# Phase2D A-prime/B Checkpoint Interpolation Preregistration

## Scientific question

Can weight-space interpolation between the selected A-prime and B checkpoints
preserve A-prime's Pixel AP while recovering some of B's higher Pixel AUC,
without retraining?

## Parents

| Parent | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-prime | 13 | 94.8038 | 55.5341 | 97.9028 | 98.4225 |
| B | 13 | 96.2236 | 55.1342 | 97.8750 | 98.4287 |

## Locked candidates

Lambda is the B weight:

`theta_AB = (1 - lambda) * theta_A + lambda * theta_B`

| Candidate | Lambda |
| --- | ---: |
| AB25 | 0.25 |
| AB50 | 0.50 |
| AB75 | 0.75 |

No additional lambda may be introduced after validation results are observed.

## Interpolation scope

- Interpolate only corresponding floating-point tensors in the model state.
- Perform arithmetic in FP32 and cast each result back to the A-prime tensor dtype.
- Copy matching non-floating tensors from A-prime, after requiring exact equality.
- Do not interpolate optimizer, scheduler, scaler, epoch, or training history.
- Preserve the A-prime payload structure and add explicit interpolation metadata.

## Evaluation protocol

- Dataset: VisA, existing fixed seed-42 manifests.
- Same model architecture, image size, `cls_only` image-scoring rule, and validation implementation.
- No training, augmentation changes, or medical evaluation.
- Use one locked validation batch size and worker count for A-prime, B, AB25, AB50, and AB75.

## Parent reproduction gate

Re-evaluate A-prime and B through the new evaluation path before evaluating
interpolated candidates. Every reproduced macro metric must be within 0.05
percentage points of its registered historical value. If this fails, do not
evaluate or select interpolation candidates; report the environment or pipeline
mismatch without tuning the pipeline to force agreement.

## Primary success

An interpolation candidate must satisfy both:

- Image AP >= 97.4225
- Pixel AP > 55.5341

## Secondary Pareto criterion

An interpolation candidate must satisfy all:

- Pixel AUC > 94.8038
- Pixel AP >= 55.0341
- Image AP >= 97.4225

## Ranking

Among eligible interpolation candidates, rank by:

1. Highest Pixel AP.
2. Highest Image AP.
3. Highest Pixel AUC.
4. Smaller B interpolation weight.

No new scalar that averages metrics will be used.

## Stop decision

- If a primary winner is found, stop and do not run LB_0p1.
- If there is no primary winner, close the interpolation test after AB25, AB50, and AB75.
- Only after closure may LB_0p1 be preregistered as a later experiment.
- LB_0p1 is not implemented in Phase2D.
