# R3 final freeze summary

BEST_VERIFIED_CONFIGURATION

Canonical H2 A15 checkpoint at
`runs/h2_clean_factorial_e20_20260902/A/adapter_15.pth`, exact checkpoint
SHA256 `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe`,
under the locked TTA-F policy and exact evaluator contracts. The R3 E15
confirmation remains recorded, but its Medical Pixel AP was lower than this
previously verified A15 audit.

SOURCE_SCREEN_RESULT

`HARD_BACKGROUND_SOURCE_GATE=FAIL`. The best approved candidate was lambda
`0.02` at source TTA-F Pixel AP `23.8179621%`, versus locked baseline
`24.6335168%`. Candidate lambda `0.05` scored `20.6178665%`.

MEDICAL_PIXEL_AUROC

`91.75512323421079`

MEDICAL_PIXEL_AP

`40.2459204527138`

MEDICAL_IMAGE_AUROC

`74.68376755714417`

MEDICAL_IMAGE_AP

`76.70050263404846`

MVTEC_PIXEL_AUROC

`90.41629629847156`

MVTEC_PIXEL_AP

`47.29750359996327`

MVTEC_IMAGE_AUROC

`90.77691078186035`

MVTEC_IMAGE_AP

`95.27245442072551`

PUBLISHED_CONTEXT_TARGET = `43.03`

TARGET_DELTA

Medical Pixel AP minus published-context target: `-2.7840795472862` percentage
points.

FINAL_DECISION: `KEEP_PREVIOUS_BEST`

FAILED_NOVELTY_IDEAS

Training-only hard-background patch ranking, screened at lambda `0.02` and
`0.05`, failed the source gate. Neither failed candidate was target-evaluated.

REMAINING_UNTESTED_IDEAS

No previously preregistered next novelty mechanism satisfies the strict
post-failure policy. No additional idea is being introduced or tested.

CLAIM_BOUNDARY

The result supports a reproducible source-only negative finding for this
hard-background ranking mechanism and a locked target audit of the previously
verified A15 model. The published-context target was not met. No target labels
or target metrics were used to select the failed novelty candidates, and no
claim of target improvement is made.
