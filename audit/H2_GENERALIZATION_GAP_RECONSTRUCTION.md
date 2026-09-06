# H2 generalization-gap reconstruction

## Result in one sentence

The frozen Seed-0 A-E15 result is not separated from the contextual published result by one uniform accuracy deficit: Medical is chiefly an AP/top-of-ranking problem despite near-matched AUROC, while MVTec combines lower macro AUROC with contextually higher AP and a small set of severe class regressions. **[DERIVED; DESCRIPTIVE_TARGET_POSTHOC]**

## Scope and comparability

- Frozen comparison: H versus Safe Anchor A at Seed 0, E15. **[MEASURED]**
- Medical and MVTec numbers come from the repository's already-computed result CSVs; no target inference or tuning was run for this audit. **[MEASURED]**
- Public ACD-CLIP values are approximate contextual references (Medical 91.55 AUROC / 43.03 AP; MVTec 91.4 / 43.6). Protocol match is partial, so differences are not claims of exact superiority or inferiority. **[PUBLISHED]**
- Seed-1/2 target confirmation is unavailable because those runs were invalidated by the hard H/A global-step mismatch. Seed-0 patterns are discovery evidence, not replicated estimates. **[MEASURED]**

## Medical: AUROC is near the contextual reference but AP remains lower

At E15, A reaches 91.251786 pixel AUROC and 39.468370 pixel AP versus H at 90.815332 / 35.874253. A therefore adds 0.436454 percentage points AUROC and 3.594117 points AP. Relative to the approximate public context, A is only about 0.30 points lower in AUROC but about 3.56 points lower in AP. **[MEASURED; DERIVED; PUBLISHED]**

The dataset pattern is heterogeneous:

| Dataset | A−H AUROC (pp) | A−H AP (pp) | Pattern |
|---|---:|---:|---|
| Brain | +0.777 | +4.120 | win both |
| Colon_Kvasir | +0.932 | +0.972 | win both |
| Colon_clinicDB | +1.675 | +6.502 | win both |
| Colon_colonDB | −1.106 | +3.140 | AUROC down, AP up |
| Liver | −0.332 | −0.642 | regression both |
| Retina | +0.673 | +7.473 | win both |

Why can AUROC be near the reference while AP is materially lower? AUROC weights positive–negative pair ordering across the entire score distribution; with abundant easy background pixels, it can remain high even when a relatively small high-score background tail intrudes into the precision-critical leading ranks. AP is prevalence-sensitive and emphasizes those early ranks. The observed AUROC/AP discordance on Colon_colonDB and the concentration of AP gains on Retina, ClinicDB, and Brain are consistent with a residual tail/localization problem rather than a uniform separability loss. **[DERIVED; HYPOTHESIS]** Target score maps were not stored, so the exact Medical false-positive and false-negative tails remain **[UNKNOWN]** rather than measured.

## MVTec: AP is contextually high while AUROC remains lower

At E15, A reaches 90.041289 pixel AUROC and 45.159349 pixel AP versus H at 86.868623 / 41.612306, gains of 3.172666 and 3.547043 points. Against the approximate public context, AP is about 1.56 points higher while AUROC is about 1.36 points lower. Because protocols only partially match, that cross-paper sign difference is diagnostic context, not an exact benchmark win/loss. **[MEASURED; DERIVED; PUBLISHED]**

A improves AUROC in 11/15 classes and AP in 11/15, but only 10/15 improve both. Cable (−10.945 AUROC, −0.169 AP), leather (−0.169, −8.564), and pill (−4.090, −6.014) regress on both metrics. Toothbrush loses AUROC but slightly gains AP; zipper gains AUROC but loses AP. Large gains on metal_nut AUROC (+24.597), hazelnut AP (+29.761), and screw AP (+18.154) materially affect the macro averages. **[MEASURED; DERIVED]**

Why can AP exceed the contextual value while AUROC is lower? Macro AP can rise when anomaly pixels are concentrated near the top of class-specific rankings, even if numerous mid-ranking positive/background inversions reduce pairwise AUROC. The large, concentrated class gains combined with cable/pill/leather regressions are consistent with class- and morphology-dependent ranking rather than a universal threshold shift. **[DERIVED; HYPOTHESIS]** Exact cross-paper attribution is **[UNKNOWN]** because class weighting, preprocessing, checkpoint selection, and other protocol details are not fully matched.

## Residual failure dimensions must remain separate

1. **Pixel ranking/localization:** supported as a candidate by AUROC/AP disagreement, but target curves/maps are unavailable. **[POSSIBLE; DESCRIPTIVE_TARGET_POSTHOC]**
2. **Small versus large anomalies:** existing mixed Seed-0 target artifacts do not contain morphology-linked maps, so the target-side relationship is **[UNKNOWN]**. The frozen VisA source audit supplies the non-target diagnostic evidence.
3. **Boundary versus interior:** unavailable on frozen mixed target outputs and therefore **[UNKNOWN]** target-side. Source-only boundary/interior measurements are reported separately.
4. **Dataset/class regressions:** directly observed and material; a single aggregate improvement does not explain them. **[MEASURED]**
5. **Source overfit or domain shift:** consistent with heterogeneous target transfer but not established by target aggregates alone. **[HYPOTHESIS]**

## Evidence sources

- `results/H2_4ARM_E15_MEDICAL_SUMMARY.csv`
- `results/H2_4ARM_E15_MEDICAL_PER_DATASET.csv`
- `results/H2_4ARM_E15_MVTEC_SUMMARY.csv`
- `results/H2_4ARM_E15_MVTEC_PER_CLASS.csv`
- `results/H2_DATASET_CLASS_BOTTLENECK.csv`
- `results/H2_PIXEL_ERROR_DECOMPOSITION.csv`
- `audit/H2_GENERALIZATION_FORENSICS.md`

