# Medical factorial report

Status: PASS. This report reuses frozen P/C_OLD/A Medical cells and adds only the 36 PA native cells.

Primary contrast: CIR_WITH_ANCHOR = A0 - PA. A0 is the frozen anchored CIR run; PA is the new native Phase2B plus the same image anchor. A05 is intentionally absent from this primary comparison because inference RMT is not a PA factor.

| epoch | metric | n targets | P | C_OLD_0 | PA | A0 | A0-PA | interaction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E10 | pixel_auroc | 6 | 0.879118 | 0.864998 | 0.897496 | 0.892831 | -0.004665 | +0.009455 |
| E10 | pixel_ap | 6 | 0.318093 | 0.299577 | 0.345872 | 0.335465 | -0.010407 | +0.008109 |
| E10 | image_auroc | 3 | 0.699125 | 0.716897 | 0.747029 | 0.724973 | -0.022056 | -0.039828 |
| E10 | image_ap | 3 | 0.722193 | 0.717143 | 0.740768 | 0.735019 | -0.005749 | -0.000700 |
| E12 | pixel_auroc | 6 | 0.887800 | 0.868187 | 0.903053 | 0.896441 | -0.006612 | +0.013001 |
| E12 | pixel_ap | 6 | 0.327435 | 0.301531 | 0.347516 | 0.338826 | -0.008691 | +0.017214 |
| E12 | image_auroc | 3 | 0.702226 | 0.705355 | 0.718347 | 0.744360 | +0.026012 | +0.022883 |
| E12 | image_ap | 3 | 0.724045 | 0.720790 | 0.714472 | 0.745912 | +0.031440 | +0.034695 |
| E14 | pixel_auroc | 6 | 0.882263 | 0.868275 | 0.895874 | 0.897154 | +0.001281 | +0.015268 |
| E14 | pixel_ap | 6 | 0.323308 | 0.310151 | 0.330987 | 0.338738 | +0.007751 | +0.020908 |
| E14 | image_auroc | 3 | 0.677974 | 0.712141 | 0.719612 | 0.725225 | +0.005612 | -0.028555 |
| E14 | image_ap | 3 | 0.704857 | 0.725429 | 0.714410 | 0.732043 | +0.017633 | -0.002940 |
| E16 | pixel_auroc | 6 | 0.881549 | 0.870886 | 0.894735 | 0.897095 | +0.002360 | +0.013023 |
| E16 | pixel_ap | 6 | 0.322745 | 0.308630 | 0.323211 | 0.337676 | +0.014465 | +0.028581 |
| E16 | image_auroc | 3 | 0.681336 | 0.708425 | 0.721135 | 0.729837 | +0.008702 | -0.018387 |
| E16 | image_ap | 3 | 0.710013 | 0.720803 | 0.719587 | 0.740356 | +0.020768 | +0.009978 |
| E18 | pixel_auroc | 6 | 0.884967 | 0.877208 | 0.891985 | 0.895711 | +0.003725 | +0.011485 |
| E18 | pixel_ap | 6 | 0.329389 | 0.306427 | 0.320136 | 0.331511 | +0.011374 | +0.034336 |
| E18 | image_auroc | 3 | 0.684045 | 0.709639 | 0.741586 | 0.743626 | +0.002039 | -0.023554 |
| E18 | image_ap | 3 | 0.709687 | 0.721165 | 0.730483 | 0.752116 | +0.021633 | +0.010156 |
| E20 | pixel_auroc | 6 | 0.879457 | 0.872548 | 0.893050 | 0.894554 | +0.001504 | +0.008412 |
| E20 | pixel_ap | 6 | 0.327511 | 0.302431 | 0.316057 | 0.330561 | +0.014504 | +0.039583 |
| E20 | image_auroc | 3 | 0.689410 | 0.699275 | 0.727438 | 0.738762 | +0.011325 | +0.001460 |
| E20 | image_ap | 3 | 0.709636 | 0.721828 | 0.718889 | 0.745062 | +0.026173 | +0.013981 |

Decision rule applied to the measured six-epoch macro signs: CIR_TRAINING_VALUE=INCONCLUSIVE; FINAL_ARCHITECTURE=MIXED_UNRESOLVED.

Red-team answers:
1. A0 beats PA on source: see SOURCE_FACTORIAL_2X2.csv; source CIR-with-anchor effects are {'pixel_auroc': {'E10': -0.004651609515319621, 'E12': 0.0026874045613030084, 'E14': -0.00021537375273550374, 'E16': 0.0016447749458577965, 'E18': -5.708514807045706e-05, 'E20': -0.002718840104746656}, 'pixel_ap': {'E10': 0.07765177609043744, 'E12': 0.04233333570475367, 'E14': 0.029213945236149286, 'E16': 0.04976353367045655, 'E18': -0.03648644663387701, 'E20': -0.009578627604614032}, 'image_auroc': {'E10': -0.0030381944444444198, 'E12': -0.0390625, 'E14': -0.00477430555555558, 'E16': -0.00520833333333337, 'E18': 0.0017361111111110494, 'E20': 0.0017361111111110494}, 'image_ap': {'E10': -0.007762123906879692, 'E12': -0.024361630986648564, 'E14': -0.003963908765039048, 'E16': -0.00426803639511153, 'E18': 0.0025207983074402307, 'E20': 0.0015244833553904602}}.
2. A0 beats PA on Medical six-domain macro: pixel sign pattern is mixed across the reported epochs.
3. Epoch consistency: pixel AUROC all-positive=False; pixel AP all-positive=False.
4. Domain consistency is reported in MEDICAL_FACTORIAL_2X2.csv; no single-domain result is used as a selection rule.
5. Concentration in one dataset must be judged from the per-domain rows and median effects, not the macro alone.
6. Pixel and image effects are reported separately; a metric-family trade-off is not collapsed into one score.
7. PA reproduces all A gains only if A0-PA is approximately zero across both source and Medical; the measured table is the test.
8. Interaction is the explicit A0-C_OLD_0-PA+P column; its sign and epoch stability are reported without post-hoc tuning.
9. Representation association is in PA_FACTORIAL_DRIFT.csv; it is correlational, not an independent causal intervention.
10. PA is scientifically sufficient only if its simpler trajectory has no robust A0 advantage under the locked protocol.
11. A skeptical reviewer should ask why CIR is retained whenever A0-PA is mixed or near zero.
12. The A0-PA evidence is the measured factorial answer; MVTec remains the untouched confirmatory benchmark and was not run.

No target tuning occurred. No MVTec data were accessed. No new architecture was introduced.
