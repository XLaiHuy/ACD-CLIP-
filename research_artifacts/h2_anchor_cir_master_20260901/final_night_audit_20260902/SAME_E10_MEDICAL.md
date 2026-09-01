# Same-E10 Medical comparison

Status: COMPLETE. This is an evaluation-only comparison of the already completed H2 master runs. All three arms use the native H2 deployment path with `deployment_alpha=0.0`; no inference-time RMT was applied.

The arms are:

- `R`: current shared-E0 control, Anchor off, CIR off.
- `RA`: current shared-E0 trajectory with the historical per-tensor relative-L2 image Anchor.
- `RCA`: current shared-E0 trajectory with the same Anchor plus train-time CIR/RMT.

The output CSV contains the exact per-target values and checkpoint/config/evaluator identities.

| Method | Target | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---|---:|---:|---:|---:|
| R | Brain | 0.944438743 | 0.409820653 | 0.819920732 | 0.950281912 |
| R | Liver | 0.945245611 | 0.055799077 | 0.625412892 | 0.491126445 |
| R | Retina | 0.922691132 | 0.378323450 | 0.791146124 | 0.796632000 |
| R | Colon_clinicDB | 0.884070058 | 0.485659910 | — | — |
| R | Colon_colonDB | 0.813886645 | 0.267778756 | — | — |
| R | Colon_Kvasir | 0.855465327 | 0.529169570 | — | — |
| RA | Brain | 0.960660458 | 0.571914468 | 0.836819614 | 0.957780430 |
| RA | Liver | 0.944224992 | 0.047378128 | 0.690121503 | 0.564036324 |
| RA | Retina | 0.886757166 | 0.311696701 | 0.771241029 | 0.709549221 |
| RA | Colon_clinicDB | 0.855879833 | 0.493718656 | — | — |
| RA | Colon_colonDB | 0.794153449 | 0.306858296 | — | — |
| RA | Colon_Kvasir | 0.829081610 | 0.498522588 | — | — |
| RCA | Brain | 0.962055141 | 0.560289702 | 0.849631098 | 0.962887500 |
| RCA | Liver | 0.934945233 | 0.066156610 | 0.731729055 | 0.585634991 |
| RCA | Retina | 0.865719115 | 0.338953550 | 0.777542234 | 0.761322206 |
| RCA | Colon_clinicDB | 0.867986612 | 0.496972106 | — | — |
| RCA | Colon_colonDB | 0.782272108 | 0.310733358 | — | — |
| RCA | Colon_Kvasir | 0.806309398 | 0.471723815 | — | — |

## Macro summaries

| Method | Six-target pixel AUROC | Six-target pixel AP | Supported-image AUROC (3) | Supported-image AP (3) |
|---|---:|---:|---:|---:|
| R | 0.894299586 | 0.354425236 | 0.745493249 | 0.746013452 |
| RA | 0.878459585 | 0.371681473 | 0.766060715 | 0.743788659 |
| RCA | 0.869881268 | 0.374138190 | 0.786300796 | 0.769948232 |

Same-E10 deltas, computed from the CSV:

| Contrast | Pixel AUROC | Pixel AP | Interpretation |
|---|---:|---:|---|
| RA − R | −0.015840001 | +0.017256237 | Anchor training effect conditional on this current trajectory |
| RCA − RA | −0.008578317 | +0.002456717 | CIR/RMT training increment conditional on the pathological Anchor trajectory |

The same-E10 RA-versus-R result is mixed by domain: Brain improves, while Retina, both Colon variants, and Kvasir lose Pixel AUROC; AP also moves in both directions. RCA-versus-RA is likewise mixed. The supported image metrics rise for RCA, but the primary six-target pixel result does not show a robust gain.

These are not clean historical-H2 causal estimates. The historical H2 E10 result is a separate retrospective reference, the historical run has no recorded seed/RNG state, and `RCA` inherits the ill-conditioned Anchor. An inference-only alpha comparison is not run in this final-night audit; the old pre-fix `inference_rmt_effect.csv` remains conditional on the buggy-trained representation.
