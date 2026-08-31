# Corrective matched retrain results summary

Status: COMPLETE. The E14 image-parameter-anchor continuation resumed the existing E14 cursor and trained through E20 under the matched Adam/StepLR protocol.

P is the matched Phase2B parent; C_OLD is the previously trained CIR run; A is the anchored CIR run. P and C_OLD Medical rows were reused from the frozen exact evaluation. Only A0/A05 cells were newly evaluated after the target-blind freeze.

| epoch | P pixel AUROC | C_OLD_0 pixel AUROC | A0 pixel AUROC | A05 pixel AUROC | A0-P | A05-A0 |
|---:|---:|---:|---:|---:|---:|---:|
| E10 | 0.879118 | 0.864998 | 0.892831 | 0.892659 | +0.013713 | -0.000172 |
| E12 | 0.887800 | 0.868187 | 0.896441 | 0.896344 | +0.008641 | -0.000097 |
| E14 | 0.882263 | 0.868275 | 0.897154 | 0.897146 | +0.014892 | -0.000008 |
| E16 | 0.881549 | 0.870886 | 0.897095 | 0.897045 | +0.015546 | -0.000050 |
| E18 | 0.884967 | 0.877208 | 0.895711 | 0.895618 | +0.010743 | -0.000093 |
| E20 | 0.879457 | 0.872548 | 0.894554 | 0.894449 | +0.015097 | -0.000105 |

Final decision: `KEEP_ANCHOR_DISABLE_INFERENCE_RMT_CANDIDATE`.

The complete per-domain matrix, macro definitions, and target deltas are the authoritative numeric artifacts. A05-minus-A0 is the conditional inference RMT effect on the anchored representation; it is not a clean CIR-vs-Phase2B effect.

Target tuning: NO. MVTec: NOT_RUN.
