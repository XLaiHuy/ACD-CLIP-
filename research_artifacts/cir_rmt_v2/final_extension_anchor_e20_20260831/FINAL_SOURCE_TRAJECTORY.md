# Final source trajectory

Status: PASS. This is a source-only, deterministic 96-image VisA matrix.

P is the matched Phase2B parent alpha-0 source row; C_OLD_0/C_OLD_05 are reused from the frozen prior CIR run; A0/A05 are the image-parameter-anchor continuation. New GPU forwarding was limited to A E16/E18/E20. A E10/E12/E14 rows were reused from the completed E14 anchor source stage.

The primary decomposition is: anchor training effect = A0 - P; conditional inference RMT effect = A05 - A0; total anchored CIR effect = A05 - P. These are source-sample associations and do not substitute for the target-domain freeze.

| epoch | P pixel AUROC | C_OLD_0 pixel AUROC | A0 pixel AUROC | A05 pixel AUROC | A0-P | A05-A0 |
|---:|---:|---:|---:|---:|---:|---:|
| E10 | 0.971833 | 0.927512 | 0.936343 | 0.935898 | -0.035490 | -0.000444 |
| E12 | 0.960599 | 0.937280 | 0.962230 | 0.962570 | +0.001631 | +0.000340 |
| E14 | 0.961153 | 0.937895 | 0.958567 | 0.958819 | -0.002586 | +0.000252 |
| E16 | 0.954718 | 0.951339 | 0.963720 | 0.964033 | +0.009002 | +0.000313 |
| E18 | 0.969225 | 0.956966 | 0.968500 | 0.968353 | -0.000726 | -0.000146 |
| E20 | 0.961068 | 0.959783 | 0.971399 | 0.971118 | +0.010332 | -0.000281 |

AP-tail, raw-vs-deployed, image-branch, and held-out-category diagnostics are in the companion CSVs. Representation preservation is reported separately in SAME_EPOCH_FEATURE_DRIFT.csv and SAME_EPOCH_PARAMETER_DRIFT.csv.

No Medical or MVTec data were accessed by this stage.
