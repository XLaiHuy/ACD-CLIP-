# Matched-horizon corrected results

Status: PASS for the locked E14 source-only experiment.

Frozen P/C0 E10/E12/E14 source rows were reused. Only the new CIR checkpoint was forwarded, producing C0 (alpha 0) and C05 (alpha .5).
C05 minus C0 is the conditional inference RMT effect on the new anchored representation; it is not a clean training effect.

| epoch | P pixel AUROC | C0 pixel AUROC | C05 pixel AUROC | train effect | RMT inference effect | total CIR effect |
|---:|---:|---:|---:|---:|---:|---:|
| E10 | 0.971833 | 0.936343 | 0.935898 | -0.035490 | -0.000444 | -0.035935 |
| E12 | 0.960599 | 0.962230 | 0.962570 | +0.001631 | +0.000340 | +0.001971 |
| E14 | 0.961153 | 0.958567 | 0.958819 | -0.002586 | +0.000252 | -0.002334 |

Medical evaluation: NOT_RUN. MVTec evaluation: NOT_RUN. No final cross-domain RMT decision is made from this bounded source-only stage.
The scientific purpose of this run is to establish the matched-horizon timing and source decomposition under the selected E14 anchor intervention.
