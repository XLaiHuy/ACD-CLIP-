# SABRA-CURE R2 Final Decision

Decision: `R2_SCIENTIFIC_STOP`

This is the sole authorized frozen 12-fold VisA LOCO R2 execution, from
execution base `fccea49380c225011f157fbf2c97701c1a624e1e` and preregistration
freeze `83caf6f3c221d39b4b0b2bdd79483cf9ba8b42cc`. The R2 controller met its
source-only safety gates, but it did not meet the frozen downstream gates.

| Gate | Result | Observed |
| --- | --- | --- |
| R2_G1 audit | PASS | all pre/post audits passed |
| R2_G2 accepted wrong-sign | PASS | 3.8861% (required <=5%) |
| R2_G3 coverage | PASS | 20.3712% (required >=10%) |
| R2_G4 risk reduction | PASS | 47.9373% (required >=25%) |
| R2_G5 macro pixel AP | FAIL | -0.4373 pp vs native (strictly positive required) |
| R2_G6 pAP breadth | FAIL | 7/12 non-regressing classes (required >=9) |
| R2_G7 macro pixel AUROC | FAIL | -0.8372 pp (required >=-0.50 pp) |

The selective policy had a qualified operating point in six held classes and
emitted deterministic KEEP in the other six. There is no R2 retry, threshold
change, R3, R4, MVTec evaluation, or Medical evaluation authorized from this
terminal result. See `results/sabra_cure/r2/summary.json` and
`results/sabra_cure/r2/post_execution_audit.json` for complete evidence.
