# P25R2 — Exact Batched Patch Actionability Recovery V1

P25R2 begins at P25R terminal `87d3c15b6fe4f62762bc87760960c1f83eda90d3`.
P25R had zero runs, targets, Q1, and Q2 results.  This recovery retains the
native-anchored patch-benefit hypothesis and replaces only the invalid sparse
post-deployment target reconstruction with direct frozen batched deployment.

For each of the 12 VisA source classes, exactly 2,000 panel patches are chosen
without GT by native-score-rank quintile x deployment-sensitivity quintile;
the frozen global cap is 16 patches/image and each of the 25 strata contributes
80 records.  The selection order is SHA256 of class, image path, and patch
index, followed by image path and patch index.

For source patch j, `V_j` is exact class pixel AP after its sole correction
minus exact native class pixel AP.  The correction is source utility-sign,
alpha=.25, and the frozen correction scale.  Candidate logits are replicated
within an image batch, given exactly one non-zero patch correction per row,
and passed through the actual frozen industrial deployment function.  `V` is
source supervision only; it is neither additive nor a deployable feature.

`BENEFIT_EPS=1e-10`.  It was derived before target generation as
`max(1e-10, 20 * max batch-route AP difference)`; the deterministic candle
128-candidate batch=1 versus batch=16 audit observed zero score, order, and AP
difference.

The 32 fixed GT-free features are the frozen R2-v2 22D harm order; harm risk;
harm-policy action; support native-rank median/q90; signed delta mean over
image IQR; absolute delta q90 over image IQR; support-rank-shift median/q90;
and top-5/top-20 rank-boundary crossing fractions.  The only benefit model is
a deterministic float64 zero-initialized linear pairwise logistic ranker,
L2=1, within-class non-adjacent-decile pairs, at most 8,192 pairs/class.

Strict 12-class LOCO is required.  All source direction/harm quantities are
nested class-excluded; held labels remain closed until models, calibration,
benefit scores, and held action artifacts are frozen.  Q1 gates and conditional
Q2 policy/gates are exactly those in the P25R2 recovery authorization.

One marker is permitted only after a published clean execution base.  MVTec,
Medical, new CLIP forwards, Phase2B training, prompts, alpha sweeps, and model
search are forbidden.
