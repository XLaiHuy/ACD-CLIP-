# P25R Gates and Policy Contract

Q1 requires: positive and negative V support in >=10/12; median held Spearman
>=.20; positive Spearman in >=9/12; macro V-positive sign AUROC >=.65; macro
Benefit-Capture@20 >=.35; and capture >.20 in >=9/12. Transfer unit is held
category, not panel patches.

Only after Q1 pass, source-OOF calibration selects one of risk percentiles
`.40,.60,.80` x benefit percentiles `.80,.90,.95`, using linear quantiles.
ACT requires nonzero direction, risk <= source threshold, and benefit score >
source threshold. Policies with source wrong-sign >5% or weighted-harm
reduction <50% are ineligible. Choose highest class-balanced source normalized
target value, with fixed ties: positive-class count, lower risk percentile,
higher benefit percentile. This criterion is not pAP.

Q2 requires wrong-sign <=5%, weighted-harm reduction >=50%, macro pAP >=
native+.0025, non-regressing >=9/12, improving >=7/12, pAUROC >= native-.005,
and pAP above frozen harm-only R2-v2. Held candidates are never swept.
