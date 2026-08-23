# SABRA-CURE R2-v2 Harm-Aware Selective Intervention v1

Status: `FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS`.
Base: `dbd0666898f19864cf48d36a06243021b03d13fc`.

R2-v2 is one 12-fold VisA outer-LOCO confirmatory study. It preserves R2 as
`R2_SCIENTIFIC_STOP`, uses no MVTec, Medical, CLIP forward, Phase2B training,
adaptive alpha, magnitude regression, or new representation. The sole fixed
actuator is abnormal-logit-only, shared-stage signed alpha `.25`.

Direction is exactly R1: the frozen 14 GT-free features in committed R1 order,
training-only median/IQR, float64 centered ridge lambda 1.0 with unregularized
intercept and `numpy.linalg.solve`, and `y=tanh(u/P75(abs(u_train)))`.

The 22 risk features are the 14 direction features plus, in this exact order:
`mu`, `abs(mu)`, `sigma`, `abs(mu)/(sigma+1e-8)`, `sign(mu)*signed_native_margin`,
`sign(mu)*robust_peer_signed_margin_consensus`, `sign(mu)*cross_stage_signed_margin_difference`,
and `abs(cross_stage_signed_margin_difference)`. Level-1 training features use
strict inner-OOF `mu_cf`; held inference uses outer `mu`.

Primary target: `harm=1[sign(mu_cf)!=sign(y_cf)]*abs(y_cf)`, finite in `[0,1]`.
Matched ablation target: `binary_wrong=1[sign(mu_cf)!=sign(y_cf)]`. Both risk
heads are identically scaled float64 ridge lambda 1.0, with no calibration.

For each outer holdout, Level-1 direction OOF groups exclude their class. Level-2
risk OOF scores exclude their class again. Threshold is exactly
`numpy.quantile(level2_score, .20, method='linear')`; accept `risk<=tau`, else
KEEP. Ties retain the `<=` rule. Fixed descriptive coverage points are
`{.10,.20,.30,.40,.50}` and cannot alter deployment.

Comparators are native, direction-only fixed alpha, persisted published R2,
matched binary-risk (ablation), and primary harm-risk. Oracle post-R2 cohorts
are excluded from candidates and selection. Exactly one attempt is allowed.
