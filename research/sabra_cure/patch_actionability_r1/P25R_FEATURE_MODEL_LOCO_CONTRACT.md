# P25R Feature, Model, and LOCO Contract

The fixed 32D feature order is F01--F22 frozen R2-v2 `HARM_ORDER`:
the R1 14D direction/trust order followed by `mu`, `abs_mu`, `sigma`,
`standardized_direction_strength`, `proposed_native_margin_support`,
`proposed_peer_margin_support`, `proposed_stage_difference`, and
`absolute_stage_difference`; then F23 `harm_risk`, F24 `harm_policy_action`,
F25 `support_native_rank_median`, F26 `support_native_rank_q90`, F27
`signed_delta_mean_over_image_iqr`, F28 `abs_delta_q90_over_image_iqr`, F29
`support_rank_shift_median`, F30 `support_rank_shift_abs_q90`, F31
`top5_boundary_cross_fraction`, F32 `top20_boundary_cross_fraction`.

All deployed inputs are GT-free. The sole model is deterministic float64
advantage-weighted linear pairwise logistic ranking, zero initialized, L2=1,
fixed CPU L-BFGS. Pairs are within source class only, use V percentile deciles,
skip same/adjacent deciles, are deterministic/balanced, and cap at 8192 pairs
per source class. No feature/model/loss/lambda search is allowed.

For held H, all scalers, ranker fits, pairs, calibration, and policy selection
use only the other 11 classes. Source J feature predictions use nested
class-excluded direction/harm predictions that exclude J and H. H labels open
only after direction, harm, ranker, thresholds, and held actions are frozen.
