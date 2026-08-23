# P14 Target and Feature Contract

The primary target is the signed, bounded-by-AP-difference image target `V_j`
defined in `P14_PREREGISTRATION.md`.  For an empty expansion band, every band
summary is zero; `safe_fraction` and `expanded_fraction` remain computed.

The exact GT-free image-context feature order is:

1. `safe_fraction`
2. `expanded_fraction`
3. `expansion_band_fraction`
4. `band_boost_fraction`
5. `band_suppress_fraction`
6. `band_risk_median`
7. `band_risk_q90`
8. `band_abs_mu_median`
9. `band_abs_mu_q90`
10. `band_native_rank_median`
11. `band_native_rank_q90`
12. `band_top10_native_rank_fraction`
13. `band_sigma_median`
14. `band_proposal_native_support_median`
15. `band_proposal_peer_support_median`
16. `band_abs_stage_disagreement_median`

The sole value model is float64 centered ridge: training-only median/IQR,
lambda `1.0`, unregularized intercept, and `numpy.linalg.solve`.  It outputs a
value score, not a probability; no nonlinear model or feature search is allowed.
