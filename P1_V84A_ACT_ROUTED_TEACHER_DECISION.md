# P1-v8.4-A ACT Routed-Teacher Decision

## 1. Previous mismatch evidence

The committed frozen teacher-semantics audit proved `ACT_TEACHER_OBJECTIVE_MISMATCH`. In normal patches, ResidualOracleMulti improved Base by `0.782201%`, while the ACT=1 routed mixture changed performance by `-0.605282%`; `99.523956%` of normal patches were oracle-positive/routed-harm. The best-factor utility was therefore not the utility of the action ACT gates.

## 2. Source semantic change

P1-v8.4-A ACT labels now use `g_route`, the detached relative BCE gain of the current Router-weighted residual mixture with ACT forced to one:

`routed_delta = sum_m(router_probability_m * factor_residual_m)`

`g_route = (L_base - BCE(z_base + rho * routed_delta, target)) / max(L_base, denominator_floor)`

The previous `g_best` object remains available for factor/Router utility logic but no longer labels ACT. The ACT zones remain `g_route > 0.02` ON, `g_route <= 0` OFF, and `0 < g_route <= 0.02` ambiguous. No positive `T_off` was introduced.

## 3. Isolation tests

The focused suite passes `60/60`. Tests prove the oracle-positive/routed-harm counterexample labels OFF, routed gain above threshold labels ON, small positive routed gain is ambiguous, and changing only the routed mixture can change ACT labels. The same mixture change leaves factor responsibility, Router q, normalized entropy, and Router informative masks exactly unchanged. Existing P1-v8.3, exact ACT=0 no-op, and zero-logit/probability=0.5 contracts also pass.

## 4. Current threshold consequence

The committed audit artifact was checked programmatically. Current `act_gain_threshold = 0.02`, while maximum frozen `g_route = 0.004175810609012842`; therefore `0.02 > max(g_route)` and:

`ACT_POSITIVE_SUPPORT_AT_CURRENT_THRESHOLD = 0`

The source change does not lower or otherwise recalibrate the threshold.

## 5. Frozen positive-g_route distribution

- p25: `0.0007460805936716497`
- p50: `0.0019944568630307913`
- p75: `0.002651029732078314`
- p90: `0.003021989017724991`
- p95: `0.0036714801099151373`
- max: `0.004175810609012842`
- natural OFF boundary: `g_route <= 0`

These values are diagnostic context only; no `T_on` is selected from support count.

## 6. Decision

`ACT_GAIN_THRESHOLD_RECALIBRATION_REQUIRED`

`EXIT_FOR_DISCUSSION`

## 7. Next discussion

Choose and scientifically justify only `T_on` using the existing frozen risk/coverage evidence. Hold the natural OFF boundary at zero and every other control fixed.

## 8. Forbidden next actions

No training, replay, Router formulation change, Router margin eligibility, loss rebalance, positive `T_off`, capacity change, or automatic threshold selection.
