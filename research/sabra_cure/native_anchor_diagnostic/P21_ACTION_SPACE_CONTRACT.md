# P21 Action-Space Contract

NATIVE is no SABRA correction. SAFE20, SAFE30, and EXPAND40 apply frozen direction, frozen harm risk, source-only tau20/tau30/tau40 respectively, deterministic `risk <= tau`, and alpha `0.25`. Runtime thresholds are float64; SAFE30 alone is `quantile(source OOF harm risk,.30,method="linear")` and may exist only if Stage C is routed.

A0 uses action order `NATIVE < SAFE20 < EXPAND40`; A1 adds `SAFE30` between SAFE20 and EXPAND40. For each held class, cyclic coordinate witness seeds are all-NATIVE and P20 image-oracle (or best A0 for A1). Images are visited in frozen `image_path` order. A candidate changes state only for pAP gain `>1e-12`; ties choose the most conservative action. Stop at zero changes or ten sweeps. This is explicitly `POST_HOC_MULTI_START_COORDINATE_WITNESS`, never a global oracle or deployable policy.

Strong headroom requires wrong-sign <=.05, weighted-harm reduction >=.50, macro pAP >= native+.0025, >=9 non-regressing classes, >=7 improving classes, and frozen pAUROC guardrail. A0 strong skips SAFE30; A1 weak terminates P21 as `P21_CONTEXTUAL_ACTION_SPACE_INSUFFICIENT` and skips probes.
