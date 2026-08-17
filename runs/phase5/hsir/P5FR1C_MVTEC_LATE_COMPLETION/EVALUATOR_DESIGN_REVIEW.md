# P5-FR1C evaluator design review

This static review was completed before GT access and before any scientific
metric was read. The old evaluator `tools/audit_p5fr1_mvtec_posthoc.py` is
preserved unchanged. P5-FR1C uses only the new evaluator
`tools/audit_p5fr1c_mvtec_posthoc.py`.

## Frozen defects and corrections

D1. `exact_sign_flip` used `abs(mean)` and `abs(T_s)`, which is two-sided.
The preregistered family-vs-B1 alternative is family evidence greater than B1,
so P5-FR1C uses the exact one-sided statistic `mean(d)` and counts
`T_s >= T_obs - 1e-15` over all `2^15` assignments.

D2. The old provisional winner used `eligible[0]`. P5-FR1C ranks eligible
families by descending OOF macro mean `delta_vs_B1`, independent of family
enumeration order.

D3. The old evaluator had no preregistered best-vs-runner-up head-to-head test.
P5-FR1C adds exact one-sided paired class sign-flip tests for the best eligible
family against every other eligible family, with Holm correction across those
comparisons.

D4. The old canonical zero-tune track was disabled by an unconditional false
branch. P5-FR1C evaluates each frozen canonical zero-tune configuration as an
adaptability-only track and never uses it for outer-fold selection.

D5. The old `OUTPUT_CHECK` asserted a PASS payload without independently
reconstructing critical invariants. P5-FR1C has a separate
`tools/audit_p5fr1c_outputs.py` checker that reconstructs class counts, fold
isolation, selections, gates, exact sign-flip p-values, Holm adjustment,
ranking, winner state, and zero-tune IDs from scalar/CSV outputs.

D6. The old research-value output hard-coded most components. P5-FR1C freezes
design priors and threshold rules before GT, then computes adaptability,
useful, success, and evidence components mechanically. Research value cannot
override scientific eligibility.

## Isolation and memory

The corrected evaluator loads one class at a time, builds its score, D_rank,
GT, B1, risk, and deterministic matches once, then evaluates the 26 frozen
evidence maps sequentially. It retains only scalar per-class/config rows and
discards class-resolution arrays before moving on. It has no model loader,
checkpoint load, model forward, training, or medical path.
