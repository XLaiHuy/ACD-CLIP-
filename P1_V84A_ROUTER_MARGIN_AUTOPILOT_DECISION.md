# P1-v8.4-A Router Margin Autopilot Decision

Base: `e55cf7c6d77b022e5ebfe3d18a59138548463be1`. This workflow made no model or data-state mutation and launched no training, calibration, replay, or Router smoke run.

## Source and diagnostic optimization

The Router confidence mode is now explicitly decoupled:

```text
entropy (default): best_gain > router_gain_threshold and entropy < threshold
margin_rel:        best_gain > 0 and margin_rel > 0.10 and valid
margin_rel:        (best_gain - second_gain) / max(abs(best_gain), 1e-12)
```

Only Router `informative` eligibility changes in `margin_rel` mode. Router q, factor q/responsibility, factor teacher, ACT routed teacher/loss, residual correction, rho, architecture, optimizer, and surgery are unchanged. The entropy default retains the legacy gate exactly, including P1-v8.3 behavior.

Trajectory diagnostics now cache concatenations within an aggregation and offer opt-in `fast` aggregation: intermediate milestones report an exact recent block while exact cumulative quantiles/AUROC are deferred to the final milestone. `legacy` remains the default. Exact AUROC uses vectorized average tie ranks; all-positive/all-negative returns `None`, and a constant score with both labels returns `0.5`.

Focused verification: 69 tests passed, plus `py_compile` and `git diff --check`. The no-training benchmark is stored in `runs/p1_v84a_gpu/router_margin_autopilot/diagnostic_optimization.json`: legacy-small and optimized-small AUROC agree exactly; the optimized large case ran without invoking the pathological legacy reference.

## Frozen candidate provenance

`post300_teacher_semantics_audit.json` defines margin support as `best_gain > 0 and margin_rel > threshold`, matching the implementation. At threshold 0.10, frozen support is 1,071,260 overall (99.428%), 1,066,750 normal (99.592%), and 4,510 anomaly (71.553%). The anomaly-selected winners are F2=2,215, F3=1,895, F4=400; therefore the numerical normal/F1 imbalance alone is not a stop condition.

## Target-sharpness gate

The frozen artifact does not contain Router q distributions at tau 0.05/0.03/0.02 restricted to the same margin-selected patches, nor their normal/anomaly/factor-winner splits. Its available tau rows are global entropy-gated summaries and cannot reconstruct the required selected-set targets.

Decision: `MISSING_ROUTER_TARGET_SHARPNESS_EVIDENCE`.

No 6-microbatch lambda calibration and no Router 8B smoke run are authorized from this evidence. Do not change tau, threshold, loss weighting, capacity, factor/ACT semantics, or Router formulation further. The next action is discussion of how to obtain target-sharpness evidence without silently changing the scientific protocol.

EXIT_FOR_DISCUSSION
