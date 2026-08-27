# P34 Reporting Wrapper Root Cause and Regression

Status: `P34_REPORTING_BUG_FIXED_ENGINEERING_ONLY`

## Cause

The completed P34 attempt reached the locked prediction freeze and
post-freeze scoring. Final Markdown assembly then raised `KeyError:
'source_only'`. The actual qualification schema stores the report input at
`diagnostics.source_only_actionability`; the report formatter used the stale
`diagnostics.source_only` path. The same formatter also expected old exact
count names (`target_zero`, `weight_one`, and similar), while the frozen
preflight schema uses explicit fraction names such as
`target_exact_zero_fraction` and `weight_one_fraction`.

The gate formatter had a second reporting-contract defect: it stored
`automatic_rerun: false` as if it were a positive structural check, so the
generic failure collector classified the required absence of a rerun as a
failure. This did not change the training, prediction, scoring, or the
scientific endpoint result.

## Engineering-only correction

`tools/sabra_v2/run_p34_scientific_stage2.py` now:

1. validates the frozen `source_only.exact_counts` schema before an attempt
   identity can be created;
2. consumes the actual `source_only_actionability` qualification schema;
3. maps the frozen exact-count field names without recomputation; and
4. represents the no-rerun requirement as the positive structural condition
   `automatic_rerun_forbidden=True`.

The existing P34 scientific evidence was not regenerated, rewritten, or
rescored. The already-recorded engineering stop and scientific stop remain
immutable evidence of the original run.

## Regression evidence

`tests/test_p34_reporting.py` uses frozen qualification metadata and a
synthetic gate payload only. It proves that:

- the actual preflight source schema is validated;
- final-report assembly accepts the actual qualification schema without the
  legacy `source_only` key; and
- a forbidden rerun is not reported as a gate failure.

No held data, model forward, cache rebuild, scientific UUID, or optimizer
step was used for this regression.
