# P20 JSON Sentinel Contract

## Read-only inventory

P19 writes `value_threshold` in `parameters.json` and `fold_summary.json`; it is read through JSON only for audit/provenance. The sole legitimate non-finite control value is `+inf` when the frozen source-only selection records `selected=NO_EXPANSION`. P19 computes it as `float('inf')` and deploys `expand_image = (vhat > value_threshold)`. No persisted P19 JSON contains `NaN`, `-Infinity`, or another legitimate non-finite control sentinel.

## Canonical boundary encoding

The runtime threshold remains an IEEE `float('inf')`. Persisted JSON is strict (`allow_nan=False`):

| Runtime value | `value_threshold` | `value_threshold_encoding` |
| --- | --- | --- |
| finite float | exact finite float | `FINITE` |
| positive infinity | `null` | `POSITIVE_INFINITY` |

The decoder maps only these two consistent pairs back to a float. It rejects NaN, negative infinity, unknown encodings, null finite values, and finite values tagged as infinity. `None` is never used in scientific computation.

This encoding is applied to every P20 persisted JSON location carrying `value_threshold`. All model and policy calculations happen before encoding and after decoding use the original runtime float.
