# P18 Aggregation Sufficiency Audit

## Scope and frozen source

This read-only audit examines the exact P14 aggregate implemented in
`tools/sabra_cure/context_value_risk.py` at the immutable P17 terminal
`1c7cba7a65c09d6d796becc66bafa9b8dd917833`.  P18 permits a parent to read
only compact JSON summaries; it may not deserialize `fold.npz` or read
image-level `V_j`/prediction arrays.  No P18 attempt has been created.

## Exact aggregate inventory

| Frozen result / gate | Exact fold-level sufficient statistic | Parent needs scientific array? |
| --- | --- | --- |
| Macro native, SAFE20, EXPAND40, context, and image-oracle pAP | one class pAP scalar per comparator | NO |
| Macro native/context pAUROC | one class pAUROC scalar per comparator | NO |
| Macro loss diagnostics | one class loss scalar per comparator | NO |
| Global action coverage | accepted-action count; total patch count | NO |
| Global accepted wrong-sign rate | accepted count; wrong-sign accepted count | NO |
| Global weighted-harm reduction | accepted count; wrong-sign `sum(abs(y))`; baseline-wrong count; baseline-wrong `sum(abs(y))` | NO |
| Global EXPAND40 image fraction | expanded-image count; image count | NO |
| No-expansion-fold count and selected-q list | selected q / null per held class | NO |
| Non-regressing and improving breadth | native/context pAP scalars per held class | NO |
| G1 audit | compact PASS bit from every audit child | NO |
| G2--G5 | the coverage, wrong-sign, weighted-harm, and expansion statistics above | NO |
| G6--G10 | pAP/pAUROC comparator scalars above | NO |
| G11 selection | selected-q scalar/null per held class | NO |
| Value Pearson | `n`, `sum(x)`, `sum(y)`, `sum(x*x)`, `sum(y*y)`, `sum(x*y)` for `x=vhat`, `y=V_j` | NO |
| Value sign accuracy on `abs(V_j)>EPS` | eligible-image count; sign-match count | NO |
| **Value Spearman** | stable global ranks of every finite `vhat` and every finite `V_j` before their Pearson correlation | **YES** |

The last row is a frozen P14 reported metric: `p14.corr(vh, v)['spearman']`.
The implementation first concatenates all held-class `vh` and `v`, then applies
two stable `argsort(argsort(..., kind='stable'), kind='stable')` operations.
It is neither a macro of per-fold Spearman values nor a function of moments.

## Why the Spearman row has no legal compact statistic

For a value in one held class, its global stable rank depends on comparisons
with every value in every other held class.  Per-fold counts, moments, local
ranks, local correlations, quantiles, and any bounded scalar schema lose those
cross-fold comparisons.  A deterministic stable-sort counterexample is:

| case | fold A `(vhat,V_j)` | fold B `(vhat,V_j)` | local rho A / B | global rho |
| --- | --- | --- | --- | --- |
| above | `([0,1],[0,1])` | `([2,3],[3,2])` | `+1 / -1` | `+0.8` |
| below | `([0,1],[0,1])` | `([2,3],[-2,-3])` | `+1 / -1` | `-0.8` |

Both fold-local rank relations are identical.  Only the cross-fold target
ordering changes; therefore the global stable rank correspondence, and hence
P14 global Spearman, changes.

An exact representation would need the complete ordered `vhat`/`V_j` value
multisets (or an equivalent cross-fold comparison matrix).  That is an
image-level prediction/`V_j` array in another encoding.  It is explicitly
forbidden to the P18 parent and cannot be reclassified as compact scalar
metadata.  No fixed histogram is exact for arbitrary floating-point values;
no bounded moment summary determines stable ranks.

## Decision

`P18_AGGREGATION_NO_GO`

All P14 gates and all other reported aggregate fields have legal exact compact
sufficient statistics.  The frozen global value Spearman diagnostic does not.
Changing it to macro Spearman, omitting it, allowing the parent to read the
arrays, or allowing an additional global array-owning worker would each change
the P18 contract supplied for this study.  Therefore a compact-only parent
cannot generate **every** frozen P14 final result exactly.  P18 must not be
preregistered, implemented, or executed.
