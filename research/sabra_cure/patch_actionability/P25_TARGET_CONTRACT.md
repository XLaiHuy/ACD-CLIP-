# P25 Target and Panel Contract

For source class C and panel patch j, let `native_C` be frozen native deployed
output and let `proposal_sign_j` be the source-GT direction-target sign. The
candidate changes only patch j by `proposal_sign_j * alpha`, with `alpha=.25`
and the frozen deployment transform. The source-only target is:

`V_j = exact_pixel_AP(candidate_j) - exact_pixel_AP(native_C)`.

`V_j` is source supervision, not a causal effect, global oracle, predicted pAP,
or deployable feature. Candidate AP uses exact frozen float32 score ordering.

## Frozen deterministic panel

`TARGET_PATCHES_PER_CLASS=4096`, `MAX_TARGET_PATCHES_PER_IMAGE=16`, 25 strata
from GT-free native-score-rank quintile x GT-free deployment-sensitivity
quintile. Quintile cutpoints use `numpy.quantile(method="linear")`; bin index
is `min(4, searchsorted(cutpoints, value, side="right"))`.

Strata use ascending identifier `5*rank_quintile+sensitivity_quintile`.
Initial quotas are 164 for identifiers 0--20 and 163 for 21--24 (total 4096).
Within each stratum candidates are ordered by SHA256 of
`class_name + "\\0" + image_path + "\\0" + patch_index`, then image path and
patch index. Select in that order while enforcing the 16-per-image cap.
If an initial quota cannot be filled, its remaining units are redistributed one
at-a-time by repeated ascending stratum scan over remaining eligible ordered
candidates, still enforcing the cap, until 4096 are selected. Panel membership
is serialized and hashed before any V is computed; V is never consulted for
sampling or resampling.

The exact engine uses one native grouped-count state, a sparse per-patch score
effect, group-count apply/AP/revert, and compact one-shard-per-class output.
No full deployment per candidate, scalar AP-delta composition, score rounding,
or approximation is allowed.
