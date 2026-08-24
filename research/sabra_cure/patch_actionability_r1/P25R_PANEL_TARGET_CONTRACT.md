# P25R Panel and Target Contract

Each class has exactly 2000 target patches, selected before V computation from
25 GT-free strata: `margin_within_image_rank` quintile x
`deployment_sensitivity` quintile. Quantiles use `numpy.quantile` with
`method="linear"`; values use `searchsorted(..., side="right")`, capped at 4.
Each stratum has quota 80. Candidates sort by SHA256 of
`class_name + "\\0" + image_path + "\\0" + patch_index`, then image path and
patch index. Selection enforces exactly 16 patches/image. If a stratum lacks
quota, unused slots redistribute one-at-a-time over ascending strata in repeated
scans of remaining hash-ordered eligible candidates. No mask, utility, label,
V, or AP enters panel selection.

For source patch j, `proposal_sign_j` is the source GT-bearing direction-target
sign. With frozen alpha=.25 and frozen deployment, candidate j changes only
that patch. `V_j = exact_pixel_AP(candidate_j)-exact_pixel_AP(native_C)`.
V is source-only supervision, not a causal estimate, pAP prediction, additive
contribution, deployment feature, or performance bound.
