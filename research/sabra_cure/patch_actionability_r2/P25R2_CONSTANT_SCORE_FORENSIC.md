# P25R2 Post-Stop Constant-Score Forensic Review

## Scope and terminal state

This is a read-only forensic review of terminal commit
`99ad3ab6292ca3b95fbda0cb8c6985ed9afe3253`. It performed no scientific fit,
attempt, Q2 execution, result selection, CLIP forward, or Phase2B step, and did
not access MVTec or Medical. P25R2 remains `NON_INTERPRETABLE`.

## Finding

**FORENSIC ROOT CAUSE: `COMPUTATION_BUG`.**

The held `chewinggum` score was already an exact all-zero vector immediately
after inference. Persistence, loading, dtype conversion, alignment, masking,
schema handling, and aggregation did not change it.

The fitted ranker artifact contains a 32-dimensional beta vector with every
coefficient exactly zero. Because the ranker has no intercept, its inference
formula necessarily returns zero for every finite feature row. More
importantly, a read-only reconstruction of the 11-class source training design
found that the gradient of the exact frozen objective at beta=0 has L2 norm
`1750236.1845595562` and maximum absolute component `1747535.9091769257`;
all 32 components are nonzero. The frozen optimizer tolerance was `1e-12`.
Thus beta=0 was not a converged stationary solution. The zero-init L-BFGS path
returned its initialization, and the implementation accepted it because it
checked only finiteness, not convergence or first-order optimality.

The reconstructed pair design is finite but extremely ill-scaled: its maximum
absolute value is `9435578848.964195`. This is strong evidence that the
strong-Wolfe line search failed to make a usable update under the frozen
parameterization. The optimizer's internal status and iteration count were not
persisted, so the exact internal line-search failure code cannot be recovered.

## A/B answers

### A. Was the prediction constant before persistence?

Yes, conclusively.

- `q1_fold` computes `score`, then computes Pearson, Spearman, and
  `score_variance`, and only afterwards writes the NPZ. The parameter artifact
  records `score_variance=0.0`, `pearson=null`, and `spearman=null` from that
  in-memory vector.
- The serialized model contains beta=`[0,...,0]` and no intercept. The frozen
  inference expression is standardized-X matrix-multiplied by beta, hence its
  raw output is exactly zero for every finite held row.
- A deterministic held-feature reconstruction that did not read held targets
  reproduced an all-zero float64 vector exactly: raw-versus-persisted maximum
  absolute error `0.0`, `array_equal=true`.

### B. Did persistence or an interface convert a nonconstant vector to zero?

No.

- Raw reconstructed score: shape `(2000,)`, dtype `float64`, min/max/std
  `0.0/0.0/0.0`, one unique value.
- Persisted score: shape `(2000,)`, dtype `float64`, min/max/std
  `0.0/0.0/0.0`, one unique value.
- `atomic_npz` passes the existing array directly to `np.savez_compressed`; it
  contains no cast, fill, mask, or fallback.
- NPZ loading uses `np.asarray(data[key])` without a dtype conversion.
- Aggregation consumes the already-computed per-fold metric dictionary and
  never writes the score vector.

## Fold and model inventory

| Item | Evidence |
|---|---:|
| Outer training classes | 11 |
| Training samples | 22,000 |
| Deterministic valid pairs | 89,942 |
| Held samples | 2,000 |
| Training feature shape | `(22000, 32)` |
| Held feature shape | `(2000, 32)` |
| Training/held feature finite rate | 100% for every feature |
| Fitted beta L2 norm | `0.0` |
| Nonzero coefficients | 0/32 |
| Intercept | absent by model definition |
| Recorded optimizer loss | `0.6931471805599452` |
| Optimizer configuration | CPU float64 L-BFGS, zero init, max 100 iterations, strong-Wolfe, grad tolerance `1e-12`, change tolerance `1e-14` |
| Optimizer convergence/status | not persisted |
| Objective gradient L2 at beta=0 | `1750236.1845595562` |
| Objective gradient max abs at beta=0 | `1747535.9091769257` |
| Pair-design max abs | `9435578848.964195` |

Pair counts by source class were: candle 8172, capsules 8178, cashew 8172,
fryum 8178, macaroni1 8172, macaroni2 8178, pcb1 8172, pcb2 8172,
pcb3 8172, pcb4 8184, and pipe_fryum 8192. Their sum is exactly 89,942.

## Held feature profile

`finite` is the reconstructed held finite rate. `IQR0` is the exact held raw
IQR-zero result. `Train scale` is the persisted training scaler after the
frozen `max(raw_IQR, 1e-6)` operation; `floor` means the raw training IQR was
at most `1e-6`, not necessarily exactly zero.

| # | Feature | finite | held variance | IQR0 | Train scale | floor |
|---:|---|---:|---:|:---:|---:|:---:|
| 0 | margin_within_image_rank | 1.0 | 7.845105e-2 | no | 4.985380e-1 | no |
| 1 | robust_margin_normalization | 1.0 | 5.969375e1 | no | 2.913693 | no |
| 2 | D_rank | 1.0 | 9.380569e-3 | no | 1.491932e-1 | no |
| 3 | deployment_sensitivity | 1.0 | 9.027949e-14 | no | 1e-6 | yes |
| 4 | E | 1.0 | 4.573235e-2 | no | 3.684820e-1 | no |
| 5 | peer_coherence | 1.0 | 2.345350e-6 | no | 2.534837e-3 | no |
| 6 | query_support_mean | 1.0 | 3.131349e-3 | no | 1.842521e-2 | no |
| 7 | peer_eigen_entropy | 1.0 | 5.887772e-3 | no | 1.088269e-1 | no |
| 8 | stage_query_profile_disagreement | 1.0 | 4.425964e-3 | no | 1.125843e-5 | no |
| 9 | where(valid_p9,S9,0) | 1.0 | 1.251674e-2 | no | 1.656920e-1 | no |
| 10 | where(valid_p16,S16,0) | 1.0 | 1.245143e-2 | no | 1.676413e-1 | no |
| 11 | signed_native_margin | 1.0 | 1.457315 | no | 7.859211e-1 | no |
| 12 | cross_stage_signed_margin_difference | 1.0 | 2.712761 | no | 1.329485 | no |
| 13 | robust_peer_signed_margin_consensus | 1.0 | 1.384975e-1 | no | 1.755455e-1 | no |
| 14 | mu | 1.0 | 9.200857e-3 | no | 1.745020e-1 | no |
| 15 | abs_mu | 1.0 | 9.192085e-3 | no | 1.714801e-1 | no |
| 16 | sigma | 1.0 | 1.432217e-3 | no | 4.721845e-2 | no |
| 17 | standardized_direction_strength | 1.0 | 8.168078e-2 | no | 4.762313e-1 | no |
| 18 | proposed_native_margin_support | 1.0 | 2.179851 | no | 7.879192e-1 | no |
| 19 | proposed_peer_margin_support | 1.0 | 7.141274e-1 | no | 1.821696e-1 | no |
| 20 | proposed_stage_difference | 1.0 | 2.712921 | no | 9.970980e-1 | no |
| 21 | absolute_stage_difference | 1.0 | 1.974375 | no | 1.617051 | no |
| 22 | harm_risk | 1.0 | 1.171759e-3 | no | 2.732762e-2 | no |
| 23 | harm_policy_action | 1.0 | 1.787110e-1 | yes | 1e-6 | yes |
| 24 | support_native_rank_median | 1.0 | 5.059975e-2 | no | 3.483139e-1 | no |
| 25 | support_native_rank_q90 | 1.0 | 3.111109e-2 | no | 3.063319e-1 | no |
| 26 | signed_delta_mean_over_image_iqr | 1.0 | 5.468978e5 | yes | 1e-6 | yes |
| 27 | abs_delta_q90_over_image_iqr | 1.0 | 2.130552e5 | yes | 1e-6 | yes |
| 28 | support_rank_shift_median | 1.0 | 3.572550e-4 | yes | 1e-6 | yes |
| 29 | support_rank_shift_abs_q90 | 1.0 | 3.027917e-2 | yes | 1e-6 | yes |
| 30 | top5_boundary_cross_fraction | 1.0 | 9.168761e-6 | yes | 1e-6 | yes |
| 31 | top20_boundary_cross_fraction | 1.0 | 1.230754e-4 | yes | 1e-6 | yes |

The reconstructed training medians match the persisted scaler within
`1.1324274851176597e-14`; reconstructed clipped IQRs match within
`1.6542323066914832e-14`. This confirms that the source-only reconstruction is
the fitted design, rather than an approximate substitute.

## Alignment, masks, and fallback branches

- Source-cache paths equal the persisted R2-v2 held paths.
- Every panel path equals the source path selected by its image index.
- Fold image/patch indices equal target indices; target indices equal panel
  indices; all 2,000 composite `(image_index, patch_index)` keys are unique.
- Fold target values are byte/value-identical to target-artifact values.
- Full chewinggum validity: `valid_p9=205350/205350`,
  `valid_p16=205350/205350`, `valid_b1=205350/205350`.
- Panel validity: each validity mask is `2000/2000`; therefore the p9/p16
  zero-fill and peer-consensus fallback branches were not reached.
- Harm policy actions are non-KEEP for 466 rows and KEEP for 1,534 rows. The
  expected impact-proxy KEEP branch initializes the six delta/crossing fields
  to zero while still filling the two support-rank fields. This branch occurs
  before ranker inference and does not explain the all-zero score: beta is zero
  for all 32 features, including the 24 non-impact features.
- No post-inference mask, alignment filter, or zero-fill branch exists.

## Assignment trace for `score`

1. The ranker parameter is explicitly initialized to a float64 zero vector.
2. CPU L-BFGS is called and may update that vector. No optimizer status or
   first-order optimality check is retained.
3. The resulting parameter is copied to `beta`; the only validation is that it
   is finite. For chewinggum, the copied beta remains exactly all zero.
4. `rank_predict` computes `((X-median)/IQR) @ beta`; there is no intercept,
   mask, fallback, clipping, or cast. This produces the all-zero raw vector.
5. `q1_metrics`, Pearson, and `np.var(score)` read the raw vector without
   modifying it. The in-memory variance is recorded as zero.
6. `atomic_npz` writes the same score object with `np.savez_compressed` and
   atomically renames the temporary file.
7. NPZ loading returns the float64 score unchanged. The reconstructed raw and
   loaded arrays are exactly equal.
8. Q1 aggregation reads per-fold metric dictionaries. It does not assign to or
   reload the score vector; it stops on the undefined Spearman.

Code references:

- Rank scaler, pair construction, zero init, L-BFGS, beta extraction, and
  inference: `tools/sabra_cure/patch_actionability_r1.py:377-444`.
- GT-free feature validation and expected KEEP proxy branch:
  `tools/sabra_cure/patch_actionability_r2.py:305-350`.
- Inference-before-metrics-before-persistence ordering:
  `tools/sabra_cure/patch_actionability_r2.py:353-370`.
- NPZ writer: `tools/sabra_cure/patch_actionability_r2.py:67-72`.
- p9/p16/peer validity fallbacks: `tools/sabra_cure/r1.py:183-203`.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| P25R2 runner | `3868e712250f9ebfbff27c962c67e76d6845e82cc7206abcec9f57053449292f` |
| Frozen ranker source | `754e22c0a3c206f5e7408c0a41481def3cb6547e6c80d22a193098b83cfe02e4` |
| chewinggum parameters | `2f5f27c19d096811f937d35634bbdd48634ebb85cde10902824e614df69e9922` |
| chewinggum Q1 fold | `0748522d6761f43a1baae82f426d0b627c886f90a3dc1d71caf2fbf81476500f` |
| chewinggum target | `80bf2ff802297fa5be25dd2175a200c712ffdefa94856c3eb801083bf7c8897d` |
| chewinggum panel | `2f6b63813f69cd6e988fb4569b194b84151a08576dd19dac8bf4ec837051e9aa` |
| Persisted score bytes | `f85f2c34eb2843d2aa5951ee6e8e76985655b2e3ae2cbdd76bdfd654ecf19997` |

## Recovery assessment

**RECOVERY JUSTIFIED: YES, technically.**

The minimal exact correction is confined to ranker numerical optimization:

1. preserve the exact 32 features, source-only 11-class training set,
   deterministic 89,942 pairs and weights, pairwise logistic objective,
   L2=`1.0`, float64 arithmetic, and all Q1 metric definitions;
2. optimize that identical objective under an internal, invertible numerical
   preconditioner, including the original beta-space L2 term exactly, then map
   coefficients back to the original parameterization;
3. require a persisted optimizer status plus a post-fit first-order-optimality
   check; reject a finite beta whose gradient exceeds the frozen tolerance;
4. retain undefined Pearson/Spearman as null if a valid stationary solution
   genuinely produces a constant score. Never substitute zero.

This changes numerical optimization implementation, not the scientific target,
model family, objective, predictions' definition, or metric semantics. It must
be preregistered and independently parity-tested before any new attempt. This
review does not authorize or implement that recovery.

## Final status

- **SCIENTIFIC STATUS:** P25R2 remains `NON_INTERPRETABLE`.
- **NEXT_ALLOWED_ACTION:** explicit user review.
