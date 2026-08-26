# P30R1 design note

1. **P30 blind spot.** Self-normalized cosine made direction learnable but
   left student radius non-identifiable. The candle failure was heavy-tail
   residual scale instability and downstream logit/anomaly-score saturation,
   visible in residual q99 before scoring; SmoothL1's tail derivative remains bounded.

2. **Normalization change.** P30R1 computes `t_bar=t/C`, `s_bar=s/C`, then
   `a_t=stop_gradient(sqrt(mean(t_bar^2)+eps^2))`, and compares
   `z_t=t_bar/a_t` with `z_s=s_bar/a_t`. The denominator is teacher-only and
   shared by both tensors.

3. **Why beta is identifiable.** For `t=alpha*u`, `s=beta*u`, the residual is
   proportional to `beta-alpha`; SmoothL1 therefore has its unique minimum at
   `beta=alpha`. `0.1x`, `10x`, and `100x` are not low-loss equivalents.

4. **Why zero gradients return.** With `t=0`, `a_t=eps`, `z_t=0`, and any
   nonzero student residual receives a finite gradient toward zero. Exact-zero
   targets are active rather than excluded.

5. **Why it remains one objective.** The only scientific loss is
   `F.smooth_l1_loss(z_s,z_t,beta=1,reduction="mean")`; there are no auxiliary
   cosine, sign, normal, ranking, feature, segmentation, calibration, or
   class-specific terms.

6. **Why inference is unchanged.** The unchanged RegionResidualAdapter,
   residual deployment, symmetric-margin integration, logits, and anomaly
   scoring remain in place. Normalization exists only during training, so
   target inference overhead is 0%.

7. **Principal risk.** The epsilon floor creates bounded but potentially
   stronger weighting for tiny teacher radii. The fixed mixed-scale test and
   source-cache diagnostic therefore gate the design; observed source q99/RMS
   median is 2.41 and only 2 nonzero samples lie below epsilon.

8. **Cheapest rejection.** Reject before any training if the deterministic
   suite loses scale identifiability, zero-teacher restoring force, finite
   near-zero gradients, mixed-scale bound, or heavy-tail detection. The suite,
   cache gate, and frozen-P30 counterfactual currently pass; this is
   `PASS_TO_IMPLEMENTATION`, not permission to run Stage 2.
