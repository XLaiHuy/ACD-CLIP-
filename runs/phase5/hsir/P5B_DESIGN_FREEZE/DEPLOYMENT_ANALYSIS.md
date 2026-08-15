# Deployment analysis

## Exact operator

`tools/audit_phase5_reference_validity.py:244-254` applies, for each stage/group:

```text
native logits [37,37] -> Gaussian blur (7,7), sigma (1,1)
-> bilinear interpolation to 518x518 with align_corners=True
-> mean over stages/groups -> softmax over normal/anomaly logits.
```

The pre-softmax portion is linear. The installed Kornia implementation constructs Gaussian weights from `exp(-x^2/(2 sigma^2))` and normalizes by their sum; convolution uses nonnegative normalized weights. Bilinear interpolation is a nonnegative convex combination of neighboring samples, and stage mean is positive linear. Thus the pre-softmax map `L` is linear and positive for anomaly-channel perturbations: `L(a+b)=L(a)+L(b)` and `a>=0 => L(a)>=0`.

## Scratch-only synthetic validation

The exact repository deployment function was called with synthetic native tensors only; no model or dataset forward was performed.

```json
{
  "linearity_max_abs_error": 0.0,
  "model_forwards": 0,
  "native_impulses": [
    [
      18,
      18,
      1.0
    ],
    [
      18,
      18,
      -1.0
    ]
  ],
  "negative_impulse": {
    "l1": 206.22585659011878,
    "l2": 3.754008429151376,
    "max": 0.0,
    "min": -0.15490808136404854,
    "sum": -206.22585659011878
  },
  "negative_impulse_has_no_positive_lobes": true,
  "operator": "native 37x37 -> Gaussian blur 7x7 sigma1 -> bilinear resize align_corners=True -> stage mean; pre-softmax logits only",
  "positive_impulse": {
    "l1": 206.22585659011878,
    "l2": 3.754008429151376,
    "max": 0.15490808136404854,
    "min": 0.0,
    "sum": 206.22585659011878
  },
  "positive_impulse_has_no_negative_lobes": true,
  "shape": [
    1,
    2,
    518,
    518
  ],
  "signed_native_impulses": [
    [
      0,
      0,
      1.0
    ],
    [
      36,
      36,
      -1.0
    ]
  ],
  "signed_pair": {
    "l1": 108.95704689457958,
    "l2": 2.768571364877486,
    "max": 0.15924112569070248,
    "min": -0.15924112569070248,
    "sum": 2.842170943040401e-14
  },
  "signed_pair_has_both_signs": true,
  "softmax_not_included": true,
  "status": "PASS",
  "training_steps": 0
}
```

The corrected probe passes: linearity error is zero in float64; a positive native anomaly impulse has no negative deployed pre-softmax lobe; a negative impulse has no positive lobe; and separated signed impulses retain both signs. Softmax was intentionally excluded from this operator probe.

## Design consequence

Rank-locality is not spatial-locality. B3.1 rank-gap/spatial correlation is approximately zero and rescued/broken pairs are spatially broad. A future native correction needs a spatial-support/authority guardrail and full deployment validation. Positive-only projection avoids negative anomaly-channel lobes under this operator, but can still move positive score mass over broad support. Symmetric projection introduces both positive and negative native changes. Broad partial-order projection can move many patches and is not acceptable without a new guarantee.

The existing VisA comparison demonstrates the risk: aligned evidence is better natively but loses to shifted evidence after deployment, with reversal in 7/12 classes. No candidate is claimed deployable from the operator proof or B3 outcomes.
