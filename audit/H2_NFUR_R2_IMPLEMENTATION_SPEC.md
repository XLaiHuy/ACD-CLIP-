# H2 NFUR-R2 implementation specification

Status: frozen before source training.

## Scientific scope

NFUR-R2 is one residual uncertainty refinement path attached after the
existing equal-weighted three-stage H2 score fusion. It uses only VisA source
supervision during training and uses no target data, category identifier,
ground-truth mask, or target-derived statistic at inference.

The retained H2 representation inventory is:

| source | shape | resolution | trainability | role |
| --- | --- | --- | --- | --- |
| frozen patch embedding / early visual tokens | `[B,1369,1024]` | 37x37 | frozen backbone | existing transformer stream |
| stage 1/2/3 Conv-LoRA outputs | `[B,1369,1024]` | 37x37 | adapter trainable | existing stage outputs |
| stage 1/2/3 projected segmentation tokens | `[B,1369,768]` | 37x37 | adapter trainable | NFUR native visual input |
| stage margins and fused margin | `[B,37,37]` each | 37x37 | derived, differentiable | NFUR uncertainty inputs |
| final score map | `[B,518,518]` | image-aligned | derived | existing output plus residual |

There is no existing genuinely finer spatial feature map than the final
coarse 37x37 token grid. NFUR therefore refines native uncertainty at 37x37
and interpolates the correction; it does not claim sub-patch detail.

## Exact formulation

Let `z_i` be the existing native two-class logits for stage `i`, `i=1..3`,
and `v_i` the corresponding normalized 768-dimensional token map. Define

```text
m_i(p) = z_i(p, abnormal) - z_i(p, normal)
m_f(p) = mean_i m_i(p)
u_margin(p) = 1 - tanh(abs(m_f(p)) / 10)
u_disagreement(p) = 1 - exp(-std_i(m_i(p)))
g_uncert(p) = clamp((u_margin(p) + u_disagreement(p))/2, 0, 1)
```

The divisor 10 is the existing H2 visual/text score scale. The standard
deviation uses population (`unbiased=False`) standard deviation over the
three stages. The refiner input is
`concat(mean_i(v_i), m_1, m_2, m_3, m_f)` with shape `[B,772,37,37]`.

```text
delta_s(p) = 0.25 * tanh(Conv1x1(GELU(Conv3x3(x(p)))))
c(p) = Upsample(g_uncert(p) * delta_s(p))
s_refined(normal)   = s_base(normal)   - 0.5*c
s_refined(abnormal) = s_base(abnormal) + 0.5*c
```

The head is `Conv2d(772,32,3,padding=1)`, GELU, `Conv2d(32,1,1)` and has
222,401 trainable parameters. The last convolution weight and bias are zero
initialized, so `delta_s=0` exactly at initialization and
`s_refined=s_base` exactly when the residual is forced OFF. The correction is
bounded by 0.25 before uncertainty gating.

## Gradient and inference contracts

The head receives gradients through the existing final segmentation loss on
VisA. `g_uncert` is continuous and not detached; it is a deterministic
score-derived gate, not a label/GT gate. The image/text adapters and existing
H2 training objective remain unchanged. At inference, the same frozen formula
uses only current visual tokens and current stage logits. Test-time Gaussian
blur and interpolation are applied to the residual with the same domain
kernel convention as the existing H2 path.

The `use_nfur=False` / `set_nfur_enabled(False)` path returns the historical
fusion tensor without entering the new branch. This is the identity-off test
contract. The NFUR model starts from the retained Safe Anchor E10 full-state
model; retained E1-E10 checkpoints are preserved as the pre-NFUR warm-start
trajectory, while E11-E20 are produced by the single NFUR-R2 continuation.

## Files and limitations

Changed files: `model/adapter.py`, `h2_clean/contract.py`, and `train.py`.
The checkpoint contract now retains `nfur_refiner` state. No second NFUR
variant, decoder, transformer, SS2D branch, or target-side tuning is added.

Limitations: the method can only refine errors represented by the native
37x37 H2 grid; it cannot recover detail absent from that grid. Medical epoch
selection is target validation, so it does not support an untouched zero-shot
claim.
