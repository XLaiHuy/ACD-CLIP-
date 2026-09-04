# H2 loss and gradient forensics

## Coverage

The source logs expose epoch means for the main, classification,
segmentation, KG, and K terms, plus family-safe Anchor telemetry for A. They
do not expose complete per-term gradients, `cos(g_seg,g_cls)`, optimizer
moments, or a full gradient vector dump. Claims below are therefore bounded by
the logged evidence.

## Loss scale

At E2, the main loss is approximately `1.138` to `1.190`, with classification
approximately `.626` to `.676` and segmentation approximately `.508` to
`.515`. At E15, the main loss is approximately `.759` to `.943`, with
classification `.411` to `.549` and segmentation `.349` to `.407`.

The raw KG term falls from about `.596` to `.625` at E2 to about `.220` to
`.329` at E15. With lambda `.01`, its weighted magnitude is about `.006` early
and `.002` to `.003` late. The weighted K term is near zero while the soft
prompt is frozen and is only about `5e-6` to `6e-5` at E15. The logged scalar
losses therefore do not support a claim that a regularizer dominates the
objective by magnitude.

## Anchor direction and family imbalance

The family audit shows a small global effective Anchor/task ratio near `1e-5`
at E15 in both A seeds. The family cap is reached by some Q/K and SS2D
elements, while LoRA and `m_i_W` remain mostly negligible. The final Q/K raw
gradient ratios are large relative to their very small task norms, but their
effective ratios remain below the family budget. This is evidence of
localized family imbalance, not aggregate Anchor takeover.

The logged family cosine values are mixed and near zero or negative in several
late Q/K/SS2D cases. That is compatible with local conflict, but it is not a
complete `g_task` versus `g_anchor` experiment and cannot establish that the
Anchor caused target degradation.

## Missing conflict measurements

The logs do not contain the separate segmentation and classification gradient
vectors, so `cos(g_seg,g_cls)` is not computable from current artifacts. They
also do not contain complete KG/K gradient vectors, Adam moments, or update
vectors. No target score-map gradient or representation diagnostic exists.

Accordingly:

`LOSS_BOTTLENECK=NONE_DEMONSTRATED`.

This means no loss formulation or scale failure has been proven. It does not
mean loss conflict is impossible. The repeated nonfinite gradients should be
treated as a numerical-stability problem first, with any loss change deferred
until a source-only trace identifies a specific offending branch.
