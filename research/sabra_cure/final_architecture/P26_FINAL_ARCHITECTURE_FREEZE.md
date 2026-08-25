# P26 Final Architecture Freeze

`SABRA_FINAL_ARCHITECTURE_FROZEN = TRUE`

Name: `SABRA-FINAL-NATIVE-PHASE2B-V1`

The deployable architecture is the canonical frozen Phase2B detector with the
native output path. No SABRA/CURE sidecar is active. The only reachable action
is `KEEP`, selected for every patch and image. This conservative result follows
the preregistered P26 adjudication rule: no corrective component completed a
leakage-safe source study establishing both GT-free selection and sufficient
ranking benefit.

Runtime identities and every scientific parameter are frozen in
`SABRA_FINAL_CONFIG.json`. The model uses OpenAI CLIP ViT-L/14@336px, the
verified epoch-5 Phase2B adapter, three visual/text stages, 518x518 bicubic RGB
input normalized with CLIP statistics, and fp32 inference. Native per-stage
logits are reshaped to 37x37, blurred with kernel 7 and sigma 1, bilinearly
resized to 518x518 with `align_corners=True`, averaged across stages as logits,
then softmaxed; channel 1 is the anomaly map.

There is no TTA, target calibration, test-time training, prompt tuning,
threshold selection, or runtime architecture selection. Seed is 0. H6 and the
legacy branch are disabled. The historical signed actuator alpha is fixed at
0.25 but unreachable because intervention is disabled.

This commit is the scientific architecture boundary. An external result cannot
change it. External validation is not authorized by P26.
