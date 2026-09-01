# Anchor reference decision

Status: FROZEN BEFORE NEW MEDICAL EVALUATION.

Rule: use the preregistered exact-H2 E1 model-state checkpoint as the fixed, target-blind reference for the image-adapter-only anchor. Do not use current C2 P E14 or any Medical-selected checkpoint.

- path: `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_1.pth`
- epoch: `1`
- SHA256: `da893014cdd9ca643f632cedbf5d43fc57eb3acee343e4795bc7de9aa12c3074`
- lambda: `0.001`
- scope: `image_adapter_parameters_only`
- parameter names: `99` exact names match common E0
- shapes: exact match to common E0 and H2 model
- inference: anchor disabled; native H2 alpha=0

The reference is a training-only frozen tensor set and is not registered as an optimizer parameter.
