# H2 checkpoint identity

Status: CONFIRMED.

The exact historical H2 E10 checkpoint is `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth` with SHA256 `ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb` and source commit `e03966997d4cecfd985943a4053a93e1e40197ec`. It is a legacy model-state payload: optimizer, scheduler, and RNG state are absent. Historical E10 selection was retrospective Medical-informed, so it is replay/oracle evidence and not a new target-blind selection rule.

The fixed, target-blind anchor reference is H2 E1 `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_1.pth` with SHA256 `da893014cdd9ca643f632cedbf5d43fc57eb3acee343e4795bc7de9aa12c3074`. Its image-adapter parameter identity and shapes were checked against the exact H2 model before training. It is used only by RA/RCA during training and is absent at inference.

The current C2 E10 checkpoint is intentionally not reused as an anchor: its protocol and parameter trajectory are not H2-compatible.
