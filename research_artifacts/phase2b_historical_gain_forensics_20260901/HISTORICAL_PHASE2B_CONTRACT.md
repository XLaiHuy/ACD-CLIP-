# Historical Phase2B H2 contract

Status: CONFIRMED from the H2 run log, source at commit e03966997d4cecfd985943a4053a93e1e40197ec, the exact H2 checkpoint payload, and historical test.py.

H2 is the 15-epoch VisA hybrid run with hybrid_alpha_max=0.2, lambda_kg=0.01, lambda_k=0.002, soft_prompt_lr=5e-5, AMP enabled, and the DFG attention/SS2D weight-residual configuration. Its optimizer is Adam with image/text base learning rates 1e-3/5e-4; StepLR uses gamma=0.9 and is actually stepped once per epoch after the epoch loop and before checkpoint save. The soft-prompt group is frozen at LR zero for the first three epochs and then follows a constant 5e-5 policy after the StepLR call.

The exact K-reg computation is documented in KREG_FORENSICS.md. The historical E10 checkpoint is ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb and contains model adapter state but no optimizer, scheduler, or RNG state. E10 was selected retrospectively using six-Medical pixel AP, so it is a useful historical champion but not a clean target-blind baseline.
