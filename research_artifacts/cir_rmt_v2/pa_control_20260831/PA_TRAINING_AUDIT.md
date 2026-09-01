# PA training audit

Status: PASS. PA completed a fresh native Phase2B E1-E20 run with the fixed image-parameter anchor.

- control: PA_PHASE2B_IMAGE_ANCHOR_V1
- source: VisA, seed 0
- training forward: native_phase2b
- CIR/RMT training: disabled
- anchor: P_E14 image_adapter only, lambda=0.001, train-only
- precision: FP32, AMP=false, TF32=false
- optimizer: Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0
- scheduler: StepLR(step_size=1,gamma=0.9), after epoch and before checkpoint
- clipping: norm 1 once per optimizer update
- candidate checkpoints: E10/E12/E14/E16/E18/E20
- gradient probe: NOT_RECORDED; `PA_GRADIENT_TRAJECTORY.csv` preserves empty probe fields and no gradient conclusion is drawn

Manifest SHA256: 3df4156f0992585628f5c3850a03d9839d699b93abe58d01b5e44c2854a4dadc
Verifier status: PASS

Candidate checkpoint SHA256:
- E10: 9f12f21327061eea36af03d24f73dfcc06b11158019b37e7691a103c529ba0c2
- E12: 2bf264f6ab7b5596fa5eeb425a517da28ece3be477c50e395e4bb4b82f9900b8
- E14: 1a69943c1ec3fb7513000f2f4f274ffb4071ee4ff4c81394a0b597a3241be4b6
- E16: 1e275f2f6a4b7d348af7a81398ca497b5b8c394d8e7d1d92978425e58634ff62
- E18: b167d424140f208e3b055f6e29bca37f39e94517c4c5a65a2d5711a92de5e18d
- E20: 22c44a28f6dd88c59525fa9699249acb2744f4c23c7d69f8bd882311d52837ee

Medical and MVTec were not accessed by the training stage. Target tuning: NO.
