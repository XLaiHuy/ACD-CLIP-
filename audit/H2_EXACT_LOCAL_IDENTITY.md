# Exact H2 local identity

Status: PASS.

The hash search under `/home/ai4/caohuy` identifies the authoritative H2 E10
checkpoint at:

`/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth`

Its SHA256 is
`ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb`.
The run contains exactly `adapter_1.pth` through `adapter_15.pth`; E16-E20
are absent. The checkpoint is model-only: it has adapter and prompt weights,
but no Adam moments, scheduler, GradScaler, or RNG/DataLoader state. It is a
replay parent, not an exact continuation parent.

The logged contract is ViT-L/14-336, 518px, three groups, attention DFG
(`dim=256`, `tau=8`), SS2D weight-residual fusion, beta warmup-to-0.1,
hybrid prompt alpha 0.2, `lambda_kg=.01`, `lambda_k=.002`, Adam,
`StepLR(gamma=.9)`, AMP, batch 6, and 15 epochs. The historical Medical
launcher explicitly used `METRIC_THRESHOLDS=none` and `PIXEL_STRIDE=4`.

The exact H2 source commit is `e03966997d4cecfd985943a4053a93e1e40197ec`.
The current artifact directory is in the later phase2c worktree, whose
current HEAD is recorded separately above; the new clean branch is based
directly on e039. The post-hoc training archive launcher is recorded as
provenance but is explicitly not claimed to be part of e039.

CLIP and VisA manifest hashes are recorded in the JSON companion. Historical
E10 replay results are stored separately in the evaluator-parity audit.
