# H2 SAFE-ANCHOR/CIR-V2 E20 preflight audit

Audit status: READY FOR COMMIT/PUSH REVIEW; full training is still blocked until the repository commit and remote equality checks pass.

## Provenance

- Branch: research/h2-clean-repro-anchor-cir-v1
- Starting SHA: fa61677e240999f707153505327e7305048bbfb4
- Base historical H2 commit: e03966997d4cecfd985943a4053a93e1e40197ec
- Exact CIR-V2 reference: 9cc0ad4cc6b34e34a8c15e74df881866516b3181
- Full-run code SHA must be recorded from git rev-parse HEAD after the final preflight commit/push and before launching training.

## Fixed factorial contract

- Arms: H native H2, A Anchor-only, C CIR-only, AC Anchor+CIR.
- Horizon: native shared E1, then E2-E20; E15 is primary and E20 is secondary.
- Target-valid checkpoints: only E15 and E20; no intermediate target selection or tuning.
- Shared fixed state: VisA, ViT-L-14-336, image size 518, batch 6, seed 0, AMP, deterministic algorithms, grad checkpointing, workers 6, and the frozen H2 optimizer/scheduler/model settings.
- Anchor active lambda: 0.0021633926715180626; family budget rho: 0.10; no lambda sweep.
- CIR settings: alpha 0.5, peer count 8, spatial radius 3, exact score-space transport/reference.

## Anchor evidence

- The backward path scales and backpropagates task loss only, unscales it, captures unscaled task gradients, obtains raw Anchor gradients separately, and replaces/combines only image-adapter gradients before clipping and stepping.
- Calibration uses only strict TASK_ACTIVE rows from the existing fixed source-only VisA geometry audit: historical E5/E10/E15 vision_text_k ratios 18.520829477442355, 33.84580840834666, and 23.11184680352771.
- R_MED is 23.11184680352771, target effective ratio is 0.05, and active lambda is target/R_MED = 0.0021633926715180626.
- Fresh activation gate: PASS / FAMILY_SAFE_ACTIVE. Active maximum family effective ratio is 0.09999937754084845; meaningful TASK_ACTIVE maximum is 0.03957792788580696; H remains native; A differs only through Anchor; finite and near-zero checks pass.
- Activation root: /tmp/h2_anchor_family_short_e20_20260902. Machine-readable evidence: audit/h2_anchor_family_short.json.

## Reproducibility and bounded smoke

- Complete current-repository tests: 46 passed.
- Anchor tests cover huge-lambda non-image isolation, family cap, AMP finiteness, weighted-anchor exclusion from the initial scaled backward, identity branches, and resume behavior.
- Exact medical evaluator focused test: 1 passed; historical medical tests remain separate and unchanged in protocol.
- Fresh bounded smoke: PASS at /tmp/h2_clean_smoke_e20_20260902.
- Smoke checks passed: one shared E1, equal E2 identities across H/A/C/AC, and repeated H E3 resume identity plus image-adapter-state equality.
- Smoke was five batches per epoch and did not run target evaluation, Medical, or MVTec.

## Evaluation and run guards

- Medical launcher is fixed to frozen H/A/C/AC E15/E20 checkpoints, raw exact metrics, pixel stride 1, and no checkpoint selection/tuning.
- The full launcher refuses a failed activation gate, wrong calibration/horizon, negligible Anchor signal, reused run root, incomplete shared E1, or wrong protocol version.
- Full root must be fresh and sequential on one GPU. Expected duration is approximately 13-16 or more GPU-hours.
- MVTec remains deferred until Medical completes and a local exact protocol is confirmed.

## Remaining publication gate

- Commit and push the intended current-repository changes and the separate medical evaluator changes using validated remote names.
- Verify local HEAD equals the pushed remote branch and both worktrees are clean.
- Record FULL_RUN_CODE_SHA, then launch the fresh H/A/C/AC E20 factorial. No scientific code edits are permitted after that SHA.
