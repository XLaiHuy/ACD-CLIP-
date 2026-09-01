# Machine handoff manifest

## Frozen identity

- `H2_EXACT_PATH=/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth`
- `H2_ORACLE_SHA=ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb`
- `H2_EXACT_HORIZON=E15`
- `H2_E20_SEARCH=FOUND_DIFFERENT_RUN`
- `H2_BASE_COMMIT=e03966997d4cecfd985943a4053a93e1e40197ec`
- `CANDIDATE_BRANCH=research/h2-clean-repro-anchor-cir-v1`
- `CANDIDATE_WORKTREE=/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901`

## Locked implementation

- Config: `configs/h2_clean_factorial_v1.json`
- Full-train launcher: `RUN_FULL_TRAIN=YES bash scripts/run_h2_clean_factorial.sh`
- Full-eval launcher: `RUN_EVAL=YES FINAL_FROZEN=YES ARM=<H|A|C|AC> bash scripts/eval_h2_clean_factorial.sh`
- Full train is native shared E1, then H/A/C/AC branches from the shared
  full-state E1 checkpoint at E2-E15.
- Medical evaluation is fixed E15 only; no selection logic is enabled.
- Geometry and performance-winner decisions remain unauthorized.

## Completed gates

- `15 passed` unit/integration tests.
- Python compilation and shell syntax checks pass.
- Guarded launcher defaults exit prepared-only without launching work.
- Shared-E1 plus H/A/C/AC five-batch smoke passes. Smoke evidence is in
  `H2_CLEAN_SMOKE_RESULTS.md`; scratch outputs were written under
  `/tmp/h2_clean_smoke_v1`.
- Historical legacy and current full-resolution oracle values are in
  `H2_ORACLE_EVALUATOR_PARITY.csv`.

## Provenance caveats

- The historical H2 checkpoint is model-only and replay-only; it cannot be
  used for exact optimizer/RNG continuation.
- The historical source gate overlaps VisA training categories and is disabled
  in the clean config.
- The actual runtime was torch 2.12.1+cu130/CUDA 13.0; cross-environment
  bitwise identity is not claimed.
- The implementation commit and remote head are reported by the final Git
  handoff after commit and push verification.
