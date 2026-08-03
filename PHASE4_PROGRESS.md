# Phase 4 implementation progress

- `PHASE_X`: `1` (Progress 1 only)
- Base branch: `phase2b_kgsoftprompt_ctx4_fromscratch`
- Target branch: `phase4-progress1-cops-dynamic-prompt`
- Base commit: `869fdac6f8d93bb56a5e4cab0002fbff3e01573a` (verified after remote fetch)
- Implementation commit: `e011d09fdf2deaaf8cc15cccffca605b7b4e0161`
- Train/test executed: **NO**

## Verified Phase2B source configuration

The source scripts and `PHASE2B_RUN_CONTEXT.md` establish: three groups at
visual levels `8/16/24`; Conv-LoRA rank `16`, conv rank `8`, kernels `3/5`;
DFG attention `256`, tau `8`; SS2D `weight_residual`; DFG beta warm-up
`0 -> .05 -> .10`; image/text adapter weights `.2`; Adam with StepLR `.9`;
gradient checkpointing and gradient clipping `1.0`.

The Phase2B hybrid source uses `hybrid_alpha_max=.2`, three frozen soft-context
epochs, and the effective `.00/.05/.10/.20` alpha schedule. Progress 1 keeps
that schedule while replacing its static prediction branch with the dynamic
factor bank. H6-specific `lambda_kg=.001` and `lambda_k=0` follow the Phase4
scope lock.

## Checklist

- [x] Inspect source repository and preserve the dirty original workspace.
- [x] Create an independent code-only Phase4 clone; LFS smudge is disabled.
- [x] Create the Progress 1 target branch from the Phase2B source branch.
- [x] Define the Phase2B/H6 integration contract.
- [x] Implement semantic core, dynamic prompt bank, router, losses, residual fusion, and checkpoint schema.
- [x] Add Progress 1 train/test/chained scripts and result tools.
- [x] Add dataset-free unit tests.
- [x] Run code-level checks: compileall, synthetic shapes/finite check, 5 pytest tests, synthetic adapter contract, CLI help, shell syntax, and `git diff --check`.
- [x] Commit implementation files: `e011d09fdf2deaaf8cc15cccffca605b7b4e0161`.
- [ ] Push `origin/phase4-progress1-cops-dynamic-prompt`.

## Files created

- `model/h6/{__init__,semantic_bank,router,losses,model}.py`
- `model/checkpoint_utils.py`
- `tests/test_h6_common.py`, `tests/test_h6_progress1.py`
- `tools/{check_phase4_shapes,inspect_phase4_checkpoint,summarize_phase4_results}.py`
- `scripts/phase4/{train_progress1,test_6medical_exact,run_progress1_train_test}.sh`
- `PHASE4_H6_PLAN.md`, `EXPERIMENT_LOG_PHASE4.md`

## Files modified

- `model/adapter.py`, `train.py`, `test.py`
- `dataset/__init__.py` (dataset-local transform import; removes the prior `dataset`/`utils` import cycle)

## Commands for manual execution after this branch is pushed

```bash
SAVE_PATH=runs/phase4/progress1_cops_dynamic_prompt_seed0 \
CUDA_DEVICE=0 BATCH_SIZE=1 GRAD_ACCUM=6 TEST_BATCH_SIZE=1 NUM_WORKERS=6 \
PRECISION=bf16 SEED=0 EPOCHS=20 \
bash scripts/phase4/run_progress1_train_test.sh
```

- Final train command: `bash scripts/phase4/train_progress1.sh`
- Final exact test command: `bash scripts/phase4/test_6medical_exact.sh 8 9 10 11 12 13 14 15`
- Final chained command: `bash scripts/phase4/run_progress1_train_test.sh`
- Unresolved blockers: none before code-level checks.
- Final implementation commit SHA: `e011d09fdf2deaaf8cc15cccffca605b7b4e0161`.
- Push status: pending.
