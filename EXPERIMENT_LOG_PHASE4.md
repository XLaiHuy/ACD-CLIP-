# Phase 4 experiment log

This is a results template. No train or test has been executed by this branch.

| Progress | Run path | Epoch range | Training status | Medical validation sweep | One-time medical test | Notes |
|---|---|---|---|---|---|---|
| P1 | `runs/phase4/progress1_cops_dynamic_prompt_seed0` | 1-20 | not run | not run | not run | VisA only; Colon is deterministic 30/70 val/test |
| P1-v1 | `runs/phase4/progress1_cops_dynamic_prompt_seed0_retry1` | 1-20 | completed outside Codex | completed outside Codex | completed outside Codex | Existing result files audited; used only as evidence, not rerun |
| P1-v2 | `runs/phase4/progress1_v2_specialization_seed0` | 1-20 | not run | not run | not run | Critical fixes: frozen anchor path, pre-fusion norm, decoder(mu) prompt, residual diversity |

## Medical validation selection record

Populate from `medical_val_results_by_epoch.csv`.  Select one epoch from this
table only, then record the single test result separately from
`medical_test_results_by_epoch.csv`.

| Epoch | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Combined score | Selected? |
|---:|---:|---:|---:|---:|---:|---|

## Diagnostics to review

- Factor usage, normalized routing entropy, top-1 share, dead factors.
- Normal/abnormal prototype distance, factor direction cosine, dynamic-hard cosine.
- VAE reconstruction/KL and mu/logvar range.
- Gate values: gamma state, gamma class, and per-level rho.
