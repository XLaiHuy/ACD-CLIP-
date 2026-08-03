# Phase 4 experiment log

This is a results template. No train or test has been executed by this branch.

| Progress | Run path | Epoch range | Training status | Exact medical test status | Notes |
|---|---|---|---|---|---|
| P1 | `runs/phase4/progress1_cops_dynamic_prompt_seed0` | 1-20 | not run | not run | Fill after manual execution |

## Per-epoch selection record

Populate from `results_by_epoch.csv` after the exact six-dataset evaluation.

| Epoch | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Combined score | Selected? |
|---:|---:|---:|---:|---:|---:|---|

## Diagnostics to review

- Factor usage, normalized routing entropy, top-1 share, dead factors.
- Normal/abnormal prototype distance, factor direction cosine, dynamic-hard cosine.
- VAE reconstruction/KL and mu/logvar range.
- Gate values: gamma state, gamma class, and per-level rho.
