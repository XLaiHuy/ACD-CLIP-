# Anchor reference E8 R1 recovery audit

Status: `NO_VALID_CLEAN_E8_FULL_STATE_FOUND_REPRODUCTION_REQUIRED`

## Recovery result

The historical clean Phase2B factorial definition is present in
`configs/h2_clean_factorial_v1.json` and
`scripts/run_h2_clean_factorial.sh`. Its frozen clean-H/Baseline identity is
Anchor OFF, CIR OFF, seed 0, VisA, ViT-L-14-336, 518px, three groups, and the
Phase2B DFG/SS2D/hybrid-prompt settings recorded in
`configs/anchor_ref_e8_r1.json`.

The verified training implementation is commit
`31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`. The preserved H8 result remains
reachable at commit `ee9fe31c2e3c8b92e7b878f0568b0fd6511a5d12` on
`research/h6-sper-r1-full`; this branch does not modify that commit or the
external `/workspace/h8_med_msse_r1_run/` cache.

## Checkpoint findings

| candidate | result | evidence |
|---|---|---|
| shared E1 | valid full-state | `runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth`, SHA-256 `7f9176b7ef53b572935567c574535075a57317b2aa83505d043a71d45b12b35` |
| clean H E15 | valid full-state, wrong horizon | `runs/h2_clean_factorial_e20_20260902_ampfix/H/adapter_15.pth`, SHA-256 `6830137b52fc16321192909fe7b3ca7565664afe61a8592fe7b7a78d2dff4771` |
| clean H E8 | not found | no non-pointer E8 full-state checkpoint in the searched `/workspace`, `/tmp`, or `/root` inventories |
| archival E8-looking files | excluded | Git-LFS pointer stubs or adapter-only/forensic copies; not resumable scientific parents |
| A-E8/C-E8/AC-E8 | excluded | intervention-contaminated or unavailable; forbidden substitution |

The valid E1 and E15 payload inspections confirmed model/image-adapter,
text-adapter, soft-prompt, optimizer, scheduler, scaler, epoch, seed, and
Python/NumPy/CPU/CUDA/dataloader RNG state fields. The clean H E15 payload
records Anchor OFF and CIR OFF and contains no H7/H8, Stage-2 suppression,
high-resolution, or multiscale training configuration.

## Decision

Retraining E1-E8 is necessary. The controlled continuation is:

```text
shared clean full-state E1
        -> clean H/Baseline E2-E8
        -> immutable theta_ref = clean H image_adapter at E8
        -> same clean H E8 full state, Anchor ON, E9-E15
```

The E8-to-E9 branch boundary is explicit and the resume bridge used by the
launcher changes only in-memory identity metadata required by the existing
strict validator. It does not rewrite or alter any checkpoint tensor,
optimizer state, scheduler state, scaler state, or RNG state.
