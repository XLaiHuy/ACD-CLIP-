# H2 current architecture and cross-machine Git handoff

Snapshot: 2026-09-04. This document describes the eligible seed-0 H2
four-arm factorial and the checkpoints archived in Git LFS.

## Current architecture

- Backbone: OpenAI CLIP `ViT-L-14-336`.
- Input: image size `518`; three adapter groups; batch size `6`.
- Representation: Conv-LoRA image adaptation.
- Fusion: attention DFG with `dfg_attn_dim=256`, `dfg_attn_tau=8`, and SS2D
  residual-weight fusion in FP32.
- Text side: hard text adapter plus four-token hybrid soft prompt.
- Training: 20 epochs; E15 is the primary horizon and E20 is secondary;
  AMP/BF16-safe path, gradient checkpointing, deterministic algorithms, seed
  `0`.
- Loss/config: `lambda_kg=0.01`, `lambda_k=0.002`, beta warm-up to `0.1`.

## Four frozen arms

| Arm | Anchor | CIR | Additional setting |
|---|---|---|---|
| H | off | off | H2 reference |
| A | on | off | `anchor_lambda=0.0021633926715180626`, family budget `rho=0.1` |
| C | off | on | `cir_alpha=0.5`, 8 peers, spatial radius 3 |
| AC | on | on | A and C settings combined |

All selected states passed the final finite-state and identity checks. The
shared E1 checkpoint is the common reference for CLIP, dataset manifest, base
H2 commit, and CIR reference identity.

## Archived checkpoints

E15 is primary; E20 is secondary. The E20 files include optimizer, scheduler,
scaler, RNG, and dataloader-generator state, so they are the correct resume
points for continuation from the completed run.

| Role | Git path | SHA256 |
|---|---|---|
| Shared E1 | `runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth` | `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35` |
| H E15 primary | `runs/h2_clean_factorial_e20_20260902_ampfix/H/adapter_15.pth` | `6830137b52fc16321192909fe7b3ca7565664afe61a8592fe7b7a78d2dff4771` |
| H E20 secondary | `runs/h2_clean_factorial_e20_20260902_ampfix/H/adapter_20.pth` | `a3857d018014b9ab77a06568e3c04e7fd8a4c0a7c24ccec3796e706c009dd087` |
| A E15 primary | `runs/h2_clean_factorial_e20_20260902_ampfix/A/adapter_15.pth` | `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe` |
| A E20 secondary | `runs/h2_clean_factorial_e20_20260902_ampfix/A/adapter_20.pth` | `de3d160efc4cdc24454594274ca01c4d36724ecfa1ec75d2a660d4b6c037ff8c` |
| C E15 primary | `runs/h2_clean_factorial_e20_20260902_ampfix/C/adapter_15.pth` | `3530d7cb9f912c37de07fb984beca927460237c222f4f3a09a5c1cad2e035b71` |
| C E20 secondary | `runs/h2_clean_factorial_e20_20260902_ampfix/C/adapter_20.pth` | `80be0b3c3e43f1afd2003cc1a7fb4193f24eb5a980da32d338523062d882451f` |
| AC E15 primary | `runs/h2_clean_factorial_e20_20260902_ampfix/AC/adapter_15.pth` | `2364984ad94faf3ab70833f6d0ad1e544daeebaf6e5ab0349a348b11322d608d` |
| AC E20 secondary | `runs/h2_clean_factorial_e20_20260902_ampfix/AC/adapter_20.pth` | `b0b1b20e937577d006ad1910f0b93b099d7aff894d19936787fe634ff10dd934` |

Intermediate E2-E19 checkpoints are not archived in this handoff. The
selected E15/E20 files and shared E1 are the preregistered result and resume
set. The original full run remains at
`/tmp/h2_clean_factorial_e20_20260902_ampfix` on the source machine.

## Scientific identity

- Full-run code fingerprint:
  `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`.
- Base H2 commit: `e03966997d4cecfd985943a4053a93e1e40197ec`.
- CIR reference commit: `9cc0ad4cc6b34e34a8c15e74df881866516b3181`.
- Freeze manifest: [`H2_4ARM_E15_E20_FREEZE.json`](H2_4ARM_E15_E20_FREEZE.json).
- Final audit: [`H2_4ARM_FINAL_PROTOCOL_AUDIT.md`](H2_4ARM_FINAL_PROTOCOL_AUDIT.md).
- Published result report:
  [`ACD_CLIP_PHASE_COMPARISON_PUBLISHED.md`](../results/ACD_CLIP_PHASE_COMPARISON_PUBLISHED.md).

The later H/A seed-1 and seed-2 attempts are intentionally excluded from this
eligible checkpoint set. They completed training but failed hard validity due
to non-finite-gradient skips and H/A global-step mismatch; no target metrics
were produced for them.

## Restore on another machine

```bash
git clone -b research/h2-clean-repro-anchor-cir-v1 \
  https://github.com/XLaiHuy/ACD-CLIP-.git
cd ACD-CLIP-
git lfs install
git lfs pull
git lfs ls-files | grep h2_clean_factorial_e20_20260902_ampfix
```

The CLIP weight is already an LFS-tracked repository artifact. Dataset files
are not committed and must be provisioned separately on the new machine using
the same dataset manifest identity recorded in the freeze manifest.
