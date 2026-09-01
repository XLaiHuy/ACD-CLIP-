# Machine handoff manifest

This manifest lists the large checkpoint files that must be copied or backed up externally. Checkpoints are intentionally not committed to Git. Hashes were recomputed on 2026-09-02.

## Code and data identity

| Item | Identity |
|---|---|
| Snapshot parent Git SHA | `9cc0ad4cc6b34e34a8c15e74df881866516b3181` |
| Branch | `research/cir-dfg-rmt-v2-signfix` |
| Architecture | `CIR_DFG_RMT_V2`, version 2 |
| Architecture freeze SHA256 | `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86` |
| H2 repository commit | `e03966997d4cecfd985943a4053a93e1e40197ec` |
| H2 `train.py` SHA256 | `9f0d1879d8073a5199da6967a8f4a17f65a5fd4949e60e04eede68ba964111d5` |
| H2 `model/adapter.py` SHA256 | `eb7ac87ba659cbc5392b89f581300b06c868fcf79d30a23406f6dab32d1302cf` |
| H2 evaluator SHA256 | `7bdd8cc6ada90467285a79ced9599ed778c6dc2a0ba6596d2f3311fa637fae9d` |
| Current H2 config SHA256 | `c1be71655ca245516f67a20e8712e52251e37ca627bdade77bf78f809f283876` |
| Current extension runner SHA256 | `c75afcdf4eaaa9eb31eb87d29da60af36265ee40742f53d827827ed3cb1fa4ed` |
| Final-night Anchor audit script SHA256 | `6fe7190b371fb23a0e5b970fb32ad8cbfa1ac4806de4c01603e999e87678c072` |
| Current `parameter_anchor.py` SHA256 | `3014d2bedf800b42208b69c406eaa6c37752a80c9687aa500e1fecfb30062845` |
| Current H2 Medical evaluator SHA256 | `025adefc0daba90a43754c09cb3e29a9891ed9b9b078c51a19cd58da7272dbbd` |
| CLIP asset SHA256 | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| VisA manifest SHA256 | `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842` |

## Checkpoint paths and SHA256

| Role | Absolute path | SHA256 |
|---|---|---|
| Historical H2 E1 Anchor reference | `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_1.pth` | `da893014cdd9ca643f632cedbf5d43fc57eb3acee343e4795bc7de9aa12c3074` |
| Historical H2 E10 | `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth` | `ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb` |
| Historical H2 E15 | `/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_15.pth` | `2ee8e411b18e69da7734480091f2a549d71a98ba80eee9adc8b8d62d09d190af` |
| Shared H2 E0 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/common/e0.pth` | `119ba08eb8aa8107f47bf0a62ccc1c9ee643cd1f395331a527b1c975ea1d3eca` |
| R E10 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/R/adapter_10.pth` | `e167a5a53c314f8e7f2ca84cd0ad44578564dc10d45212d6d746b8637cbdb2a8` |
| R last | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/R/last.pth` | `9760e0071799eb3370f5993dbfa59cc8f83f62b8d7bf1b941b70bf38445e2873` |
| RA E10 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_10.pth` | `8f7597db9a874af77658c176bf0ae188de484c02d4ccaa56f2db1f443a58560d` |
| RA E12 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_12.pth` | `8d099f039b2f7da6466b6995e88440e8f67cf1f1856657ecc9f8d3f37d6d28f2` |
| RA E14 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_14.pth` | `05d0fd4f76ba6b901778066e9b9f91c020a9c83e2b93e3969a4ae40a9b5e3b6a` |
| RA E16 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_16.pth` | `2e5b27b7744b571f9166c1d8ead99fcc85039f414635ba8b8aed5a999999eba0` |
| RA E18 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_18.pth` | `af010033abcfa30f9064d388be6b0e1a4ac01749be3806ec5f65646efb732a6e` |
| RA E20 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/adapter_20.pth` | `3430862665c4f834ca9fab56e8e919cdf39f392257acc5e937bb02629c6b26b6` |
| RA last | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RA/last.pth` | `19a61fe8f4b46c9c6875043fe99993f149d83437c418d75801b8de34c4e599f1` |
| RCA E10 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_10.pth` | `df73c220204c658d8119cde79f726667361a04f231d656d5ee7404b7f5327a44` |
| RCA E12 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_12.pth` | `ffb82017c7eec844bdf1a6079fb70b25c88d6743e77e30bde53c19245f095801` |
| RCA E14 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_14.pth` | `466b4e7849a0748dc71236c57d52e921c7a0862c9dc80642975bbfc44d0f589a` |
| RCA E16 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_16.pth` | `d34af8c46bdeb3606d364fa23aa4dd543d4d41508e1511fca608bc8503b7a25f` |
| RCA E18 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_18.pth` | `ebcc076121d9fe85b783c01fd97e3fe888a239e04f919dd1d0e7c6aa24b048c7` |
| RCA E20 | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/adapter_20.pth` | `243c91a6a82ad0c72f9cf86cda1271bde7791c741035da19419cfefe7d61cc81` |
| RCA last | `/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/RCA/last.pth` | `98e4865f232a8b8a97df7c0be4b6b09fd26938c2071177872a18b8e159f8d019` |

## Reproduction notes

- Same-E10 Medical output: `SAME_E10_MEDICAL.csv`; the transient evaluator spool is intentionally excluded.
- Anchor outputs were generated by `tools/cir_rmt/final_night_anchor_audit.py` with one fixed VisA training batch, seed 12345, no optimizer step.
- The exact temporary driver command for the same-E10 evaluation was not retained as a run-local file; the evaluator source, arguments, output, identities, and completion status are preserved in this directory.
- Raw checkpoints, memmaps, evaluator spools, caches, and all unrelated pre-existing untracked run directories remain outside the compact Git snapshot.
