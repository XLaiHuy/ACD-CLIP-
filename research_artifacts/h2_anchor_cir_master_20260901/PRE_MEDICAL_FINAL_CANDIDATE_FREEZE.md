# Pre-Medical final-candidate freeze

Status: `FROZEN` on 2026-09-01.

This freeze was created from deterministic VisA source evidence only. No
Medical or MVTec result, target label, or target-domain diagnostic was used.
All deployment uses the native H2 path with inference alpha=0; RCA CIR is a
training-only intervention.

## Frozen selection rule

Across every available R/RA/RCA candidate, select by:

1. highest deterministic-source Pixel AUROC;
2. highest deterministic-source Pixel AP as tie-break;
3. earliest checkpoint as final tie-break.

The fixed source sample contains 96 images from 12 VisA categories. It is a
deterministic assessment sample, but those categories were present during
VisA training, so this is not an unseen-category claim.

## Frozen source-selected checkpoints

| arm | epoch | source Pixel AUROC | source Pixel AP | checkpoint SHA256 |
|---|---:|---:|---:|---|
| R | 10 | 0.94393531 | 0.40508183 | `e167a5a53c314f8e7f2ca84cd0ad44578564dc10d45212d6d746b8637cbdb2a8` |
| RA | 16 | 0.97764765 | 0.49103001 | `2e5b27b7744b571f9166c1d8ead99fcc85039f414635ba8b8aed5a999999eba0` |
| RCA | 12 | 0.93345608 | 0.49347167 | `ffb82017c7eec844bdf1a6079fb70b25c88d6743e77e30bde53c19245f095801` |

`FINAL_CANDIDATE=RA_E16`, because it has the highest source Pixel AUROC
among all available candidates. The complete source table remains frozen in
`SOURCE_DECOMPOSITION.csv`; no Medical result may revise this choice.

R was intentionally stopped at E10 under FAST_RIGOR, so R-E12/E14/E16/E18/E20
are absent by protocol. RA and RCA were continued by exact resume through E20.

## Authorized Medical evaluation

Evaluate the frozen R-E10, RA-E16, and RCA-E12 cells on Brain, Liver, Retina,
Colon_clinicDB, Colon_colonDB, and Colon_Kvasir. Record Pixel AUROC/AP and
secondary image metrics where supported. No target tuning or checkpoint
replacement is allowed.

MVTec remains prohibited until the post-Medical architecture/checkpoint
decision and a separate final one-shot freeze.

See the machine-readable companion
`PRE_MEDICAL_FINAL_CANDIDATE_FREEZE.json` for hashes and paths.
