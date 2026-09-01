# Pre-MVTec final freeze

Status: **FROZEN** before MVTec access.

The source-only freeze selected **RA E16** (`adapter_16.pth`) as the final candidate by highest deterministic-source Pixel AUROC. The Medical evaluation was then run only as the preregistered confirmation/ablation; it cannot replace the source-selected checkpoint.

Final configuration:

- Architecture: exact historical H2 plus selective image-parameter anchor.
- Deployment: native H2, alpha = 0.0. CIR is not applied at inference.
- Checkpoint: `runs/h2_anchor_cir_master_20260901/RA/adapter_16.pth`
- Checkpoint SHA256: `2e5b27b7744b571f9166c1d8ead99fcc85039f414635ba8b8aed5a999999eba0`
- Config SHA256: `c1be71655ca245516f67a20e8712e52251e37ca627bdade77bf78f809f283876`
- Architecture freeze SHA256: `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86`
- CLIP asset SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`

Medical confirmation was mixed: RA improved Pixel AP over R but lowered Pixel AUROC; RCA did not show robust matched dominance over RA. This does not alter the already frozen candidate and is recorded as a limitation, not a post-hoc selection rule.

MVTec authorization is one final-winner evaluation only. No R/RA/RCA variants, target tuning, checkpoint replacement, or post-MVTec tuning is permitted.
