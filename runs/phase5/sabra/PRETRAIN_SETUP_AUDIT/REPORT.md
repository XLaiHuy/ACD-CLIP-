# SABRA pre-training setup audit

## Terminal status

`PRETRAIN_LOGIC_AUDIT_READY`

This is an infrastructure/readiness audit only. No SABRA scientific logic
audit, training, medical evaluation, MVTec sample inspection, or historical
P5FR1C/P5FR1CE1 job was run.

## Provenance

- Branch: `research/p5-sabra-g`
- Starting/audit HEAD: `1baa524bc8723a4ac1e1bc54c2c2c69e49f736ca`
- Required handoff HEAD: `1baa524bc8723a4ac1e1bc54c2c2c69e49f736ca`
- Artifact input branch: `origin/artifacts/p5-runtime-inputs`
- Artifact input commit: `316ce9d4a9ddf742cf17f1c98c5011891c90ab08`
- Current working-tree status was recorded in `GIT_PROVENANCE.json`; only
  intended setup files and the documented runtime config were introduced.

## Critical checks

| Check | Result |
|---|---|
| Git provenance | `True` |
| Python/dependency environment | `True` |
| CUDA/GPU | `True` |
| Phase2B checkpoint | `True` |
| CLIP asset | `True` |
| Phase2B config | `True` |
| VisA metadata/files | `True` |
| Deterministic loader | `True` |
| Phase2B frozen load | `True` |
| Native deployment parity | `True` |
| GT firewall | `True` |
| Source/implementation readiness | `True` |

## Data and domain firewall

VisA is resolved through the documented path adapter
`/workspace/data/VisA_20220922 -> /workspace/data/data/VisA_20220922`.
The GT-free path is `tools/sabra/data.py::VisaEvidenceDataset`; its runtime
mask-read guard passed. `VisaEvaluationDataset` is separate and was used only
to validate deterministic mask loading. `mvtec_science_reads` is
`0` and `medical_reads` is
`0`.

## Required artifacts

All required JSON reports are in `/workspace/ACD-CLIP-/runs/phase5/sabra/PRETRAIN_SETUP_AUDIT`:

`ENVIRONMENT_AUDIT.json`, `GIT_PROVENANCE.json`, `MODEL_ASSET_AUDIT.json`,
`VISA_DATA_AUDIT.json`, `DOMAIN_FIREWALL_AUDIT.json`,
`PHASE2B_LOAD_AUDIT.json`, `PHASE2B_DEPLOYMENT_PARITY.json`,
`DETERMINISM_AUDIT.json`, and `READINESS_DECISION.json`.

## Prompt 2 handoff environment

```bash
source /workspace/ACD-CLIP-/.runtime/miniconda3/bin/activate
conda activate torchhuy
export ACDCLIP_DATA_ROOT=/workspace/data
export ACDCLIP_CLIP_VITL14_336=/workspace/ACD-CLIP-/.runtime/assets/ViT-L-14-336px.pt
```

Prompt 2 may begin only after reviewing these artifacts. No Prompt 2 command
was started automatically.

## Source readiness

The existing adapter, authoritative B1/nonlocal peer implementation, and
canonical PGM/PCRR modules were imported. Their formulas were not rewritten;
Need/Trust scientific logic was not implemented.
