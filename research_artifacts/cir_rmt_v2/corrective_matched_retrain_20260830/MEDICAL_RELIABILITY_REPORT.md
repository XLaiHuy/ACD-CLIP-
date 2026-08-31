# Corrected Medical evaluation reliability report

Status: PASS — exact matrix complete and auditable.

## Coverage and integrity

The evaluator completed 108/108 planned cells:

- methods: P, C0, C05;
- epochs: E10, E12, E14, E16, E18, E20;
- targets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir;
- all pixel AUROC/AP values finite;
- Colon image AUROC/AP are intentionally undefined under the frozen evaluator and remain blank in the compact CSV;
- every completed cell has an atomic JSON payload, a journal entry, and a verified cell SHA;
- no FAILED.json exists in the completed Medical output root.

The authoritative compact result is corrected_medical_decomposition.csv. The raw matrix is retained outside the tracked archive at the run root; raw per-pixel spools were cleaned after exact metric computation.

## Resource admission and resilience

RESOURCE_ADMISSION_REPORT.json records a bounded balanced Brain preflight using 24 images, four batches, full 518×518 pixel spools, exact metrics after model/worker teardown, and the same frozen forward/evaluator path. Admission was SAFE: bounded VRAM/RSS, clean worker shutdown, 43 open file descriptors after teardown under a 1024 soft limit, and approximately 178.8 GiB free disk. The conservative full-Brain cell spool estimate was approximately 14.9 GiB.

SPEED_MEMORY_TELEMETRY.csv records training rows, source evaluation cells, Medical cells, and the preflight sample. OOM_KILLED_INCIDENTS.csv has a NO_REAL_INCIDENTS record: no CUDA OOM, SIGKILL, or killed process occurred. The first source harness error (KeyError: img_size) and the first unbalanced preflight sample were corrected before any scientific cell was accepted and were not OOM events.

## Protocol and metadata caveat

P, C0, and C05 used the same deterministic Medical loader, frozen evaluator, 518 resolution, FP32 policy, and exact disk-backed metric path. C0 and C05 came from the same CIR forward pass; C05−C0 is therefore a paired inference effect conditional on the corrected CIR representation.

The unmodified runner writes the CLI CIR config SHA into P-cell metadata even though P model construction uses the parent config. This is documented metadata provenance, not a score-path change. The compact table records parent_config_sha256 for P and cir_config_sha256 for C0/C05, and checkpoint SHA/manifest identities are the authoritative method identity.

No target tuning, threshold selection, alpha selection, MVTec training, precision change, resolution change, or automatic scientific retry was performed.
