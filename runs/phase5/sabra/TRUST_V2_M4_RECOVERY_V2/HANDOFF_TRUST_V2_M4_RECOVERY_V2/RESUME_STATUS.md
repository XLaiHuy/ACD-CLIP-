# Resume status

`TRUST_V2_M4_RECOVERY_V2_EXTERNAL_VALIDATION_FAILURE`

`VALID=false` applies to the MVTec external-validation stage only. Its complete failure record is under `FAILED_EXTERNAL_VALIDATION_20260819_MVTEC_UNAVAILABLE/`. The VisA result, frozen candidate, and GT-free cache remain valid and usable. MVTec image/mask reads are 0, metadata probes are 1, medical reads are 0, and `FULL_20E_TRAIN_AUTHORIZED=false`. Next action, if ever permitted, is to supply the authorized MVTec image root and run only the frozen external evaluation.
