# Seed-0 provenance reconciliation

Status: `PASS` with a metadata-only path correction.

The Medical `C_E15/command.txt` records evaluation of `/tmp/h2_clean_factorial_e20_20260902_ampfix/C/adapter_15.pth`, and its SHA256 matches the frozen C E15 checkpoint. The compact Medical manifest previously named the nonexistent pre-AMP-fix path `/tmp/h2_clean_factorial_e20_20260902/C/adapter_15.pth`; this is classified as `METADATA_PATH_INCONSISTENCY_ONLY`, not as a scientific or evaluation failure. The manifest now records the evaluated/frozen path and the stale path explicitly.

The clean evaluator fixes `image_score=cls_only`. The historical evaluator used a Medical image-score blend (`0.5 image + 0.5 pmax`) and an industrial blend, so historical image metrics are not directly comparable: `IMAGE_METRIC_HISTORICAL_COMPARABILITY=NO`. Pixel metrics and the frozen target protocol are unaffected.

The scientific source files are unchanged from seed-0 commit `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`; the current additions are audit/export/protocol plumbing only. Therefore `SCIENTIFIC_CODE_PARITY_WITH_SEED0=PASS`.
