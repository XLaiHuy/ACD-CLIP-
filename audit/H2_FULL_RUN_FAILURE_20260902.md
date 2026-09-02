# H2 full-run failure audit — 2026-09-02

Status: invalid and excluded from scientific analysis.

The fresh factorial attempt at `/tmp/h2_clean_factorial_e20_20260902` stopped in arm H during E8 at batch 257 after 21 non-finite-loss skips exceeded the configured threshold of 20. The first diagnostic reported finite input and finite Anchor loss, but non-finite detection/segmentation predictions and non-finite stage-3 SS2D diagnostics. The preserved diagnostic is:

`/tmp/h2_clean_factorial_e20_20260902/H/nonfinite_diagnostics/epoch_008_batch_00257_skip_0001.pth`

The decisive protocol defect was earlier in the launch path. `main()` loaded and validated the resume checkpoint and copied its epoch/global-step into the call, but did not pass the loaded `resume_payload` into `train()`. Consequently each arm began with newly initialized model/optimizer/scheduler/scaler/RNG state while starting its loop at the resumed epoch number. The resulting artifacts are not a shared-E1 factorial and must not be evaluated or compared.

The defect is fixed in commit `3f4347c99853f66ca6697150a87f48fb261ae6e1`, which passes `resume_payload=resume_payload` to `train()` and adds `tests/test_train_resume_wiring.py`. The full main test suite passed (`47 passed, 58 warnings`). A fresh factorial root is required; the failed root is retained for audit evidence only.
