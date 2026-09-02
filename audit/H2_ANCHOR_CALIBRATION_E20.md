# H2 Anchor calibration and E20 activation

## Frozen calibration

Source: `audit/h2_redteam_visa_anchor.json`.
Calibration scope: historical model-only VisA geometry; no optimizer or update-parity claim.

- Valid rows: `historical_E5/vision_text_k`, `historical_E10/vision_text_k`, and `historical_E15/vision_text_k`.
- Raw ratios: `18.520829477442355`, `33.84580840834666`, and `23.11184680352771`.
- `R_MED=23.11184680352771`.
- Target effective ratio is `0.05`; family cap rho is `0.10`.
- Historical lambda is `0.001`; active lambda is `0.0021633926715180626`.
- No lambda or rho sweep was performed.

The active value is exactly `target_effective_ratio / R_MED`. `vision_text_q` was `ANCHOR_MODERATE`, and near-zero, dormant, zero, and non-finite rows were excluded by the strict `TASK_ACTIVE` rule.

## Activation gate

Fresh root: `/tmp/h2_anchor_family_short_e20_20260902`.
Shared E1 checkpoint SHA-256: `9e48d6a5b6b9433c4f18870f3ecee785504e4f32d42c16778b807dbeb761aa7d`.

- H_short and A_active_short ran E2-E5 with two batches per epoch.
- H/A batch identities and transformed image/mask hashes matched.
- H remained native with zero effective Anchor contribution.
- A stayed within rho (`0.09999937754084845` maximum active-family effective ratio).
- A reached a meaningful active-family effective ratio of `0.03957792788580696`.
- All finite-skip, near-zero, image-only-difference, and no-pathology checks passed.

Result: `FAMILY_SAFE_ACTIVE`; full H/A/C/AC E20 training is authorized after the code is committed and pushed.

## Limits

The activation gate is source-only and does not use Medical, MVTec, target labels, or full-training results.
It authorizes the fixed active Anchor branch; it does not select E15 versus E20 or select a performance winner.
The full run must create a new shared E1 after the preflight code commit and preserve E15 as primary and E20 as secondary.
