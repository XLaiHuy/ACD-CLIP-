# H2 pre-full-train red-team diff classification

Reference implementation: frozen CIR-V2 commit `9cc0ad4cc6b34e34a8c15e74df881866516b3181` in `tools/cir_rmt/core.py` and `scripts/cir_rmt/train_h2_anchor_cir.py`.

Reference H2 parent: `e03966997d4cecfd985943a4053a93e1e40197ec`.

| Change | Classification | Review conclusion |
| --- | --- | --- |
| `h2_clean/cir_v2.py` | `EXACT_CIR_V2_PORT` | Frozen midpoint/MAD peer selection, delta, score-space transport, and optimized score ported with the frozen constants and direction. |
| `model/adapter.py` and `h2_clean/__init__.py` | `EXACT_CIR_V2_PORT` | Native H2 DFG weights/logits, detached peer delta, transported weights, and final logits are wired to the exact path; the additive shift is not live. |
| `h2_clean/contract.py` | `REPRO_PLUMBING`, `SAFE_ANCHOR`, `AUDIT_ONLY` | Full-state scientific identity, the unchanged global Anchor formula, the complete family partition, and the opt-in rho=.10 family-safe gradient budget are enforced; CIR shift remains disabled. |
| `train.py` | `REPRO_PLUMBING`, `SAFE_ANCHOR`, `AUDIT_ONLY` | A/AC alone opt into the family-safe image-adapter gradient budget after unscale and before the existing clip/update path; H/C retain native gradients and all optimizer/LR/scaler settings. |
| `scripts/run_h2_clean_factorial.sh` | `REPRO_PLUMBING` | Pre-start hash/determinism settings, fresh-root guard, fresh shared E1 creation, exact checkpoint preflight, and H/A/C/AC launch wiring. |
| `scripts/run_h2_clean_smoke.sh`, `scripts/validate_h2_clean_smoke.py` | `EVALUATOR`, `AUDIT_ONLY` | Bounded shared-E1/fork/resume checks; no full training or target evaluation. |
| `scripts/run_h2_redteam_visa_audits.py` and `audit/h2_redteam_visa_*` | `EVALUATOR` | Source-only fixed VisA batch evidence for CIR parity and E5/E10/E15 anchor strength. |
| `tests/test_*.py` additions/updates | `AUDIT_ONLY` | Regression coverage for exact CIR parity, stale resume rejection, telemetry parity, complete family partition, cap/floor semantics, raw/lambda ratio separation, and disabled additive shift. |

No unexplained CIR or target change was found. `anchor_lambda` remains `0.001`; the historical model-only audit is geometry evidence only and makes no optimizer/RNG/update-parity claim. The old global `0.1 / median(raw global ratio)` rule is preserved as `DEPRECATED_UNSAFE_GLOBAL_SCALING` with `GLOBAL_LAMBDA_UPSCALING_AUTHORIZED=NO`; it is not executed. The bounded fresh family-safe check is `FAMILY_SAFE_BUT_NEGLIGIBLE`, so H/C remain eligible while A/AC full training stays blocked.
