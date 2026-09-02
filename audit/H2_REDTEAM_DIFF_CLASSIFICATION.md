# H2 pre-full-train red-team diff classification

Reference implementation: frozen CIR-V2 commit `9cc0ad4cc6b34e34a8c15e74df881866516b3181` in `tools/cir_rmt/core.py` and `scripts/cir_rmt/train_h2_anchor_cir.py`.

Reference H2 parent: `e03966997d4cecfd985943a4053a93e1e40197ec`.

| Change | Classification | Review conclusion |
| --- | --- | --- |
| `h2_clean/cir_v2.py` | `EXACT_CIR_V2_PORT` | Frozen midpoint/MAD peer selection, delta, score-space transport, and optimized score ported with the frozen constants and direction. |
| `model/adapter.py` and `h2_clean/__init__.py` | `EXACT_CIR_V2_PORT` | Native H2 DFG weights/logits, detached peer delta, transported weights, and final logits are wired to the exact path; the additive shift is not live. |
| `h2_clean/contract.py` | `REPRO_PLUMBING`, `SAFE_ANCHOR`, `AUDIT_ONLY` | Full-state scientific identity and parent/operational separation are enforced; anchor summation is order-stable; `CIR_LOGIT_SHIFT_EXPERIMENTAL` is retained only as a disabled guard. |
| `train.py` | `REPRO_PLUMBING`, `AUDIT_ONLY` | Resume checks, deterministic batch tracing, sorted diagnostics, and opt-in anchor gradient auditing add no objective coefficient or schedule change. |
| `scripts/run_h2_clean_factorial.sh` | `REPRO_PLUMBING` | Pre-start hash/determinism settings, fresh-root guard, fresh shared E1 creation, exact checkpoint preflight, and H/A/C/AC launch wiring. |
| `scripts/run_h2_clean_smoke.sh`, `scripts/validate_h2_clean_smoke.py` | `EVALUATOR`, `AUDIT_ONLY` | Bounded shared-E1/fork/resume checks; no full training or target evaluation. |
| `scripts/run_h2_redteam_visa_audits.py` and `audit/h2_redteam_visa_*` | `EVALUATOR` | Source-only fixed VisA batch evidence for CIR parity and E5/E10/E15 anchor strength. |
| `tests/test_*.py` additions/updates | `AUDIT_ONLY` | Regression coverage for exact CIR parity, stale resume rejection, telemetry parity, and disabled additive shift. |

No unexplained scientific change was found. `anchor_lambda` remains `0.001`; no Medical, MVTec, target-label, or full-training step was run. The anchor audit classified the current source-only anchor as `EFFECTIVELY_INACTIVE`, so A/AC full training remains blocked pending the single source-only preregistered scaling rule recorded in `audit/h2_redteam_visa_anchor.json`.
