# P24 Architecture Decision

| Component | Evidence | Status | Reuse? |
| --- | --- | --- | --- |
| Frozen Phase2B | Frozen across valid studies; firewall audits pass | SUPPORTED | REUSE |
| Signed proposal direction | Direction transfer and action evidence | SUPPORTED | REUSE |
| Harm-risk estimator | P11 safety reduction; P20/P23 safety gates | SUPPORTED | REUSE |
| SAFE20-only controller | SAFE20 pAP below NATIVE in P20 | NOT_SUPPORTED | Do not use as mandatory baseline |
| P14 image-level value ridge | Weak value Pearson/Spearman/sign accuracy | CLOSED | No |
| Image-level N/S20/S30/S40 family | P23 A0/A1 both fail H3 | CLOSED | No |
| SAFE30 | A1-A0 = 0.0019785 percentage points | CLOSED for core | Drop |
| Image-level RankNet | Stage D correctly skipped for insufficient action-family headroom | NOT JUSTIFIED | No |
| More image-level budgets/q levels | SAFE30 negligible; coarse family fails H3 | CLOSED | No |
| Alpha sweep | No supporting evidence; alpha frozen | NOT JUSTIFIED | No |
| Patch-level benefit selector | P13 opportunity; P23 compression; GT-free predictability unresolved | NEW RESEARCH REQUIRED | Only after review/preregistration |
| MVTec | No final architecture result | UNTOUCHED | External gate remains closed |
| Medical | Firewall | FORBIDDEN | No |

P23 is not evidence that the actions fail across classes: A0 and A1 improved
12/12 classes and passed all safety/breadth/guardrail gates except H3.  It is
evidence that their **macro pAP magnitude** is insufficient for the frozen
headroom rule.  Strong pAUROC is supporting evidence and cannot substitute for
the primary pAP H3 gate.

**Decision:** `P24_PATCH_LEVEL_STUDY_JUSTIFIED`.  This justifies a new
hypothesis, not a solution: a frozen-detector, GT-free, low-capacity patch
selector must still prove that it can identify positive ranking-level value
under nested, leakage-safe source evaluation.
