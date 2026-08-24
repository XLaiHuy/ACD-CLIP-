# P24 Claim Safety

| Candidate claim | Safety | Reason |
| --- | --- | --- |
| Source-trained selective correction for a frozen ZSAD detector | SAFE | Persisted source-only evidence |
| Signed intervention risk can be separated from benefit | SAFE WITH QUALIFIER | Safety evidence is supported; benefit remains unresolved |
| Harm-aware filtering reduces wrong-sign failures | SAFE | P11 preregistered science |
| Tested image-level budget selection is broad but lacks required pAP headroom | SAFE | P23 A0/A1 gates |
| Patch diagnostics motivate finer actionability research | SAFE WITH QUALIFIER | P13 is post-hoc and GT-bearing |
| First selective/risk-aware/image-conditioned method | UNSAFE | No novelty audit or exhaustive prior-art evidence |
| Learned patch controller works | UNSAFE | No such study exists |
| External industrial/MVTec superiority or ACD-CLIP outperformance | UNSAFE | No external validation exists |

All P13 oracle statements must retain the label **POST_HOC_ORACLE_DIAGNOSTIC**.
No claim may substitute pAUROC for the frozen pAP gate or infer external
generalization.
