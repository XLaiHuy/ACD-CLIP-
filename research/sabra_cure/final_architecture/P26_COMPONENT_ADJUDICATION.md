# P26 Component Adjudication

| Component | Evidence | Decision | Final-runtime role |
| --- | --- | --- | --- |
| Frozen Phase2B | Canonical source detector throughout all valid studies | `RETAIN_DEPLOYABLE` | Sole inference model |
| Signed proposal direction | R0 headroom and R1 sign/rank transfer | `RETAIN_DISABLED_REFERENCE` | None; no reachable correction |
| R1 magnitude head | R1 G3 failure | `DROP` | None |
| R1 uncertainty / R2 selector | Safety improved, downstream pAP gates failed | `DROP` | None |
| R2-v2 harm-risk head | Strong safety evidence but no positive/stable value gate | `RETAIN_DISABLED_REFERENCE` | None |
| P14 image value ridge | Weak value correlations and failed final gates | `DROP` | None |
| NATIVE/SAFE20/SAFE30/EXPAND40 contextual family | Broad safe gain but P23 H3 headroom failed | `DROP` | None |
| P25 patch-benefit RankNet | Valid P25R3 Q1; G2-G6 failed | `DROP` | None |
| Native fallback | Fully specified and requires no unvalidated selector | `RETAIN_DEPLOYABLE` | Always selected |

The adjudication distinguishes a useful component in isolation from a valid
complete policy. Direction and harm control retain scientific value, but
neither establishes whether a safe proposed action will improve final ranking.
P25R3 directly tested the last authorized patch-benefit formulation and found
it non-identifiable under the frozen source protocol. Consequently no tested
correction is allowed to reach external inference.

The final action space is exactly `{KEEP}`. Correction coverage is `0`; every
patch abstains to native. Historical alpha `0.25` is recorded for provenance
but is unreachable and does not alter logits.
