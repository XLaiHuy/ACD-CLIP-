# H2_THBR_R1 audit decision

- `HARD_BACKGROUND_TAIL_PREMISE=SUPPORTED`
- `BOUNDARY_ARTIFACT_RISK=LOW`
- `FOCAL_REDUNDANCY=LOW`
- `THBR_TRAINING_AUTHORIZED=YES`

The audit is source-only. It uses the frozen 96-image VisA endpoint cohort, existing 7x7 boundary/near/far semantics, and 16 fixed E10-state source batches without optimizer updates. The GradBudget endpoint is descriptive evidence only and remains numerically invalid under its prior paired-run decision.

Audit rubric values: top-background-1% mass concentrations=[0.9980220332221782, 0.9949721762489571, 0.9966292061268112]; top-P near-background fractions=[0.2786043708849341, 0.2904791407372565, 0.27129062068985915]; mean top-background-1% near-boundary fractions=[0.1634557082900837, 0.16602274754637608, 0.16324557454449903]; THBR/focal gradient cosine median=0.09488262981176376; focal top-K overlap=0.10299576166279395.

The audit authorizes exactly one bounded THBR candidate/control screen.

No target inference or automatic follow-up is authorized.
