# H2 THBR-R1 bounded decision

- `DECISION=FAIL`
- `HARD_BACKGROUND_TAIL_PREMISE=SUPPORTED`
- `BOUNDARY_ARTIFACT_RISK=LOW`
- `FOCAL_REDUNDANCY=LOW`
- `THBR_TRAINING_AUTHORIZED=True`
- `CONTROL_ATTEMPTED/SUCCESSFUL=500/499`
- `CANDIDATE_ATTEMPTED/SUCCESSFUL=500/499`
- `PAIR_SKIP_MATCH=True`
- `EXACT_BATCH_IDENTITY_MATCH=True`
- `R=0.2670493421165961`
- `LAMBDA_THBR=0.18723131689337608`
- `FINAL_AP_DELTA=-0.010847331119427928`
- `FINAL_AUROC_DELTA=-0.03728301430718173`
- `NEAR_BG_P95/P99_DELTA=0.008221146836876858/0.06695230543613492`
- `MATCHED_VIOLATION_MEAN_DELTA=0.01172799505221378`
- `RED_TEAM_CASE=TAIL_GATE_FAILURE_WITH_PERFORMANCE_DROP`

Far-background tail improves, but the near-background p95/p99 tail worsens while both final AP and AUROC fall; THBR does not solve the measured bottleneck.

No confirmatory or target evaluation run is authorized. See the JSON artifact for all stage metrics, tail statistics, inversion rates, matched-cardinality diagnostics, geometry/parameter drift, gates, and provenance.
