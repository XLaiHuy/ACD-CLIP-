# ACD-CLIP Phase4 P1-v8 Experiment Decision Tree

```mermaid
graph TD
    Start[P1-v8 Target Architecture Evaluation] --> T1[Tier-1: Vector Specialization Probe]
    T1 -->|delta-T cos < 0.90 & rank >= 2.0| T1_PASS[Vector Specialization: PASS]
    T1 -->|delta-T cos >= 0.90| T1_FAIL[FIX_FACTOR_SPECIALIZATION]

    T1_PASS --> T2[Tier-2: Functional Specialization Probe]
    T2 -->|factor logit corr < 0.90 & dense/sparse diffs > 1e-4| T2_PASS[Functional Specialization: PASS]
    T2 -->|factor logit corr >= 0.99| T2_FAIL[Functional Specialization: FAIL]

    T2_FAIL --> T3[Tier-3: Unsupervised Cluster Responsibility]
    T3 -->|cluster anomaly share gap > 10% & loss share <= 5%| T3_PASS[READY_FOR_8_EPOCH_VALIDATION]
    T3 -->|loss share 44-52% & anomaly gap = 0.677%| T3_FAIL[FIX_LOCAL_OBJECTIVE]

    T3_FAIL --> Next[Mask-Supervised Patch Strata Objective]
```

## Current Decision State
- **State**: `FIX_LOCAL_OBJECTIVE`
- **Status of 8-Epoch / 20-Epoch Gate**: `BLOCKED`

## Decision Matrix Summary

| Milestone | Gate Criteria | Result | Action / Outcome |
| :--- | :--- | :--- | :--- |
| **Metric Parity** | Match baseline evaluation results | `PASS` | Verified |
| **50-Batch Wiring Smoke** | Full backward & forward execution without NaNs | `PASS` | Verified |
| **Tier-1 Vector Diversity** | $\Delta T$ Cosine $< 0.900$, Rank $\ge 2.0$ | `PASS` | $\Delta T$ median cosine $= 0.8930$, rank $= 2.64$ |
| **Tier-2 Functional Diversity** | Factor logit correlation $< 0.900$ | `FAIL` | Logit correlation $= 0.999269$ |
| **Tier-3 Responsibility Calibration** | Loss share $\le 5\%$, Anomaly gap $> 10\%$ | `FAIL` | Loss share $= 44 - 52\%$, Anomaly gap $= 0.677\%$ |
| **Final Decision** | Choose exact next objective | **`FIX_LOCAL_OBJECTIVE`** | Plan mask-supervised strata |
