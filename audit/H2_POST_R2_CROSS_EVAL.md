# H2 Post-R2 Frozen 2x2 Cross-Evaluation

`PROTOCOL_ID=H2_POST_R2_CAUSAL_DECOMPOSITION_2X2_FROZEN_CROSS_EVALUATION`

`DIAGNOSTIC_ONLY=YES`

`ENDPOINT_IDENTITY=PASS`

`CROSS_EVAL_REPRODUCTION=PASS`

No training, optimizer/scaler step, backward pass, target inference, Medical/MVTec inference, tuning, or recalibration was performed.

## Frozen cells

| Cell | Checkpoint | Inference fusion | Final AUROC | Final AP | R_late |
|---|---|---:|---:|---:|---:|
| C_EQ | CONTROL / equal | `[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]` | 0.858624210801 | 0.367357224419 | 0.657385408878 |
| C_W | CONTROL / weighted | `[0.3736138197153701, 0.3270300383596602, 0.2993561419249697]` | 0.862974707733 | 0.368807667233 | 0.617303371429 |
| T_EQ | WEIGHTED-TRAINED / equal | `[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]` | 0.865152122223 | 0.361115267753 | 0.666281461716 |
| T_W | WEIGHTED-TRAINED / weighted | `[0.3736138197153701, 0.3270300383596602, 0.2993561419249697]` | 0.868829033325 | 0.361962335332 | 0.626247644424 |

## Interpretation inputs

Training-stage deltas are T_EQ minus C_EQ, so equal inference isolates the training-trajectory contrast.

- Stage-1 AP delta: `-0.012786217261`
- Stage-2 AP delta: `-0.017633093426`
- Stage-3 AP delta: `0.012361037277`

The complete compact metrics, background-tail diagnostics, and PR diagnostics are in the required CSV/JSON artifacts.
