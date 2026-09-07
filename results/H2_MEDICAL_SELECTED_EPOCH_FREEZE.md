# H2 Medical-selected epoch freeze

Protocol: `H2_SAFE_ANCHOR_E20_MEDICAL_SELECTED`

Medical is the target-validation set used for epoch selection. MVTec had not
been observed when this freeze was made.

| Field | Frozen value |
|---|---|
| Selected epoch | E15 |
| Selection metric | Weighted Medical macro pixel AP |
| Weighted Medical macro pixel AP | 35.472911947909374 |
| Weighted Medical macro pixel AUROC | 89.28949333223254 |
| Equal Medical macro pixel AP | 35.48717926990417 |
| Equal Medical macro pixel AUROC | 89.0802473607016 |
| Weighted minus equal AUROC | +0.20924597153093316 |
| Weighted minus equal AP | -0.014267321994793747 |
| Checkpoint | `/workspace/h2_safe_anchor_e20_medical_selected/adapter_15.pth` |
| Checkpoint SHA256 | `11deefa24b1a6138bc833689cdee6ab247c8b420ba8fb160334abd1e2991e094` |

Tie-breaking was weighted Medical macro pixel AUROC, then earlier epoch. The
Medical trajectory was evaluated at E10–E20 under both equal fusion
`[1/3, 1/3, 1/3]` and fixed weighted fusion
`[0.3736138197153701, 0.3270300383596602, 0.2993561419249697]`.

`MEDICAL_TARGET_SELECTION_USED=YES`

`MEDICAL_UNTOUCHED_ZERO_SHOT_CLAIM_ALLOWED=NO`

`MVTEC_OBSERVED_BEFORE_SELECTION=NO`

The selected Medical result is therefore not an untouched zero-shot benchmark.
MVTec is reserved for the post-selection holdout evaluation. No target
hyperparameter sweep or fusion-weight sweep was used.
