# H2 Medical-selected final target report

## Frozen result

Medical macro pixel AP selected E15 under fixed weighted inference. The
selected checkpoint is:

`/workspace/h2_safe_anchor_e20_medical_selected/adapter_15.pth`

SHA256: `11deefa24b1a6138bc833689cdee6ab247c8b420ba8fb160334abd1e2991e094`

Medical was used as `TARGET_VALIDATION_FOR_EPOCH_SELECTION`; it is not an
untouched zero-shot final benchmark. MVTec was observed only after this freeze
and was not used to revise the epoch.

| Target / inference | Pixel AUROC | Pixel AP | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Medical E15 equal | 89.0802473607016 | 35.48717926990417 | 74.99794363975525 | 77.58847077687581 |
| Medical E15 weighted | 89.28949333223254 | 35.472911947909374 | 74.99794363975525 | 77.58847077687581 |
| MVTec E15 equal | 86.89838438614198 | 41.89632001389817 | 88.09939543406169 | 94.42826112111409 |
| MVTec E15 weighted | 87.19126575759987 | 41.82859892570771 | 88.07996153831482 | 94.41276510556538 |

Weighted minus equal deltas at the selected epoch:

- Medical pixel AUROC: `+0.20924597153093316`; pixel AP: `-0.014267321994793747`.
- MVTec pixel AUROC: `+0.29288137145788085`; pixel AP: `-0.06772108819045997`.

Descriptive best observed macro pixel AP epochs across E10–E20 were Medical
equal E15 and weighted E15; MVTec equal E13 and weighted E13. By macro pixel
AUROC, Medical equal/weighted were E15/E15 and MVTec equal/weighted were
E12/E12.

## Protocol identity

- Source-only training: VisA; Medical and MVTec were not used during training.
- Full-state resume: E1 checkpoint SHA256
  `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`;
  training completed through E20 and retained E10–E20.
- Safe Anchor: ON; `anchor_lambda=0.0021633926715180626`; family
  `rho=0.10`; functional feature anchor OFF.
- Train-time fusion: equal `[1/3, 1/3, 1/3]`; weighted fusion was inference-only.
- Inference: equal `[1/3, 1/3, 1/3]` and fixed weighted
  `[0.3736138197153701, 0.3270300383596602, 0.2993561419249697]`; no fusion
  weight sweep and no target hyperparameter sweep.
- Numerical validity: PASS for every retained E10–E20 checkpoint. Nonfinite
  gradient skips occurred at E11 and E17; no nonfinite loss skips occurred in
  the retained epochs.
- Evaluator provenance: pinned Medical evaluator commit
  `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`; raw exact metrics, pixel stride
  1, current shared prompt configuration.

`MEDICAL_TARGET_SELECTION_USED=YES`

`MEDICAL_UNTOUCHED_ZERO_SHOT_CLAIM_ALLOWED=NO`

`MVTEC_OBSERVED_BEFORE_SELECTION=NO`

`MVTEC_USED_FOR_SELECTION=NO`
