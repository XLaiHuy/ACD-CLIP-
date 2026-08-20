# SABRA Trust-v2 MVTec external validation

## Result

- Trust-v2: `SUPPORTED`
- Authority-v2 primary: `UNRESOLVED_MISSING_FROZEN_M0_E_CALIBRATOR`
- Authority-v2 raw secondary: `FALSIFIED`
- Next full SABRA training study: `NOT AUTHORIZED`

Trust class-level image AUROC deltas have mean `0.210747`, median `0.178363`, and `14/15` positive classes. The registered Authority comparator requires the frozen M0_E OOF calibrator, which is not present in the frozen artifacts. No substitute or MVTec-fitted comparator was created.

The exact run used pre-external boundary `5cacf3abc5df271adbe655c318d516777fd8ad61`; medical reads were zero and 20e training was not started.
