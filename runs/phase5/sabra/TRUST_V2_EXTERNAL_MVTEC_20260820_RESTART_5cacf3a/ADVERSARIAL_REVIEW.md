# SABRA Trust-v2 MVTec external review

- Exact external run used the pushed boundary `5cacf3abc5df271adbe655c318d516777fd8ad61`.
- GT-free construction completed before MVTec labels/masks were read.
- Trust-v2 uses the frozen M1_E_Credibility model and frozen Need C1 parameters.
- PCRR remains DROP; no MVTec tuning or refitting was performed.
- Trust-v2 MVTec gate is SUPPORTED.
- The registered Authority-v2 primary comparator is unresolved because frozen M0_E OOF parameters are not persisted.
- The raw Authority comparison is diagnostic-only and has a catastrophic negative tail; it is not substituted for the primary comparator.
- Medical data was not accessed. Full 20e training was not started and is not authorized.
