# H2 residual bottleneck root-cause table

All target claims below are bounded by the fact that Seed 1 and Seed 2 target
evaluation was not run. `Seed-0 discovery only` is not confirmatory evidence.

| rank / candidate cause | evidence for | evidence against | source-only evidence | target diagnostic evidence | confidence | likely Medical AP contribution | likely MVTec AUROC contribution | architecture change? | scientific-setting-only fix? | next experiment required? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. numerical gradient invalidity | 10 repeated nonfinite-gradient skips across four runs; unequal H/A successful steps | final checkpoints finite; exact operand is unknown | Strong repeated validity failure | No valid Seed-1/2 target metrics | High for validity, not target gap | Indirect / unknown | Indirect / unknown | No | Yes, after diagnosis | Source-only numerical trace |
| 2. protocol mismatch | public details incomplete; N/group and training/test details not fully recoverable | internal H2 contract passes; H/A evaluator is matched | Contract table is consistent | Oracle parity passes internally; public parity unknown | Moderate | Possible | Possible | No | Yes | Reconstruct public evaluator if needed |
| 3. evaluator mismatch | historical stride-4/rounding differs from current stride-1/raw exact | bounded oracle parity agrees with current reference | Exact evaluator is frozen internally | No public score arrays | Moderate externally, low internally | Possible | Possible | No | Yes | Public-protocol reconstruction |
| 4. training horizon | Seed-0 H/A E15 to E20 changes are metric-specific | no valid confirmatory horizon; E15 remains preregistered primary | loss keeps decreasing | E15/E20 target metrics exist only for Seed 0 | Low to moderate | Possible | Possible | No | Yes | None before validity repair |
| 5. LR dynamics | fixed image/text/prompt LRs and StepLR; numerical failure occurs during schedule | decay is consistent and no update telemetry proves excess | schedule is logged | no valid target curves | Low / unresolved | Unknown | Unknown | No | Yes | Source-only update trace |
| 6. optimizer behavior | Adam plus AMP is compatible with gradient instability | moments/update norms absent | no Adam state telemetry | none | Low / unresolved | Unknown | Unknown | No | Yes | Source-only optimizer-state trace |
| 7. loss conflict | classification and segmentation coexist; family cosines can be mixed | weighted KG/K magnitudes are small; separate loss gradients absent | no g_seg/g_cls cosine | none | Low / not demonstrated | Unknown | Unknown | No | Yes | Source-only per-term gradient trace |
| 8. loss scaling | regularizer weights are nonzero and prompt unfreezes at E4 | scalar weighted terms are small relative to main loss | no full gradient-scale report | none | Low / not demonstrated | Unknown | Unknown | No | Yes | Source-only scale instrumentation |
| 9. initialization | fresh E1 per seed; source endpoints differ and skip epochs differ | no valid target variance or parameter-distance analysis | trajectory variation observed | target variance unavailable | High concern, not estimated | Unknown | Unknown | No | Yes | Valid multi-seed replication |
| 10. seed variance | robust effect cannot be tested with invalid seeds | same skip count by arm across both seeds | validity hazard repeats | no confirmatory metrics | High concern, not estimated | Unknown | Unknown | No | Yes | Valid multi-seed replication |
| 11. prompt drift | alpha reaches .2; soft prompt changes after unfreeze; context gradients logged | no target text margin or feature dump | source drift is visible | none | Possible | Possible | Unknown | No | Yes | Source-only prompt trace |
| 12. image adapter drift | image branch is trainable and source loss improves | drift distances are not stored | no family update norms | none | Unknown | Unknown | Unknown | No | Yes | Checkpoint-distance audit |
| 13. DFG behavior | stage weights are peaked and A/H route differently | weights finite and normalized; correctness is unknown | DFG is not collapsed | no target DFG trace | Possible | Possible | Possible | Not established | Yes | Target-free source trace first |
| 14. source overfitting | source loss decreases while target behavior is heterogeneous | no source validation curve; target seeds invalid | source fit improves | Seed-0 E15/E20 only | Possible | Possible | No | Yes | Valid target replication |
| 15. medical domain shift | medical datasets are heterogeneous and AP differs by dataset | feature-level shift not measured | no source-target feature statistics | no valid multi-seed target features | Unknown | Possible | N/A | Not established | Yes | Feature dump and descriptive shift audit |
| 16. pixel ranking | AP and AUROC move differently on ColonDB, zipper, toothbrush | no PR/ROC arrays or rank positions | no ranking telemetry | Seed-0 aggregate metric pattern only | Unknown | Possible | Possible | No | Yes | Store score maps/rank arrays |
| 17. small-anomaly localization | AP can be sensitive to sparse positives | no mask-size/area or morphology stratification | none | no per-image score maps | Unknown | Possible | Unknown | Not established | Yes | Mask-joined score-map diagnostic |
| 18. architecture capacity | none | source loss falls; target degradation/invalidity dominates; all-family capacity not audited | no underfitting proof | no valid target evidence | Low / unsupported | Unknown | Unknown | Possibly, but not authorized | No | First resolve simpler causes |

## Ranking conclusion

The only high-confidence current cause is failure of confirmatory numerical
validity. Protocol/evaluator differences remain a moderate external-comparison
confound. Seed variance, ranking, domain shift, prompt drift, and DFG behavior
are plausible hypotheses but lack the required target or feature evidence.
Capacity is not supported. This ranking does not authorize a new module.
