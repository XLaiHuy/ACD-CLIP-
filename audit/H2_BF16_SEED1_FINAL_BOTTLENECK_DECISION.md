# H2 BF16 Seed1 final bottleneck decision

## Validity and matched result

The global BF16 protocol eliminated the observed FP16 numerical validity blocker in this screen: fresh Seed1 shared E1 and both matched H/A E15 trajectories completed 5,415 successful optimizer steps with zero nonfinite loss/gradient events and finite model and Adam states. This confirms that FP16 dynamic range was a real training-validity bottleneck for the adapted visual path. It does **not** establish a multi-seed BF16 performance claim.

A improves the matched BF16 Seed1 pixel metrics: Medical `+1.7192` AUROC and `+4.0862` AP; MVTec `+0.7639` AUROC and `+1.4057` AP. MVTec image metrics favor H by `1.1775` AUROC and `0.6246` AP. The one-seed result supports an Anchor effect on pixel ranking under BF16, but cannot estimate its variance or generality.

## Candidate bottlenecks

| Candidate | Classification | Evidence |
|---|---|---|
| FP16 numerical precision | CONFIRMED | All preserved failure paths and both fresh E15 BF16 trajectories were finite without GradScaler. |
| BF16 as a quality improvement | NOT_SUPPORTED | Numerical validity improved, but one seed cannot estimate a generalization or performance effect. |
| LR / scheduler / optimizer | NOT_SUPPORTED | Frozen unchanged; no causal comparison was run. |
| Loss composition / weighting / gradient conflict | UNKNOWN | No isolated loss intervention or full gradient-conflict audit was performed. |
| Safe Anchor strength / budget | POSSIBLE | A improves matched pixel metrics; the E15 family budget stayed capped and non-dominant. One seed cannot infer the preferred strength. |
| Conv-LoRA / image-adapter utilization | UNKNOWN | Source telemetry has nonzero image-adapter gradients, but no causal utilization test or target-feature dump exists. |
| Text / prompt branch | UNKNOWN | Source telemetry has nonzero text/prompt gradients, but no branch ablation is authorized or available. |
| DFG / SS2D routing | UNKNOWN | Frozen evaluators did not retain target routing statistics; no mechanism evidence supports a change. |
| Architecture capacity | NOT_SUPPORTED | Valid BF16 training and A’s positive pixel deltas provide no direct capacity mechanism evidence. |
| Pixel ranking / localization / calibration | UNKNOWN | Raw pixel score maps and joins were not retained. Medical image-score distributions cannot explain pixel AP. |
| Cross-domain generalization / initialization sensitivity | LIKELY | Valid BF16 Seed1 metrics differ materially from historical single-seed outputs, confounded by precision, seed, and evaluator history. |

`PRIMARY_BOTTLENECK=CROSS_SEED_GENERALIZATION_AND_INITIALIZATION_UNCERTAINTY_AFTER_NUMERICAL_VALIDITY_REPAIR`

`SECONDARY_BOTTLENECK=PIXEL_RANKING_LOCALIZATION_MECHANISM_UNOBSERVED`

## Recommended next research actions

1. `REPLICATE_MORE_SEEDS`: run a preregistered fresh BF16 Seed2 matched H/A source trajectory, then freeze both checkpoints before any target access. This tests the primary uncertainty without choosing settings from targets.
2. `SOURCE_ONLY_OBSERVABILITY_AUDIT`: before another target run, add fixed, noninterventional retention of pixel score maps, DFG routing summaries, and prototype/feature statistics to the evaluator contract. Validate it on source data first; do not use it for target selection.
3. `OPTIMIZATION_AUDIT`: only if a new valid BF16 seed still diverges strongly, compare source-side update/drift and Anchor-budget telemetry across valid H/A runs. Do not alter LR, loss, optimizer, or architecture first.

`RECOMMENDED_NEXT_ACTION=REPLICATE_MORE_SEEDS`

## Result-boosting review

| Idea | Classification | Rationale |
|---|---|---|
| EMA / SWA / checkpoint averaging | REQUIRES_NEW_EXPERIMENT | Changes training/evaluation selection semantics and needs preregistration. |
| LR schedule or layer-wise LR | SOURCE_SIDE_SCIENTIFIC_CHANGE | No causal evidence; requires a fresh controlled run. |
| Loss reweighting / gradient-conflict handling | SOURCE_SIDE_SCIENTIFIC_CHANGE | No isolated diagnosis supports it. |
| Prompt ensembling / source augmentation / DFG change | REQUIRES_NEW_EXPERIMENT | Scientific interventions, not evaluation fixes. |
| Target threshold, smoothing, prompt, Anchor lambda, seed, or epoch choice | TARGET_LEAKAGE_PROHIBITED | Selecting after Medical/MVTec outcomes invalidates frozen evaluation. |
| Architecture redesign | NOT_RECOMMENDED | Numerical validity, seed replication, and source observability explanations remain unfalsified. |

No new experiment was started from this decision.
