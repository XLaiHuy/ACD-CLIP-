# H2 diagnostic completion map V2

The map records the first incomplete item at the start of this recovery and
the evidence that now closes it. `COMPLETE_VALID` means the compact artifact
has been parsed and its scope supports the stated conclusion; it does not
turn unknown target-side observables into measured facts.

| Phase | Classification | Evidence |
|---:|---|---|
| 0 baseline verification | COMPLETE_VALID | `audit/H2_EXACT_LOCAL_IDENTITY.json`; checkpoint and CLIP SHA256 checks |
| 1 gap reconstruction | COMPLETE_VALID | `audit/H2_GENERALIZATION_GAP_RECONSTRUCTION.md` |
| 2 pixel ranking/localization/morphology | COMPLETE_VALID | `audit/H2_MIXED_SOURCE_PIXEL_RANKING.csv`; final mixed bottleneck report |
| 3 DFG/SS2D routing | COMPLETE_VALID | `audit/H2_MIXED_DFG_ROUTING.csv`; `H2_MIXED_DFG_STABILITY.csv` |
| 4 feature geometry | COMPLETE_VALID | `audit/H2_MIXED_FEATURE_GEOMETRY.csv` |
| 5 parameter-family utilization | COMPLETE_VALID | `audit/H2_MIXED_ADAPTER_UTILIZATION.csv` |
| 6 optimization/LR dynamics | COMPLETE_VALID | `audit/H2_OPTIMIZATION_FORENSICS.md`; checkpoint optimizer metadata |
| 7 loss-gradient conflict | COMPLETE_VALID | `audit/H2_MIXED_LOSS_GRADIENT_CONFLICT.csv/json` |
| 8 prompt/text geometry | COMPLETE_VALID | `audit/H2_MIXED_PROMPT_GEOMETRY.csv`; `H2_MIXED_SEMANTIC_GEOMETRY.csv` |
| 9 fixed inference-only sensitivity | COMPLETE_VALID | frozen diagnostic protocol and local beta/tau/stage-weight results in `H2_MIXED_GENERALIZATION_BOTTLENECK_FINAL.md` |
| 10 frozen post-hoc source analysis | COMPLETE_VALID | `audit/H2_MIXED_GENERALIZATION_BOTTLENECK_FINAL.json`; all 2,162 source images and 48 cohorts |
| 11 master causal scorecard | COMPLETE_VALID | `audit/H2_MIXED_GENERALIZATION_BOTTLENECK_FINAL.md/json` |
| 12 primary/secondary/tertiary assignment | COMPLETE_VALID | `audit/H2_GENERALIZATION_BOTTLENECK_V2_DECISION.md/json` |
| 13 targeted mechanism research | COMPLETE_VALID | frozen Top-3 table in `H2_GENERALIZATION_BOTTLENECK_V2_DECISION.md` |
| 14 simplicity filtering | COMPLETE_VALID | Top-3 rationale and zero-inference-overhead selection |
| 15 mathematical/adversarial red-team | COMPLETE_VALID | `audit/H2_DFG_PROMPT_FORENSICS.md`; `H2_LOSS_GRADIENT_FORENSICS.md`; numerical validity evidence |
| 16 Top-K <= 3 | COMPLETE_VALID | exactly three candidates, exactly one selected in V2 decision |
| 17 cheap fixed source counterfactual | COMPLETE_VALID | `audit/H2_FUNC_ANCHOR_R1_SOURCE_EVAL.json/csv`; 96 fixed clean VisA images |
| 18 final selected direction | COMPLETE_VALID | `results/H2_MASTER_RESEARCH_DECISION_V2.md/json`; R1 mechanism marked FAIL |

## Resumption record

The first incomplete action at recovery start was the preregistered R1
bounded mechanism screen after valid preflight/calibration. It was executed
once. The control and candidate completed, endpoint source evaluation was
performed, and the final decision is now durable locally. No further
diagnostic phase or candidate screen is authorized without new evidence.

The external large-array root referenced by the historical frozen-state JSON
is absent, so raw-array reinspection remains a Tier-3 limitation even though
the compact diagnostic map is complete.

