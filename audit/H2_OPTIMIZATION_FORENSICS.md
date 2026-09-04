# H2 optimization and training-dynamics forensics

## Scope

This report parses the existing Seed-1 and Seed-2 H/A logs only. No new
forward pass, evaluation, or training was run.

## Hard validity result

| run | nonfinite-loss skips | nonfinite-gradient skips | skip epochs | final successful steps |
|---|---:|---:|---|---:|
| Seed 1 H | 0 | 3 | E2, E8, E9 | 5410 |
| Seed 1 A | 0 | 2 | E2, E10 | 5411 |
| Seed 2 H | 0 | 3 | E2, E6, E7 | 5410 |
| Seed 2 A | 0 | 2 | E2, E3 | 5411 |

The pattern repeats by arm count across two seeds, but the epochs differ.
Every final model checkpoint was finite. The unequal successful optimizer
steps between H and A make both confirmatory seeds invalid for target
comparison. This is strong evidence of a numerical/optimization validity
problem, not evidence that A is worse or that Anchor failed scientifically.

## Source loss trajectories

The log records E2 through E15. The endpoint values are:

| run | total loss E2 -> E15 | classification E2 -> E15 | segmentation E2 -> E15 |
|---|---|---|---|
| Seed 1 H | 1.144242 -> 0.769624 | 0.625694 -> 0.415063 | 0.511863 -> 0.352332 |
| Seed 1 A | 1.193046 -> 0.946489 | 0.675513 -> 0.548566 | 0.511678 -> 0.394892 |
| Seed 2 H | 1.185225 -> 0.762068 | 0.671516 -> 0.411007 | 0.507636 -> 0.348751 |
| Seed 2 A | 1.196296 -> 0.941572 | 0.675474 -> 0.530822 | 0.514895 -> 0.407447 |

Loss declines substantially in all four runs despite the skipped steps. This
shows continued source-side optimization, but there is no source validation
curve in the logs and no valid target curve. A lower training loss therefore
cannot establish better transfer or identify a generalization peak.

## Learning-rate and prompt schedule

The logged schedule is consistent across all runs:

- E2: image `9.0e-4`, text `4.5e-4`, prompt `0`, hybrid alpha `0`.
- E4 onward: prompt is unfrozen and alpha increases through the fixed
  schedule; E15 has image `2.2876792455e-4`, text `1.1438396227e-4`, prompt
  `5.0e-5`, hybrid alpha `.2`.
- StepLR decay is applied after each epoch, matching the frozen contract.

At E15, the raw KG diagnostic is approximately `.220` to `.329` and its
`.01` weighted contribution is approximately `.0022` to `.0033`. The K term
is approximately `.0027` to `.0305` raw and its `.002` weighted contribution
is approximately `5.4e-6` to `6.1e-5`. These weighted magnitudes are much
smaller than the main loss near `.94`; magnitude alone does not show loss
dominance.

## Anchor-family telemetry

The A logs contain family-safe telemetry. At E15:

| run | global effective Anchor/task ratio | global task norm | max active family ratio | notable families |
|---|---:|---:|---:|---|
| Seed 1 A | `5.7083e-6` | 4.0137 | `.1000` | Q `.0575`, K `.0283`, SS2D `.0336` |
| Seed 2 A | `5.5192e-6` | 5.3730 | `.1000` | Q `.0420`, K `.0295`, SS2D `.00052` |

Across the whole run, LoRA and `m_i_W` are almost always classified as
Anchor-negligible. Q/K and, in Seed 1, SS2D contain most active/moderate or
dominant events. The global effective ratio remains around `1e-5`, so there is
no evidence that Anchor dominates the aggregate update. Some individual
family elements reach the deliberately allowed `.10` cap, so the telemetry
does demonstrate localized intervention rather than a globally large Anchor.

## DFG and numerical observations

Final stage weights are finite and sum errors are at most approximately
`1.2e-7`. H selects stage 3 most strongly in both seeds (about `.618` and
`.623` for abnormal weights). A is more variable: Seed 1 stage 3 abnormal
weights are `[.1135,.5965,.2900]`, and Seed 2 are `[.1423,.6130,.2447]`.
This is non-uniform routing, not a collapsed uniform gate. The logs do not
contain target-conditioned DFG traces, spatial entropy, or a known-correctness
label for the selected group, so wrong routing is not demonstrated.

## What is and is not identified

The repeated nonfinite-gradient events are the leading immediate blocker.
However, the archived logs do not contain GradScaler scale histories, Adam
first/second moments, parameter update norms, or a stack-level numerical
operand trace. Consequently the current evidence cannot distinguish AMP
overflow, a particular loss branch, or a specific parameter family's
backward instability. The LR schedule is documented and decays normally, so
`LR_BOTTLENECK=POSSIBLE_BUT_NOT_SUPPORTED`; `OPTIMIZER_BOTTLENECK=POSSIBLE_BUT_NOT_IDENTIFIED`.

`TRAINING_DYNAMICS_BOTTLENECK=NUMERICAL_VALIDITY_BLOCKER_WITH_UNRESOLVED_MECHANISM`.

`TRAINING_HORIZON_DIAGNOSIS` remains secondary Seed-0 evidence only. Medical
E15 to E20 changes are H pixel AUROC `+0.183782` and AP `-0.233990`, while A
changes are AUROC `-0.007880` and AP `+2.907670`. This is metric-specific
horizon sensitivity, not proof of overfitting because Seed-1/Seed-2 target
curves do not exist.
