# Phase2C Condition C preregistration

## Purpose

Test whether delaying the joint alpha/beta activation avoids the exploratory
shared-image-LoRA conflict observed at A-prime epoch 6. This is a single-factor
curriculum ablation against A-prime; it is not an optimizer restart, loss
balancing, shared-freeze, or architecture change.

## Locked configuration

Condition C equals BF16 A-prime except for a two-epoch activation delay.

| Field | A-prime | C |
|---|---:|---:|
| `hybrid_alpha_max` | 0.20 | 0.20 |
| alpha/soft-prompt freeze epochs | 3 | 5 |
| alpha schedule, epochs 1--9 | 0, 0, 0, .05, .10, .20, .20, .20, .20 | 0, 0, 0, 0, 0, .05, .10, .20, .20 |
| beta schedule, epochs 1--9 | 0, 0, 0, .05, .05, .05, .10, .10, .10 | 0, 0, 0, 0, 0, .05, .05, .05, .10 |

All other architecture, losses, optimizer state, scheduler, split, seed,
batching, diagnostics, epoch count, BF16 mode, score rule, and selection rule
are unchanged.

## Comparison protocol

Compare A-prime e13 to C's selected checkpoint using the existing VisA primary
rule, the four-metric Pareto view, and per-category deltas. For gradient
analysis align activation-relative epochs: A-prime epochs 4--6 correspond to
C epochs 6--8, and A-prime epochs 7--13 correspond to C epochs 9--15.

## Decision after C

- Activation spike disappears: implement D (optimizer/LR restart package).
- Spike only shifts in activation-relative time: prioritize loss balancing or
  GradNorm.
- Conflict persists: prioritize F/shared freeze or PCGrad/ProGrad.
- Do not run E (`alpha_max=.15`) unless the delayed curriculum gives a
  favorable signal.
