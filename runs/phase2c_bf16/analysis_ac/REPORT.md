# Phase2C BF16 A-prime/C delayed-activation diagnostic

Selected epochs: A-prime e13; C e14.

## Macro delta (C - A-prime)

| Pixel AUC | Pixel AP | Image AUC | Image AP |
|---:|---:|---:|---:|
| 1.3090 | -0.7988 | -0.7083 | -0.7746 |

## Activation-relative diagnostic gate

- Shared-image-LoRA activation conflict reduced: `True`.
- C has any conflict flag: `True`.
- C has any norm-imbalance flag: `True`.
- Recommended branch: `TARGETED_CONFLICT_INTERVENTION`.

Exploratory screening only. Lock D's restart epoch, optimizer-state policy, learning rates, and scheduler before training.
