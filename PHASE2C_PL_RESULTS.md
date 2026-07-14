# Phase2C condition PL: P_LoRA_only seed-42 result

## Status

P_LoRA_only completed on 2026-07-14 from the RTX 4090 run
`runs/phase2c_4090/PL_lora_only_seed42_bs8/`.

PL is retained only as an exploratory Pixel-AUC-oriented checkpoint. It does
not replace A-prime under the preregistered Pixel-AP-first rule, and the PCGrad
branch is closed. Do not create further PCGrad variants without a new
preregistered hypothesis.

## Architecture and protocol

PL is A-prime plus deterministic symmetric module-scoped PCGrad only on
`shared_image_lora`.

The following modules used normal autograd gradients:

- `m_i_w`
- `hard_text_adapter`
- `soft_prompt`
- task-specific heads
- segmentation-only DFG/SS2D parameters

The run used VisA fixed split seed 42, selected epoch 15, batch size 8,
`num_workers` 10, BF16 precision, and `cls_only` image scoring. Batch size 8
differs from A-prime batch size 6, so this is an exploratory directional
comparison, not a perfectly controlled ablation. No batch-size-6 rerun is
planned.

## Selected-checkpoint comparison

| Condition | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP |
|---|---:|---:|---:|---:|---:|
| A-prime | 13 | 94.8038 | 55.5341 | 97.9028 | 98.4225 |
| PL / P_LoRA_only | 15 | 96.6840 | 52.7478 | 97.3542 | 97.9956 |
| PL minus A-prime | - | +1.8802 | -2.7863 | -0.5486 | -0.4269 |

Interpretation:

- PL passes the Image AP guardrail of 97.4225.
- PL fails the primary criterion because Pixel AP does not exceed 55.5341.
- PL fails the secondary Pareto criterion because Pixel AP is below 55.0341.
- PL improves Pixel AUC but does not beat A-prime under the preregistered
  Pixel-AP-first rule.
- A-prime remains the primary winner.

### Guardrail definitions

There are two distinct constraints. Run-internal checkpoint eligibility is
derived from the early-epoch image-AP anchor inside this PL run; that is the
registered rule used by `selection.json`. The cross-condition PL success
guardrail is A-prime Image AP minus 1.0, equal to 97.4225.

`selection.json` remains unchanged as a historical run artifact. Epoch 15 was
selected by the registered internal rule and also satisfies the cross-condition
Image AP guardrail of 97.4225. PL still fails because Pixel AP does not beat
A-prime and does not satisfy the secondary Pixel AP threshold.

## A-prime, full P, and PL

| Condition | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP |
|---|---:|---:|---:|---:|---:|
| A-prime | 13 | 94.8038 | 55.5341 | 97.9028 | 98.4225 |
| Full P | 13 | 97.1696 | 51.7660 | 96.3819 | 96.7979 |
| PL | 15 | 96.6840 | 52.7478 | 97.3542 | 97.9956 |

PL recovers Image AP and Image AUC substantially compared with full P, but it
remains below A-prime in Pixel AP. Narrowing PCGrad scope reduced damage but
did not solve the localization AP regression.

## Reproducibility and stored artifacts

Run source:

- Branch at run time: `phase2c-pl-kaggle`
- Training commit SHA: `9173e67a7a095800a417859565ef3e479b799c0a`
- Run directory: `runs/phase2c_4090/PL_lora_only_seed42_bs8/`

Committed compact artifacts:

- `config.json`
- `split_metadata.json`
- `diagnostic_batches.json`
- `train.log`
- `visa_val_metrics.csv`
- `selection.json`
- `gradient_diagnostics.csv`
- `pcgrad_diagnostics.csv`

Selected local checkpoint:

- `runs/phase2c_4090/PL_lora_only_seed42_bs8/checkpoints/adapter_15.pth`
- Size: 54M
- SHA-256: `1da52b88cd8009ad377e6c82377c7957cc1ac6df687596a800299bf54eab04f4`

Only the selected epoch-15 checkpoint should be retained in canonical Git LFS.
Epochs 1-14, smoke-run checkpoints, prediction dumps, rendered masks, caches,
dataset files, and unrelated Kaggle outputs must stay untracked.

## Next direction

The next research direction should be gradient/loss balancing rather than
another PCGrad variant. Candidate work should be preregistered before
implementation and should avoid using medical data for tuning.
