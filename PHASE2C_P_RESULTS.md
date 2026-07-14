# Phase2C condition P: seed-42 result and handoff context

## Status

Condition P (module-scoped deterministic PCGrad) completed on 2026-07-13.
It is an exploratory seed-42 result, **not** the selected Phase2C winner.
The current primary candidate remains BF16 A-prime, checkpoint epoch 13.

## Architecture and protocol

P keeps the BF16 A-prime architecture and training protocol unchanged:

- OpenAI CLIP ViT-L/14-336 backbone, resized to image size 518;
- Conv-LoRA image adaptation, attention DFG with SS2D residual-weight fusion,
  hard text adapter, and four-token hybrid soft prompt;
- VisA fixed split seed 42, 15 epochs, batch size 6, six workers, BF16,
  deterministic sampler, and `cls_only` image score;
- alpha schedule `0, 0, 0, .05, .10, .20, ...` and beta warm-up to `.10`.

P changes only gradient combination on the shared parameter groups
`shared_image_lora`, `m_i_w`, `hard_text_adapter`, and `soft_prompt`.
For a negative classification/segmentation gradient dot product it applies the
deterministic symmetric two-task PCGrad projection in FP32; all other loss
terms and unscoped parameters retain ordinary autograd gradients.  The full
pre-registered rationale and equations are in `PHASE2C_P_PREREGISTRATION.md`.

## Selected-checkpoint comparison

| Condition | Epoch | Pixel AUC | Pixel AP | Image AUC | Image AP |
|---|---:|---:|---:|---:|---:|
| A-prime | 13 | 94.8038 | 55.5341 | 97.9028 | 98.4225 |
| P / PCGrad | 13 | 97.1696 | 51.7660 | 96.3819 | 96.7979 |
| P minus A-prime | — | +2.3658 | -3.7682 | -1.5208 | -1.6246 |

P therefore fails its primary success criterion: selected Pixel AP does not
exceed A-prime while preserving image quality.  It does create a different
Pareto point through higher Pixel AUC, but this trade-off is not enough to
promote it under the registered Pixel-AP-first rule.

The per-run checkpoint rule selected P epoch 13: among checkpoints with
Image AP at least one point below P's early-epoch anchor, maximize Pixel AP,
then Image AP, then choose the earlier epoch.  For cross-condition review,
P's Image AP is also below the common A-prime-derived guardrail of 97.4225
(A-prime e13 Image AP minus one point).

## Diagnostics

PCGrad was active and numerically stable.  Across 45 fixed diagnostic rows
per scoped group, projection occurred whenever the pre-projection cosine was
negative:

| Group | Projection rate |
|---|---:|
| `shared_image_lora` | 18 / 45 (40.0%) |
| `m_i_w` | 5 / 45 (11.1%) |
| `hard_text_adapter` | 20 / 45 (44.4%) |
| `soft_prompt` | 15 / 45 (33.3%) |

At epochs 10--13, the selection region, negative pre-projection cosine was
most frequent for `shared_image_lora` (9 / 12, 75.0%).  This motivated the
narrower follow-up `P_LoRA_only`.

The completed PL run improved Pixel AUC versus A-prime but remained below
A-prime in Pixel AP.  PL also recovered Image AP and Image AUC substantially
relative to full P, but not enough to satisfy the preregistered Pixel-AP-first
rule.  Final decision: close the PCGrad branch.  Do not run multi-seed
robustness, the medical final test, or additional PCGrad variants for P/PL as
currently configured.

## Reproducibility and stored artifacts

The P run was trained from commit `94b4e707f8fe8f4adf80fc212155a6cfa660e97a`.
The committed result data are in `runs/phase2c_bf16/P_pcgrad_seed42/`:

- `config.json`, `split_metadata.json`, and `diagnostic_batches.json`;
- `train.log`, `visa_val_metrics.csv`, and `selection.json`;
- `gradient_diagnostics.csv` and `pcgrad_diagnostics.csv`.

The selected local model weight is
`runs/phase2c_bf16/P_pcgrad_seed42/checkpoints/adapter_13.pth`.  Checkpoints
remain intentionally untracked because `*.pth` is ignored; this handoff
commits the code, protocol, configuration, logs, metrics, and diagnostics,
but not the approximately 846 MB set of 15 checkpoint files.

## Related code

- Runner: `run_phase2c_P_pcgrad_seed42.sh`
- Training entry point: `phase2c_train.py`
- Projection implementation: `phase2c_pcgrad.py`
- Fixed-batch diagnostics: `phase2c_pcgrad_diagnostics.py`
- Condition and selection helpers: `phase2c_utils.py`
