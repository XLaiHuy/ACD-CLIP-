# Phase2C Condition P_LoRA_only preregistration

Short name: PL

## Condition definition

P_LoRA_only is identical to A-prime in every scientific respect except that
symmetric module-scoped PCGrad is applied only to the `shared_image_lora`
parameter group.

Parent condition: A_prime

## Scientific motivation

Full PCGrad (Condition P) applied projection across four shared parameter
groups: `shared_image_lora`, `m_i_w`, `hard_text_adapter`, and `soft_prompt`.

Condition P increased Pixel AUC by +2.37 pp versus A-prime but decreased
Pixel AP by -3.77 pp and Image AP by -1.62 pp, failing its primary success
criterion.

PCGrad projection rate analysis across diagnostic epochs 10–13 showed that
negative cosine (gradient conflict) was most concentrated in `shared_image_lora`
at 75.0% (9 / 12 diagnostic rows), compared with:

- `m_i_w`: lower conflict rate
- `hard_text_adapter`: moderate but secondary
- `soft_prompt`: moderate but secondary

Hypothesis: projecting gradient conflict on the shared image path while leaving
the text alignment modules to train normally may preserve Image AP and Pixel AP
while still resolving the dominant conflict region.

## PCGrad scope

PCGrad is applied exclusively to:

| Group | PCGrad |
|---|---|
| `shared_image_lora` | ON |
| `m_i_w` | OFF — standard gradients |
| `hard_text_adapter` | OFF — standard gradients |
| `soft_prompt` | OFF — standard gradients |

PCGrad must not be applied to:

- classification-only parameters
- segmentation-only parameters
- segmentation-only DFG/SS2D parameters
- any parameter not listed above as ON

## PCGrad algorithm

The same deterministic symmetric two-task projection from Condition P is used,
applied only to `shared_image_lora`.

For each configured group:

1. Compute FP32 flattened gradients `g_cls` and `g_seg` from the original
   unmodified classification and segmentation losses.
2. Let `dot = g_cls · g_seg`, `eps = 1e-12`.
3. If `dot >= 0`: no projection; final gradient = `g_cls + g_seg + g_other`.
4. If `dot < 0`: apply the symmetric projection:
   ```
   g_cls_proj = g_cls - (dot / (||g_seg||² + eps)) * g_seg
   g_seg_proj = g_seg - (dot / (||g_cls||² + eps)) * g_cls
   final      = g_cls_proj + g_seg_proj + g_other
   ```
5. `g_other` is computed from `total_loss - cls_loss - seg_loss` (no KG/K
   regularization is silently removed).
6. All projection mathematics are in FP32 regardless of AMP/BF16 training
   precision.

No random task ordering is used.

## Fields preserved from A-prime

All scientific fields are identical to A-prime:

| Field | Value |
|---|---|
| Architecture | OpenAI CLIP ViT-L/14-336, resized 518 |
| `hybrid_alpha_max` | 0.20 |
| alpha schedule | `[0, 0, 0, .05, .10, .20, ...]` (freeze_epochs=3) |
| `dfg_beta_schedule` | warmup010 |
| `dfg_beta_target` | 0.10 |
| `soft_prompt_freeze_epochs` | 3 |
| `lambda_kg` | 0.01 |
| `lambda_k` | 0.002 |
| `image_lr` | 0.001 |
| `text_lr` | 0.0005 |
| `soft_prompt_lr` | 5e-5 |
| `lr_gamma` | 0.9 |
| `grad_clip_norm` | 1.0 |
| `batch_size` | 6 |
| `num_workers` | 6 |
| `seed` | 42 |
| `epochs` | 15 |
| `bf16` | True |
| Dataset | VisA, fixed train/val split seed 42 |
| Manifests | `splits/visa_train_seed42.csv`, `splits/visa_val_seed42.csv` |
| Image score rule | `cls_only` |
| Checkpoint selection | existing registered rule (see below) |
| Diagnostic batches | same fixed batch IDs |

## Checkpoint selection rule (unchanged from A-prime / P)

```
Eligibility: image_ap >= (best image_ap in run epochs 1–3 average of top-2) - 1.0
Primary:     maximize pixel_ap
Tie-break:   image_ap descending, then earlier epoch
```

## Success criteria (pre-registered)

### Primary success

Both conditions must hold on the VisA fixed val split at the selected checkpoint:

```
Image AP >= 97.4225   (A-prime e13 Image AP 98.4225 minus 1.0 guardrail)
Pixel AP > 55.5341    (strictly above A-prime e13 Pixel AP)
```

### Secondary / Pareto criteria

If primary success is not achieved, a Pareto result is:

```
Pixel AUC > 94.8038   (above A-prime e13 Pixel AUC)
Pixel AP  >= 55.0341  (no more than 0.5 pp below A-prime)
Image AP  >= 97.4225  (guardrail maintained)
```

A Pareto result means PL is a candidate for a different objective but does not
replace A-prime under the primary rule.

### Failure

P_LoRA_only fails if:

- Pixel AP does not exceed A-prime Pixel AP (55.5341), AND
- Image AP fails the guardrail (< 97.4225)

or

- Pixel AP / Image AP shows a major decline similar to full P (Pixel AP drop
  >= 3.0 pp or Image AP drop >= 1.5 pp relative to A-prime)

**On failure: close the PCGrad branch. Do not create another PCGrad variant
without a new preregistered hypothesis.**

## Exploratory scope of this run

- seed 42 is exploratory until replicated with seeds 41, 42, 43
- medical datasets are held out and must not influence selection
- no multi-seed run is authorized in this task
- authorized runs: VisA seed-42 only, 15 epochs, BF16, local or Kaggle

## Output directory

```
runs/phase2c_bf16/PL_lora_only_seed42/
```

## What must not change relative to A-prime

- original VisA manifests (`splits/visa_train_seed42.csv`, `splits/visa_val_seed42.csv`)
- split metadata (`splits/visa_split_seed42_metadata.json`)
- checkpoint selection rule
- image scoring rule
- augmentations
- loss terms and all regularization
- optimizer type (Adam)
- learning rate schedule
- batch size
- diagnostic batch count and selection seed

## Related artifacts

- Runner: `run_phase2c_PL_pcgrad_lora_only_seed42.sh`
- Training entrypoint: `phase2c_train.py --condition P_LoRA_only`
- PCGrad implementation: `phase2c_pcgrad.py`
- Protocol: `phase2c_utils.py`, `phase2c_pcgrad_diagnostics.py`
