# Phase2D LB_0p1 Preregistration

## Scientific question

Can reducing classification-loss influence improve pixel-level localization
and Pixel AP while preserving acceptable image-level performance?

## Motivation

A-prime remains the highest-Pixel-AP checkpoint. B improved Pixel AUC but
reduced Pixel AP. AB25, AB50, and AB75 all improved Pixel AUC relative to
A-prime, but none exceeded A-prime Pixel AP. C reduced part of the
shared-image-LoRA activation issue, while gradient conflict and norm imbalance
remained. P and P_LoRA_only did not improve the primary metric. PCGrad is
closed. This experiment directly reduces classification-loss influence without
adding another gradient algorithm.

## Locked intervention

Exactly one coefficient changes:

```text
total_loss =
    0.1 * cls_loss
    + 1.0 * seg_loss
    + existing A-prime regularizers
```

The classification-loss coefficient is locked at `0.1`; no other coefficient
is tested. The segmentation-loss coefficient remains `1.0`.

## Reconstructed A-prime protocol

| Field | Historical A-prime | LB_0p1 | Identical | Reason |
| --- | --- | --- | --- | --- |
| Dataset | VisA | VisA | true | Same dataset only |
| Train manifest | `splits/visa_train_seed42.csv` | same | true | Fixed split |
| Validation manifest | `splits/visa_val_seed42.csv` | same | true | Fixed split |
| Split metadata | `splits/visa_split_seed42_metadata.json` | same | true | Fixed split |
| Seed | 42 | 42 | true | Single registered seed |
| Initialization | Common pretrained OpenAI ViT-L/14-336 CLIP base | same | true | A-prime did not resume e13 |
| Architecture | ACDCLIP, ViT-L-14-336, image LoRA rank 8, text LoRA rank 16, DFG/SS2D enabled | same | true | No architecture change |
| LoRA targets | image adapter and text adapter as built by `build_model` | same | true | No target change |
| Image size | 518 | 518 | true | Historical config |
| Epochs | 15 | 15 | true | Historical config |
| Batch size | 6 | 6 | true | Historical config |
| Workers | 6 | 6 | true | Historical config |
| Precision | BF16 autocast | BF16 autocast | true | Native BF16 protocol |
| Optimizer | Adam, text/image/soft-prompt parameter groups | same | true | No optimizer change |
| Learning rates | text `0.0005`, image `0.001`, soft prompt `0.00005` | same | true | No LR change |
| Scheduler | StepLR, step size 1, gamma `0.9` | same | true | No scheduler change |
| Alpha schedule | alpha max `0.20`, freeze 3 epochs, `[0,0,0,0.05,0.10,0.20...]` | same | true | No curriculum change |
| Beta/gamma | DFG beta `0.10`, warmup010 to `0.10`, gamma max `0.20` | same | true | No DFG change |
| Regularizers | `lambda_kg=0.01`, `lambda_k=0.002` | same | true | Existing regularizers unchanged |
| PCGrad | disabled | disabled | true | PCGrad branch closed |
| Score rule | `cls_only` | `cls_only` | true | Same validation rule |
| Validation | existing `validate_visa`, same selection implementation | same | true | No new score |
| Classification-loss weight | implicit `1.0` | explicit `0.1` | false | Single intended difference |
| Segmentation-loss weight | implicit `1.0` | explicit `1.0` | true | Explicitly preserved |

LB_0p1 trains from the common pretrained base. It does not initialize from
the selected A-prime epoch-13 checkpoint.

## Primary success criterion

The selected LB_0p1 checkpoint must satisfy both:

- Pixel AP > 55.5341
- Image AP >= 97.4225

## Decision rules

### Decision A: primary success

If Pixel AP > 55.5341 and Image AP >= 97.4225, LB_0p1 becomes the new primary
candidate. Preserve only its selected checkpoint through Git LFS and proceed
to multi-seed confirmation. Do not run LB_0p3.

### Decision B: signal only

If Pixel AP > 55.5341 and Image AP < 97.4225, A-prime remains primary. Do not
commit a checkpoint by default. LB_0p3 may only be considered in a separate
future preregistration; it is not implemented here.

### Decision C: no Pixel AP improvement

If Pixel AP <= 55.5341, A-prime remains primary and static loss balancing is
closed. Do not promote the LB checkpoint or run LB_0p3.

## Reporting requirements

Report selected epoch, all four macro metrics, deltas versus A-prime,
per-category metrics, raw and weighted classification/segmentation losses,
regularizer terms, total loss, learning rates, available gradient diagnostics,
and the exact protocol difference. Distinguish within-run checkpoint
selection from the cross-condition comparison against A-prime. One seed does
not establish statistical significance.
