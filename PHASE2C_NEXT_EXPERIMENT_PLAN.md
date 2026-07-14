# Phase2C: Diagnosis-Gated Curriculum Ablation Plan

## Status after BF16 A-prime/B and PCGrad P/PL

Both runs use the same Phase2C architecture.  They differ only in the
maximum hybrid-soft-prompt alpha.

| Candidate | Selected epoch | Alpha max | Pixel AUC | Pixel AP | Image AUC | Image AP | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| A-prime | 13 | 0.20 | 94.8038 | 55.5341 | 97.9028 | 98.4225 | Primary winner under the Pixel-AP-first rule |
| B | 13 | 0.15 | 96.2236 | 55.1342 | 97.8750 | 98.4287 | Pareto competitor; favors Pixel AUROC |

Do **not** describe A-prime as the best architecture.  A-prime and B are the
same architecture; A-prime is the current best training configuration and
checkpoint under the registered selection rule.

PCGrad follow-ups P and PL are now closed:

| Candidate | Selected epoch | PCGrad scope | Pixel AUC | Pixel AP | Image AUC | Image AP | Role |
|---|---:|---|---:|---:|---:|---:|---|
| Full P | 13 | `shared_image_lora`, `m_i_w`, `hard_text_adapter`, `soft_prompt` | 97.1696 | 51.7660 | 96.3819 | 96.7979 | Pixel AUC exploratory result; failed guardrails |
| PL | 15 | `shared_image_lora` only | 96.6840 | 52.7478 | 97.3542 | 97.9956 | Exploratory Pixel-AUC-oriented checkpoint; failed Pixel AP rule |

PL narrowed the PCGrad scope and recovered image metrics compared with full P,
but it remained below A-prime in Pixel AP. The PL run used batch size 8 versus
A-prime batch size 6, so treat it as directional evidence only. No batch-size-6
rerun is planned. The recommended next research direction is gradient/loss
balancing rather than another PCGrad variant.

## Preserve the completed evidence

Keep both run directories immutable:

- `runs/phase2c_bf16/A_alpha020_seed42/`
- `runs/phase2c_bf16/B_alpha015_seed42/`

The required evidence is the selected e13 checkpoint, `selection.json`,
`config.json`, `visa_val_metrics.csv`, `gradient_diagnostics.csv`, diagnostic
batch IDs, code fingerprint, split metadata, and git status captured before
the run.

## Diagnosis before new training

Analyze A-prime versus B with the same fixed VisA split.

1. Gradient diagnostics by epoch and parameter group:
   - classification and segmentation gradient norms;
   - cosine similarity/conflict;
   - focus on alpha/beta activation (epochs 4--6) and selection region
     (epochs 10--13).
2. Per-category delta at the selected checkpoints for all four metrics:
   `B - A-prime` for Pixel AUC/AP and Image AUC/AP.
3. Record whether the observed metric trade-off is broad across categories or
   concentrated in a small number of categories.

### Decision gate

The diagnostic is an input to the experimental branch, not reporting-only
logging.

| Diagnostic finding | Next action |
|---|---|
| No material shared-path conflict or norm imbalance | Continue with curriculum ablations C, D, E. |
| Material shared-path gradient conflict | Run C as the curriculum control if useful; prioritize an added F/shared-freeze intervention. |
| Material gradient-norm imbalance | Prioritize a loss-balancing intervention (for example, GradNorm) rather than automatically completing C--E. |

Predefine quantitative thresholds for “material” before applying the gate;
otherwise report the decision as exploratory rather than confirmatory. The
first A-prime/B screening report is exploratory because its thresholds were
configured after those runs completed.

## Clean curriculum ablations

All candidates must use the same dataset manifests, fixed split, epoch count,
BF16 mode, batch size, workers, diagnostic batches, score rule, checkpoint
selection code, and all non-listed hyperparameters.

| Condition | Change relative to predecessor | `alpha_max` | Purpose |
|---|---|---:|---|
| C | Delayed alpha + beta activation; no optimizer/LR restart | 0.20 | Isolate delayed curriculum activation versus A-prime. |
| D | Exactly C plus one locked optimizer + LR restart package | 0.20 | Isolate restart-package effect. |
| E | Exactly D, changing only alpha maximum | 0.15 | Isolate alpha choice inside the restarted curriculum. |

The intended causal comparisons are:

```text
C vs A-prime  = delayed activation
D vs C        = optimizer + LR restart package
E vs D        = alpha .20 -> .15 within the same curriculum
```

Before implementation, define in the experiment manifest the exact activation
epochs and every restart-package field (optimizer state behavior, learning
rates, scheduler, and restart epoch).  The current Phase2C entrypoint accepts
only `A_prime` and `B`, so C/D/E condition definitions, validation, scripts,
and tests must be added before launching them.

## Selection and comparison policy

### Primary selection rule

Keep the existing registered rule unchanged:

```text
Eligibility: image AP >= (best image AP in that run - 1.0)
Primary:     maximize pixel AP
Tie-break:   image AP descending, then earlier epoch
```

### Secondary decision views

- Pixel-AUROC-first ranking, with the same image-quality guardrail.
- Pareto frontier in `(pixel AP, pixel AUC, image AP, image AUC)` across the
  selected checkpoint from each condition.
- Per-category metrics and gradient diagnostics are explanatory evidence, not
  a replacement for the primary rule.

Do not average the four metrics into an unregistered scalar.  If a weighted
score is explored, publish its weights in advance and label it secondary.

## Robustness and final evaluation

After selecting one candidate configuration, use two separate checks.

1. **Primary training robustness:** run the baseline and selected winner on
   the *same fixed VisA split* with training seeds 41, 42, and 43.  Summarize
   mean, standard deviation, and per-seed selected epoch under the same rule.
2. **Optional split sensitivity:** only after primary robustness, evaluate a
   separately generated VisA split seed (for example 43) or a group/category
   split variant.  Do not change training seed and split seed together in the
   first replicate.
3. Lock the winner configuration and checkpoint-selection policy.
4. Run the medical final test once, without using it for tuning.

## Execution order

```text
Preserve A-prime/B artifacts
        -> gradient + per-category diagnosis
        -> decision gate
        -> C, then D/E paired alpha comparison (or targeted F/GradNorm branch)
        -> primary rule + Pixel-AUC view + Pareto review
        -> choose one configuration
        -> fixed-split training-seed robustness
        -> optional split sensitivity
        -> lock protocol and run medical final test
```
