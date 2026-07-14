# Phase2C: Diagnosis-Gated Curriculum Ablation Plan

## Status after BF16 A-prime/B, curriculum C, and PCGrad P/PL

### Completed C: delayed alpha/beta activation

C used A-prime as its parent and delayed alpha/beta activation. Its selected
epoch was 14, with the following differences from A-prime:

| Metric | C minus A-prime |
|---|---:|
| Pixel AUC | +1.3090 |
| Pixel AP | -0.7988 |
| Image AUC | -0.7083 |
| Image AP | -0.7746 |

C did not replace A-prime. The intervention reduced conflict in one activation
region, but material conflict and gradient-norm imbalance remained. The
diagnostic decision gate redirected the investigation toward targeted
interventions rather than automatically launching D.

Both runs use the same Phase2C architecture.  They differ only in the
maximum hybrid-soft-prompt alpha.

| Candidate | Selected epoch | Alpha max | Pixel AUC | Pixel AP | Image AUC | Image AP | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| A-prime | 13 | 0.20 | 94.8038 | 55.5341 | 97.9028 | 98.4225 | Primary winner under the Pixel-AP-first rule |
| B | 13 | 0.15 | 96.2236 | 55.1342 | 97.8750 | 98.4287 | Pareto competitor; favors Pixel AUROC |

Do **not** describe A-prime as the best architecture.  A-prime and B are the
same architecture; A-prime is the current best training configuration and
checkpoint under the registered selection rule.

P and P_LoRA_only are completed, and the PCGrad branch is now closed:

| Candidate | Selected epoch | PCGrad scope | Pixel AUC | Pixel AP | Image AUC | Image AP | Role |
|---|---:|---|---:|---:|---:|---:|---|
| Full P | 13 | `shared_image_lora`, `m_i_w`, `hard_text_adapter`, `soft_prompt` | 97.1696 | 51.7660 | 96.3819 | 96.7979 | Pixel AUC exploratory result; failed guardrails |
| PL | 15 | `shared_image_lora` only | 96.6840 | 52.7478 | 97.3542 | 97.9956 | Exploratory Pixel-AUC-oriented checkpoint; failed Pixel AP rule |

Full P and PL increased Pixel AUC but reduced Pixel AP. PL recovered image
metrics compared with full P, but still failed the preregistered Pixel-AP-first
rule. The PL run used batch size 8 versus A-prime batch size 6, so treat it as
directional evidence only. A-prime remains the primary winner. No additional
PCGrad variant should be launched without a new preregistered hypothesis.

### D/E status: deferred after diagnostic gate

D and E remain valid causal curriculum/restart experiments. They are deferred
after the diagnostic decision gate because they do not directly target the
observed gradient conflict and gradient-norm imbalance. They may be revisited
if the research objective becomes a complete study of delayed curriculum and
optimizer restart. They are not the current highest-priority path for improving
Pixel AP with limited compute.

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

### Decision gate (completed)

The diagnostic is an input to the experimental branch, not reporting-only
logging.

| Diagnostic finding | Action taken |
|---|---|
| No material shared-path conflict or norm imbalance | Not applicable; material imbalance was observed. |
| Material shared-path gradient conflict | C completed as the curriculum control; D/E deferred. |
| Material gradient-norm imbalance | Prioritize targeted loss balancing rather than automatically completing D/E. |

Predefine quantitative thresholds for “material” before applying the gate;
otherwise report the decision as exploratory rather than confirmatory. The
first A-prime/B screening report is exploratory because its thresholds were
configured after those runs completed.

## Clean curriculum ablations (D/E deferred after diagnostic gate)

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

## Completed A-prime/B checkpoint interpolation

The locked checkpoint-interpolation experiment is complete. Parent
reproduction passed, and the three registered candidates were evaluated:

| Candidate | Pixel AUC | Pixel AP | Image AUC | Image AP |
|---|---:|---:|---:|---:|
| AB25 | 95.5693 | 55.4652 | 97.9653 | 98.4647 |
| AB50 | 96.0225 | 55.4166 | 97.8611 | 98.3772 |
| AB75 | 96.1600 | 55.2958 | 97.9792 | 98.5016 |

No interpolation candidate exceeded A-prime Pixel AP 55.5341. All three
candidates met the secondary Pareto rule. A-prime remains the primary winner,
and checkpoint interpolation is closed. No interpolated checkpoint was
promoted to canonical Git LFS storage.

## Active next-step plan

The only active next experiment is one static loss-balancing condition,
LB_0p1:

- A-prime with `cls_loss_weight = 0.1`
- `seg_loss_weight = 1.0`
- existing regularizers unchanged
- PCGrad disabled

The primary success rule remains Image AP >= 97.4225 and Pixel AP > 55.5341.
The secondary view is Pixel AUC > 94.8038, Pixel AP >= 55.0341, and Image AP
>= 97.4225. D/E remain explicitly **Deferred after diagnostic gate**.
Medical evaluation remains held out.

## Execution order

```text
Preserve A-prime/B artifacts
        -> closed A-prime/B checkpoint interpolation
        -> LB_0p1 static loss balancing
        -> primary rule + Pixel-AUC view + Pareto review
        -> choose one configuration
        -> fixed-split training-seed robustness
        -> optional split sensitivity
        -> lock protocol and run medical final test
```
