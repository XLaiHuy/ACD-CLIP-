# Next Experiment Specification

This file records exactly two fresh training runs. No run was started on the
source machine during migration.

## Frozen result remains separate

The frozen research source of truth is commit `aa77c76` and the decision is
KEEP canonical H2 A15 with locked TTA-F. Its final target metrics remain:

- Medical Pixel AP: 40.2459 percent
- MVTec Pixel AP: 47.2975 percent

Neither fresh run changes, replaces, or reinterprets that result.

## Shared configuration for Run V and Run M

Both runs use the current frozen ACD-CLIP++ implementation and the following
scientific settings:

- model: `ViT-L-14-336`, input size 518;
- three adapter groups;
- Dual-Branch DFG: attention mode, dimension 256, temperature 8.0, SS2D
  residual query branch, weight-residual fusion;
- beta schedule `warmup010`, beta target 0.10;
- Conv-LoRA rank 8, alpha 2.0, kernels 3 and 5;
- LoRA rank 16, alpha 2.0;
- image adapter weight 0.2 and text adapter weight 0.2;
- hybrid soft prompt, maximum hybrid alpha 0.2, freeze epochs 3, context
  length 4, phrase initialization `a photo of a`;
- `lambda_kg=0.01`, `lambda_k=0.002`;
- Adam with image learning rate 0.001, text learning rate 0.0005, and StepLR
  gamma 0.9;
- batch size 6, seed 0, AMP, gradient checkpointing, deterministic algorithms,
  gradient clipping 1.0;
- Family-Safe Anchor enabled with lambda
  `0.0021633926715180626`, family budget 0.10, and the frozen shared E1
  reference checkpoint;
- CIR disabled;
- hard-background ranking disabled;
- no unfinished R3 novelty mechanism enabled;
- training horizon 20 epochs with `adapter_1.pth` through `adapter_20.pth`
  saved.

The launcher scripts encode these values. Neither launcher uses `--resume`.
The shared E1 checkpoint is a fixed Anchor reference, not a training resume
state for either fresh run.

## Run V - fresh VisA source

Source dataset: `VisA`.

After training, evaluate every E1 through E20 on the six Medical datasets:
Brain, Liver, Retina, Colon ClinicDB, ColonDB, and Kvasir. Use identity/no TTA,
the `benchmark_exact` evaluator, and `pixel_stride=1`.

Select the target checkpoint as the epoch with the highest Medical macro Pixel
AP. Break ties by higher Medical macro Pixel AUROC, then by earlier epoch.
Record every epoch in the ranking CSV and JSON, not only the winner.

Run command, to be executed only on the laboratory machine after bootstrap:

```bash
RUN_ROOT="$PWD/runs/fresh_visa_source_20260910" \
  bash handoff/lab_20260910/run_fresh_visa_source.sh
```

## Run M - fresh MVTec-AD-all source

Source dataset: `MVTec_all_supervised`.

This source manifest contains all available MVTec-AD `train/good`, `test/good`,
and labelled `test/<anomaly_type>` samples. Test anomalies use the official
ground-truth masks. It contains 5,354 images: 4,096 normal and 1,258
anomalous.

Run M uses the same architecture, hyperparameters, seed, optimizer, scheduler,
image size, horizon, checkpoint policy, Family-Safe implementation, and
Medical evaluation as Run V. Only the source dataset changes.

This is supervised MVTec-AD-all source training followed by Medical transfer.
MVTec itself must not later be described as an untouched target for Run M.

Run command, to be executed only on the laboratory machine after bootstrap:

```bash
RUN_ROOT="$PWD/runs/fresh_mvtec_all_source_20260910" \
  bash handoff/lab_20260910/run_fresh_mvtec_all_source.sh
```

## Evaluation and ranking commands

After either fresh run completes and all 20 adapter checkpoints exist:

```bash
RUN_ROOT="$PWD/runs/fresh_visa_source_20260910"
EVAL_ROOT="$PWD/results/fresh_visa_source_20260910/medical_epochs"
bash handoff/lab_20260910/run_medical_epoch_evaluation.sh \
  --run-root "$RUN_ROOT" --output-root "$EVAL_ROOT"
python handoff/lab_20260910/rank_medical_epochs.py \
  --results-root "$EVAL_ROOT" \
  --output-csv "$PWD/results/fresh_visa_source_20260910/MEDICAL_EPOCH_RANKING.csv" \
  --output-json "$PWD/results/fresh_visa_source_20260910/MEDICAL_EPOCH_RANKING.json"
```

Use the analogous `fresh_mvtec_all_source_20260910` paths for Run M. The
evaluation helper uses no TTA and does not select a checkpoint during testing;
the ranking helper applies the pre-recorded target-selected exploratory rule.

## Scientific label

Because Medical metrics are used to select the winning epoch, Run V and Run M
are **TARGET-SELECTED EXPLORATORY TRANSFER EXPERIMENTS**. They are not untouched
target confirmation experiments. The clean frozen `aa77c76` result remains
separately reported.
