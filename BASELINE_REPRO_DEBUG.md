# Baseline Reproduction Debug

This note records why the current `test_train_main_base` run does not match
the ACD-CLIP numbers reported by the upstream repository.

Upstream reference:

```text
repo: https://github.com/upupmake/ACD-CLIP
commit in this clone: 2685a66 update framework
```

## Current Status

The current run is architecturally close to mainbase ACD-CLIP, but the working
tree is not script-identical to upstream anymore. Local patches were added to
make training/evaluation run on this machine:

```text
dataset/info.py  : local data root and medical split fixes
model/adapter.py : checkpointing path and clamp_min eps in adapter norm
train.py         : AMP, grad checkpointing, num_workers, finite guards, full adapter clipping
test.py          : selected epochs, map_location, num_workers, CPU caching, binned streaming metrics
```

Therefore current results should be called a local baseline run, not an exact
upstream reproduction yet.

## Data Audit

The six medical json files and local files were checked.

```text
Brain:
  rows=3715, labels={0:640, 1:3075}, missing image=0, missing mask=0

Liver:
  rows=1493, labels={0:833, 1:660}, missing image=0, missing mask=0

Retina:
  rows=1805, labels={0:1041, 1:764}, missing image=0, missing mask=0

Colon_clinicDB:
  rows=612, labels={1:612}, missing image=0, missing mask=0

Colon_colonDB:
  rows=380, labels={1:380}, missing image=0, missing mask=0

Colon_Kvasir:
  rows=1000, labels={1:1000}, missing image=0, missing mask=0
```

Important fixes already made:

```text
Brain/Liver/Retina DATA_PATH now points to the test split.
Colon_Kvasir CLASS_NAMES/REAL_NAMES now use "Colon_Kvasir" to match json.
```

Mask sanity checks found no obvious empty/full-mask bug:

```text
Brain positive mask nonzero fraction sample mean: 0.0282
Liver positive mask nonzero fraction sample mean: 0.0104
Retina positive mask nonzero fraction sample mean: 0.0934
ClinicDB positive mask nonzero fraction sample mean: 0.0941
ColonDB positive mask nonzero fraction sample mean: 0.0508
Kvasir positive mask nonzero fraction sample mean: 0.1185
```

The medical json checksums match the old phase2/v2-phase2 repos, so metadata is
not obviously different across local repos.

## Current Binned Results

The current baseline was tested with streaming binned metrics:

```text
metric_thresholds=1000
epochs=18,19,20
```

Five completed datasets show epoch 19 as best among 18/19/20:

```text
epoch 18 mean over 5 datasets: pixel AUC/AP = 77.14 / 23.55
epoch 19 mean over 5 datasets: pixel AUC/AP = 79.94 / 24.16
epoch 20 mean over 5 datasets: pixel AUC/AP = 74.87 / 22.45
```

Epoch 19 per-dataset binned results:

```text
Brain:    92.12 / 29.51
Liver:    91.13 / 5.21
Retina:   62.57 / 24.66
ClinicDB: 79.38 / 34.60
ColonDB:  74.52 / 26.81
```

These are far below upstream `ours (N=3)` for several datasets, especially
Retina and AP values.

## Why Phase2 Had No Threshold

Older phase2 test code did not use binned thresholds. Instead, it avoided OOM
by storing predictions on CPU in `half()` and, for datasets with more than 500
images, evaluating only every fourth pixel:

```python
if masks.shape[0] > 500:
    masks_eval = masks[:, :, ::4, ::4]
    preds_eval = preds[:, ::4, ::4]
else:
    masks_eval = masks
    preds_eval = preds
```

That protocol is memory-safe but not exact full-resolution pixel metrics. It
can also differ from the upstream repo, which keeps all masks/preds and calls
`metrics_eval_gpu()` directly.

Phase2 also had optional TTA and dynamic convolution. Its strong logs are not
clean ACD-CLIP mainbase reproduction evidence.

## Checkpoint Evidence

The current mainbase checkpoint is much smaller than old phase2 checkpoints:

```text
current test_train_main_base/adapter_19.pth: ~39 MB
phase2 ckpt_VisA_N3_phase1/adapter_19.pth: ~283 MB
```

State dict inspection:

```text
current checkpoint:
  text_adapter keys=15, numel=80,640
  image_adapter keys=78, numel=9,880,338

phase2 checkpoint:
  text_adapter keys=21, numel=1,852,416
  image_adapter keys=307, numel=69,046,588
  contains dynamic/depthwise/log_tau-style parameters
```

Therefore old phase2 checkpoints cannot be used as a clean mainbase oracle.

## Confirmed Main Cause

The data paths/labels/masks are mostly correct after the Kvasir and MedAD split
fixes. The large Retina gap was reproduced as a metric/protocol issue, not only
as a checkpoint issue.

Decisive diagnostic on the same checkpoint (`adapter_19.pth`) and the same
label-balanced Retina subset (`128` normal + `128` abnormal):

```text
Retina epoch 19, binned thresholds=1000, stride=1:
  pixel AUC/AP = 58.53 / 20.01

Retina epoch 19, no threshold, pixel_stride=4:
  pixel AUC/AP = 88.15 / 42.23
```

This proves that the low Retina number in the current binned evaluation is
mostly caused by metric/test protocol, not by a simple dataset label bug. It
also shows why old phase2 logs can look much better without using thresholds:
phase2 avoided OOM with `pixel_stride=4`, not with binned thresholds.

Remaining suspicious differences still matter for final reproduction:

```text
1. Current training used AMP and had non-finite gradient skips.
2. Current training uses local finite guards and full-adapter clipping, not upstream train.py.
3. Current test has multiple non-upstream modes: binned streaming, CPU caching,
   selected epochs, and pixel_stride.
4. Previous local phase2 numbers used TTA/subsample/dynamic modules, not clean mainbase.
5. There is no clean saved upstream/mainbase checkpoint in ACD-CLIP-main to use as a direct oracle.
```

However, the immediate reason the current test looked far below phase2/newbase
expectations is the metric protocol mismatch: `metric_thresholds=1000` binned
evaluation is not comparable to phase2-style no-threshold stride-4 evaluation
or upstream exact evaluation.

## Recommended Verification Steps

To isolate the cause:

```text
Step 1:
  Run a short mainbase train without AMP, keeping grad_checkpointing if needed
  for memory. This checks whether AMP/skip-grad hurt the checkpoint.

Step 2:
  Test selected epochs with phase2-style pixel_stride=4 exact metrics, not
  binned thresholds. This matches the local protocol that previously avoided
  OOM without binned metrics.

Step 3:
  If no-AMP checkpoint is still poor, compare with N=4 default from upstream
  because README defaults to N=4 even though the table includes N=3.

Step 4:
  Only after a stable local baseline is identified, use exact-memory-safe
  metrics for the final selected epoch.
```

Suggested next baseline command:

```bash
conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --batch_size 6 \
  --epoch 20 \
  --grad_checkpointing \
  --num_workers 6 \
  --save_path test_train_main_base_noamp_n3
```

Do not add Phase 1 architecture changes until this reproduction issue is
understood, otherwise the ablation will be hard to defend.

## Commands For The Next Check

The key check is whether `metric_thresholds=1000` is responsible for the low
Retina score. Run one dataset/epoch with no binned threshold and with the same
pixel subsampling style that phase2 used:

```bash
cd /home/ai4/caohuy/ACD-CLIP-mainbase-newest
METRIC_THRESHOLDS=none PIXEL_STRIDE=4 DATASETS="Retina" \
  bash test_6medical_selected_epochs.sh 19
```

Interpretation:

```text
Observed result:
  binned threshold=1000 stayed low, while no-threshold stride-4 jumped to 88.15 / 42.23.

Conclusion:
  metric protocol is a major contributor, and binned/full-res/stride-4 results
  must not be mixed.
```

After any test run, parse the log with:

```bash
python parse_test_log.py --log test_train_main_base/test.log
```

If the full Retina run is too slow, use a label-balanced subset diagnostic
first. Do not use `MAX_SAMPLES` for Retina because the json is ordered with
many normal images first; a naive first-N subset may contain no abnormal masks.

```bash
cd /home/ai4/caohuy/ACD-CLIP-mainbase-newest

bash debug_retina_protocol.sh 19
```

Equivalent expanded commands:

```bash
cd /home/ai4/caohuy/ACD-CLIP-mainbase-newest

# Current binned protocol on 128 normal + 128 abnormal Retina samples.
MAX_SAMPLES_PER_LABEL=128 METRIC_THRESHOLDS=1000 PIXEL_STRIDE=1 DATASETS="Retina" \
  bash test_6medical_selected_epochs.sh 19

# Phase2-style no-threshold stride-4 protocol on the same balanced sample size.
MAX_SAMPLES_PER_LABEL=128 METRIC_THRESHOLDS=none PIXEL_STRIDE=4 DATASETS="Retina" \
  bash test_6medical_selected_epochs.sh 19

python parse_test_log.py --log test_train_main_base/test.log
```

This subset check is not for reporting. It was only a quick diagnostic, and it
confirmed that the binned threshold protocol was the major source of the
apparent Retina collapse.

If the checkpoint is the likely issue, run a no-AMP baseline reproduction check:

```bash
cd /home/ai4/caohuy/ACD-CLIP-mainbase-newest
bash train_baseline_noamp_n3.sh
```
