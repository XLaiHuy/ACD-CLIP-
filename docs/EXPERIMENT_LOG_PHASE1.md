# Phase 1 Experiment Log

This file is the running log for Phase 1 experiments. Append a new version
section whenever architecture, hyperparameters, train protocol, or eval protocol
changes.

## Fixed Local Protocol

Unless stated otherwise:

```text
base repo: /home/ai4/caohuy/ACD-CLIP-base-new-phase1
train dataset: VisA
test datasets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir
model: ViT-L-14-336
image size: 518
n_groups: 3
batch_size train: 6
batch_size test: 8
epoch: 20
amp: enabled
grad_checkpointing: enabled
num_workers: 6
optimizer: Adam
text_lr: 0.0005
image_lr: 0.001
lr scheduler: StepLR(step_size=1, gamma=0.9)
image_adapt_weight: 0.2
text_adapt_weight: 0.2
lora_rank: 16
lora_alpha: 2.0
conv_lora_rank: 8
conv_lora_alpha: 2.0
conv_kernel_size_list: [3, 5]
eval metric_thresholds: none
eval pixel_stride: 4
TTA: no
```

Important note:

```text
These are local reproduced results. They should be compared against the local
baseline under the same protocol, not directly against the README/paper table.
```

Metric reporting follows the paper split:

```text
Medical pixel-level: ColonDB, ClinicDB, Kvasir, BrainMRI, Liver CT, Retina OCT
Medical image-level: BrainMRI, Liver CT, Retina OCT
Colon datasets are not included in image-level means because their image AUC/AP
entries are 0.00 in the current test protocol/log format.
Use parse_test_log.py --paper-summary to print this split.
```

## Version Naming

```text
A0: local reproduced mainbase baseline
V1: Phase 1A pure dual-softmax attention DFG, d_attn=256, tau=4
V2: Phase 1A pure dual-softmax attention DFG, d_attn=256, tau=8
V3: Phase 1B attention DFG tau=8 + SS2D-style residual query, gamma_max=0.2
V4: Hybrid MLP+attention rescue if V3 SS2D-style branch fails
```

Folder/script naming:

```text
A0 folder: /home/ai4/caohuy/ACD-CLIP-mainbase-newest/test_train_main_base_eval_stride4_e19
V1 folder: phase1_v1_attn_tau4
V1 train:  train_phase1_v1_attn_tau4_n3.sh
V2 folder: phase1_v2_attn_tau8
V2 train:  train_phase1_v2_attn_tau8_n3.sh
V2 test:   test_phase1_v2_attn_tau8_selected_epochs.sh
V3 folder: phase1_v3_attn_tau8_ss2d_g02
V3 train:  train_phase1_v3_attn_tau8_ss2d_g02_n3.sh
V3 test:   test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh
```

## A0: Local Mainbase Baseline

Purpose:

```text
Reproduce ACD-CLIP mainbase locally as the fair comparison anchor.
```

Architecture:

```text
DFG: original MLP gating
DFG visual input: GAP over segmentation patch tokens, not CLS/det tokens
DFG MLP per visual stage:
  Linear(768 -> 256)
  GELU
  Linear(256 -> n_groups * 2)
  reshape to [B, n_groups, 2]
  softmax over n_groups separately for normal/abnormal
Text adapter: original LoRA
Image adapter: original Conv-LoRA
Segmentation path:
  seg_tokens [B, 1369, 768]
  img_feat = 10 * seg_tokens
  weighted text descriptors [B, 768, 2]
  matmul -> [B, 1369, 2]
  reshape -> [B, 2, 37, 37]
  interpolate to 518
SS2D: none
DoRA: none
dynamic/depthwise Conv-LoRA: none
few-shot branch: none
```

Train command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-mainbase-newest
conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --batch_size 6 \
  --epoch 20 \
  --amp \
  --grad_checkpointing \
  --num_workers 6 \
  --save_path test_train_main_base
```

Eval:

```text
checkpoint: adapter_19.pth
save_path: test_train_main_base_eval_stride4_e19
metric_thresholds: none
pixel_stride: 4
```

Results:

```text
Dataset          Pixel AUC   Pixel AP   Image AUC   Image AP
Brain              93.59      29.98       82.55      94.13
Liver              94.90       5.46       54.75      46.82
Retina             90.17      46.89       77.80      77.01
Colon_clinicDB     82.86      36.51        0.00       0.00
Colon_colonDB      84.32      31.64        0.00       0.00
Colon_Kvasir       85.21      53.84        0.00       0.00
Mean               88.51      34.05
```

Paper-style means:

```text
Medical pixel-level mean, 6 datasets: 88.51 / 34.05
Medical image-level mean, 3 datasets: 71.70 / 72.65
```

Decision:

```text
Use A0 as the local baseline for all Phase 1 ablations.
```

## V1: Phase 1A Pure Attention DFG, tau=4

Purpose:

```text
Replace original MLP DFG with dual-softmax visual-text attention.
```

Architecture:

```text
dfg_mode: attn
W_Q: Linear(768 -> 256), bias=False
W_K: Linear(768 -> 256), bias=False
W_V: none
normal/abnormal attention: separate softmax
values: original CLIP text features
T_final_N/A: normalized after weighted sum
img_feat scale: keep original img_feat = 10 * img_feat
SS2D: none
hybrid/residual MLP path: none
DFG visual input: GAP over segmentation patch tokens, same as A0
Attention per visual stage:
  v_global = mean(seg_tokens, dim=1) -> [B, 768]
  Q = W_Q(v_global) -> [B, 256]
  T_N/T_A = text features -> [B, n_groups, 768]
  K_N/K_A = W_K(T_N/T_A) -> [B, n_groups, 256]
  scores = einsum("bd,bnd->bn", Q, K) / sqrt(256) / tau
  separate softmax for normal and abnormal
  T_final_N/A = weighted sum of original text features
  normalize T_final_N/A
```

Hyperparameters:

```text
dfg_attn_dim: 256
dfg_attn_tau: 4.0
W_Q/W_K init: Xavier uniform because dfg_attn_dim=256
image_lr for W_Q/W_K: 0.001, same image_adapter group
trainable parameters: 10,545,408
baseline trainable parameters: 9,960,978
extra trainable parameters: +584,430
checkpoint folder: phase1_v1_attn_tau4
```

Train command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash train_phase1_v1_attn_tau4_n3.sh
```

Train dynamics:

```text
epoch 1:  mean_loss=1.4085, mean_seg_loss=0.7217
epoch 10: mean_loss=0.9102, mean_seg_loss=0.4604
epoch 15: mean_loss=0.7684, mean_seg_loss=0.3826
epoch 20: mean_loss=0.7217, mean_seg_loss=0.3605

non-finite gradient skips:
epoch 6: 1
epoch 8: 2
epoch 9: 1
epoch 18: 1
epoch 19: 1
```

Eval command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
DFG_MODE=attn DFG_ATTN_DIM=256 DFG_ATTN_TAU=4.0 \
METRIC_THRESHOLDS=none PIXEL_STRIDE=4 SAVE_PATH=phase1_v1_attn_tau4 \
  bash test_6medical_selected_epochs.sh 15 16 17 18 19 20
python parse_test_log.py --log phase1_v1_attn_tau4/test.log
```

Mean results:

```text
Epoch   Mean Pixel AUC   Mean Pixel AP
15          87.45           31.47
16          87.19           31.09
17          86.85           30.03
18          87.07           30.01
19          86.47           28.94
20          87.46           30.75
```

Best V1 checkpoint:

```text
epoch 15: 87.45 / 31.47
```

Dataset-level notes:

```text
Liver improves:
  A0: 94.90 / 5.46
  V1 best: about 96.56 / 6.21

Retina AUC improves, AP slightly drops:
  A0: 90.17 / 46.89
  V1 best: about 93.55 / 45.17

Colon/Kvasir degrade strongly:
  ColonDB A0: 84.32 / 31.64
  ColonDB V1 best: about 80.23 / 26.22
  Kvasir A0: 85.21 / 53.84
  Kvasir V1 best: about 78.87 / 42.06
```

Attention debug:

```text
max entropy log(3)=1.0986

Brain, epoch 15, 1 batch:
  stage 1: H_N=0.8091 H_A=0.8343 w_N=[0.0338, 0.5467, 0.4195] w_A=[0.0424, 0.5272, 0.4304]
  stage 2: H_N=0.8205 H_A=0.7824 w_N=[0.0399, 0.5572, 0.4030] w_A=[0.0249, 0.5573, 0.4178]
  stage 3: H_N=0.8951 H_A=0.9444 w_N=[0.0710, 0.5130, 0.4159] w_A=[0.0991, 0.4877, 0.4132]

Kvasir, epoch 15, 1 batch:
  stage 1: H_N=0.8275 H_A=0.8112 w_N=[0.0401, 0.5322, 0.4277] w_A=[0.0501, 0.6206, 0.3292]
  stage 2: H_N=0.9811 H_A=0.9365 w_N=[0.1689, 0.4412, 0.3899] w_A=[0.1539, 0.5027, 0.3433]
  stage 3: H_N=0.9477 H_A=0.9585 w_N=[0.1018, 0.4745, 0.4237] w_A=[0.1259, 0.5374, 0.3367]
```

Diagnosis:

```text
V1 is not a pass. It improves some AUC/Liver behavior but hurts mean AP.
Attention does not fully collapse, but group 1 is consistently under-used.
The likely issue is not too few parameters; it is that pure attention replaces
the original MLP gate too aggressively and changes calibration/level selection.
```

Decision:

```text
Do not proceed to SS2D from pure V1 yet.
Run tau=8 ablation first.
If tau=8 still fails, implement hybrid MLP+attention.
```

## V2: Phase 1A Pure Attention DFG, tau=8

Status:

```text
completed
```

Purpose:

```text
Keep architecture and d_attn fixed, only soften attention to test whether tau=4
was too sharp and caused group 1 under-use.
```

Architecture:

```text
same as V1
```

Hyperparameters:

```text
dfg_attn_dim: 256
dfg_attn_tau: 8.0
W_Q/W_K init: Xavier uniform because dfg_attn_dim=256
image_lr for W_Q/W_K: 0.001, same image_adapter group
checkpoint folder: phase1_v2_attn_tau8
```

Train command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash train_phase1_v2_attn_tau8_n3.sh
```

Eval command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash test_phase1_v2_attn_tau8_selected_epochs.sh 10 11 12 13 14 15 16 17 18 19 20
```

Train dynamics:

```text
epoch 1:  mean_loss=1.3678, mean_seg_loss=0.6834
epoch 10: mean_loss=0.7962, mean_seg_loss=0.3633
epoch 15: mean_loss=0.7451, mean_seg_loss=0.3458
epoch 20: mean_loss=0.7071, mean_seg_loss=0.3286

non-finite gradient skips:
epoch 1: 3
epoch 7: 2
epoch 18: 1
epoch 19: 1
total: 7 / 7220 train batches
```

Mean results:

```text
Epoch   Mean Pixel AUC   Mean Pixel AP
10          89.72           34.32
11          89.20           33.58
12          90.05           34.34
13          89.63           34.70
14          88.64           32.44
15          88.86           32.90
16          89.50           33.29
17          89.67           32.95
18          89.49           33.38
19          89.85           33.97
20          89.43           33.07
```

Best V2 checkpoint:

```text
epoch 13: 89.63 / 34.70
```

Paper-style means:

```text
Best medical pixel-level mean by AP:
  epoch 13: 89.63 / 34.70

Best medical image-level mean by AP:
  epoch 10: 72.65 / 73.58
```

Comparison to fixed anchors:

```text
A0 local baseline epoch 19: 88.51 / 34.05
V1 tau4 best epoch 15:      87.45 / 31.47
V2 tau8 best epoch 13:      89.63 / 34.70

V2 vs A0: +1.12 AUC, +0.65 AP
V2 vs V1: +2.18 AUC, +3.23 AP
```

Dataset-level notes at V2 best epoch 13:

```text
Brain:          93.73 / 27.11
Liver:          95.59 / 4.05
Retina:         92.53 / 41.36
Colon_clinicDB: 86.89 / 47.82
Colon_colonDB:  81.11 / 28.56
Colon_Kvasir:   87.96 / 59.31
```

Interpretation:

```text
V2 is a useful improvement over V1 and slightly beats the local baseline on
mean pixel AUC/AP. The gain is driven mainly by colon datasets, especially
ClinicDB and Kvasir. Brain/Liver/Retina AP are weaker than A0, so V2 is not a
universal win yet.

Tau=8 should replace tau=4 as the stronger pure-attention Phase 1A setting.
Do not jump straight to claiming the full Phase 1 is solved. The next step
should check attention statistics for V2, then decide between:
  1. keep V2 tau=8 and add SS2D as Phase 1B, or
  2. test hybrid MLP+attention if attention behavior is still unstable.
```

Decision rule:

```text
If mean AP is still clearly below 34.05:
  do not proceed to SS2D; implement hybrid DFG.

If Colon/Kvasir AP improves but mean AP remains slightly below baseline:
  implement hybrid DFG before SS2D.

If tau=8 matches or beats A0:
  keep tau=8 as A1 candidate, then test SS2D as Phase 1B.
```

## Phase 1B Next Plan: Add SS2D On Top Of V2 Tau8

Status:

```text
implemented, not trained yet
```

Rollback anchor:

```text
Phase 1A V2 tau8 remains the accepted A1 checkpoint.
Pixel-level anchor: phase1_v2_attn_tau8/adapter_13.pth
Image-level anchor: phase1_v2_attn_tau8/adapter_10.pth
Do not overwrite or delete phase1_v2_attn_tau8.
```

Phase 1B changes only:

```text
Add a lightweight four-direction SS2D-style spatial residual branch before
computing Q_v for attention DFG.
```

Keep unchanged from V2:

```text
dfg_mode: attn
dfg_attn_dim: 256
dfg_attn_tau: 8.0
dual-softmax normal/abnormal attention
no W_V
original text values in CLIP space
img_feat segmentation scale: 10
optimizer/LR/scheduler/batch/epoch/runtime settings
```

New Phase 1B hyperparameters:

```text
use_ss2d_dfg: true
dfg_gamma_max: 0.2
gamma: 0.2 * tanh(raw_gamma)
raw_gamma init: 0
SS2D-style branch: LN -> Linear(768->768) -> SiLU -> 4-direction scan -> LN -> GAP
```

Expected folder/script names:

```text
folder: phase1_v3_attn_tau8_ss2d_g02
train:  train_phase1_v3_attn_tau8_ss2d_g02_n3.sh
test:   test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh
```

Implementation note:

```text
model/adapter_modules.py:
  added dependency-free FourDirectionSS2D-style scan and DFGSS2DResidualBranch

model/adapter.py:
  added use_ss2d_dfg and dfg_gamma_max
  registered dfg_ss2d_branches and dfg_raw_gamma under image_adapter
  v_global = v_gap + gamma * v_ss2d only when use_ss2d_dfg=true
  raw_gamma starts at 0, so gamma=0 and V3 starts exactly from V2 behavior

train.py/test.py:
  added matching CLI args and checkpoint metadata checks

test_6medical_selected_epochs.sh:
  added USE_SS2D_DFG and DFG_GAMMA_MAX env switches
```

Important implementation wording:

```text
This V3 implementation is a lightweight four-direction SS2D-style scan branch,
not the full VMamba CUDA selective-scan implementation. In reports, describe it
as "dependency-free SS2D-style residual branch" or "lightweight four-direction
SS2D-style spatial scan branch".
```

Acceptance/rollback rule:

```text
If V3 underperforms V2 on both medical pixel-level mean and image-level mean:
  rollback to V2 tau8.

If V3 improves pixel AP but hurts image AP, or vice versa:
  keep both as ablations and report the trade-off.

If V3 improves Brain/Liver/Retina AP while preserving ClinicDB/Kvasir gains:
  V3 becomes the preferred Phase 1 candidate.
```

## Next Candidate If V3 SS2D Fails: V4 Hybrid DFG

Planned idea:

```text
variant: V4 / hybrid DFG rescue
dfg_mode: hybrid
T_mlp = original MLP-fused text descriptor
T_attn = attention-fused text descriptor
lambda initialized near 0 or bounded small
T_final = normalize((1 - lambda) * T_mlp + lambda * T_attn)
```

Rationale:

```text
Hybrid starts from baseline behavior and lets attention act as a correction.
This is more conservative for VisA-to-medical zero-shot transfer than pure
attention replacement.
```

## Append Template

Use this template for later edits:

```text
## Vx: <short name>

Status:
Architecture change:
Hyperparameters:
Train command:
Eval command:
Train dynamics:
Results:
Diagnosis:
Decision:
```

## V3 Result Snapshot: Phase 1B SS2D g02 vs Phase 1A tau8

Status:

```text
Train finished for phase1_v3_attn_tau8_ss2d_g02.
Official quick eval completed for epochs 19 and 20 only.
Full epoch sweep was intentionally skipped to save time.
```

Protocol:

```text
train dataset: VisA
eval datasets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir
metric_thresholds: none
pixel_stride: 4
Phase 1A comparison folder: phase1_v2_attn_tau8
Phase 1B comparison folder: phase1_v3_attn_tau8_ss2d_g02
```

Epoch-matched comparison:

```text
Medical pixel-level mean, 6 datasets:
  Phase 1A tau8 epoch 19: 89.85 / 33.97
  Phase 1B SS2D epoch 19: 89.91 / 32.24
  delta:                  +0.06 / -1.73

  Phase 1A tau8 epoch 20: 89.43 / 33.07
  Phase 1B SS2D epoch 20: 89.21 / 31.63
  delta:                  -0.22 / -1.44

Medical image-level mean, 3 datasets:
  Phase 1A tau8 epoch 19: 70.40 / 70.78
  Phase 1B SS2D epoch 19: 72.88 / 72.29
  delta:                  +2.48 / +1.51

  Phase 1A tau8 epoch 20: 71.82 / 71.93
  Phase 1B SS2D epoch 20: 72.89 / 72.79
  delta:                  +1.07 / +0.86
```

Key dataset-level deltas:

```text
Retina OCT pixel:
  epoch 19: Phase 1A 93.52 / 40.66 -> Phase 1B 90.71 / 29.46, delta -2.81 / -11.20
  epoch 20: Phase 1A 93.41 / 40.74 -> Phase 1B 91.90 / 31.12, delta -1.51 / -9.62

Kvasir pixel:
  epoch 19: Phase 1A 88.55 / 59.96 -> Phase 1B 87.21 / 56.05, delta -1.34 / -3.91
  epoch 20: Phase 1A 88.14 / 58.96 -> Phase 1B 85.91 / 54.01, delta -2.23 / -4.95

Liver CT pixel:
  epoch 19: Phase 1A 95.80 / 4.21 -> Phase 1B 97.43 / 7.93, delta +1.63 / +3.72
  epoch 20: Phase 1A 95.58 / 4.29 -> Phase 1B 97.20 / 8.50, delta +1.62 / +4.21

Liver CT image:
  epoch 19: Phase 1A 52.99 / 42.82 -> Phase 1B 60.89 / 50.35, delta +7.90 / +7.53
  epoch 20: Phase 1A 56.01 / 46.56 -> Phase 1B 61.31 / 50.79, delta +5.30 / +4.23
```

Train dynamics:

```text
V3 best train loss:
  epoch 18 mean_loss=0.7201, mean_seg_loss=0.3310

V3 final gamma values:
  stage1 gamma:  0.0279
  stage2 gamma: -0.0322
  stage3 gamma:  0.0371

V3 final ss2d_ratio:
  stage1: about 1.0
  stage2: about 1.23
  stage3: about 1.25
```

Diagnosis:

```text
The epoch-19/20 comparison shows a real trade-off, not just a protocol mismatch.
Phase 1B SS2D improves image-level transfer and strongly helps Liver CT, but it
hurts pixel AP on Retina OCT and Kvasir.

Gamma itself is small and not saturated, but ss2d_ratio is about 1.0-1.3 near
the end of training. Therefore the SS2D residual branch is not weak in practice:
it can shift the visual query used by attention DFG by a magnitude comparable to
the original GAP visual query.

The current branch likely changes the spatial bias toward smooth/global scan
features. This is useful for Liver CT but hurts fine-grained retinal structure
and Kvasir boundary/texture localization.
```

Comment on the quick-screen assumption:

```text
Using only epoch 19/20 as a cheap screen is reasonable for deciding whether V3
is promising. However, it is not enough to conclude that all earlier epochs
would also beat Phase 1A tau8.

Phase 1A tau8 has its best pixel AP at epoch 13, while epochs 19/20 are lower.
Therefore V3 should be compared against Phase 1A tau8 at matching epochs for a
quick screen, but anchor selection still needs either targeted earlier epochs or
a full sweep.
```

Decision:

```text
Do not promote V3 SS2D g02 to the main anchor yet.

Keep Phase 1A tau8 as the primary anchor for pixel-level AP.
Keep V3 SS2D g02 as an ablation because it improves medical image-level mean and
Liver CT clearly.
```

Optimal next plan for Phase 1B:

```text
1. Run a targeted V3 eval on epochs 13, 15, 17, 18 instead of a full sweep.
   Command:
     bash test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh 13 15 17 18

2. If Retina/Kvasir AP still drops, train a smaller SS2D variant:
   dfg_gamma_max=0.05 first, then 0.10 only if 0.05 is too weak.

3. If modifying the architecture, normalize or cap the SS2D residual contribution
   so ss2d_ratio stays closer to 0.3-0.5 instead of 1.0-1.3.

4. Do not add DoRA, dynamic depthwise, or few-shot components until Phase 1B is
   resolved. Otherwise the Retina/Kvasir regression source becomes ambiguous.
```

## V3 Full Tested Epochs 10-20: Best Epoch Table

Status:

```text
Phase 1B SS2D g02 was later evaluated for epochs 10-20 on all six medical
pixel-level datasets and the three medical image-level datasets.
This supersedes the earlier quick-only epoch 19/20 decision.
```

Summary table:

| Model | Selection rule | Epoch | Medical pixel-level mean, 6 datasets | Medical image-level mean, 3 datasets |
|---|---:|---:|---:|---:|
| Phase 1A tau8 | best pixel AP | 13 | 89.63 / 34.70 | 72.23 / 73.11 |
| Phase 1A tau8 | best image AP | 10 | 89.72 / 34.32 | 72.65 / 73.58 |
| Phase 1B SS2D g02 | best pixel AP | 11 | 90.23 / 34.77 | 73.23 / 72.58 |
| Phase 1B SS2D g02 | best image AP | 12 | 88.17 / 30.57 | 74.08 / 74.24 |
| Phase 1B SS2D g02 | best pixel AUC | 14 | 90.52 / 33.68 | 72.20 / 72.74 |

Best-to-best comparison:

| Comparison | Phase 1A tau8 | Phase 1B SS2D g02 | Delta |
|---|---:|---:|---:|
| Pixel-level best AP mean, 6 datasets | e13: 89.63 / 34.70 | e11: 90.23 / 34.77 | +0.60 / +0.07 |
| Image-level best AP mean, 3 datasets | e10: 72.65 / 73.58 | e12: 74.08 / 74.24 | +1.43 / +0.66 |

V3 best pixel checkpoint, epoch 11:

| Dataset | Pixel AUC / AP | Image AUC / AP |
|---|---:|---:|
| BrainMRI | 93.01 / 28.16 | 80.99 / 94.61 |
| Liver CT | 96.87 / 6.33 | 59.25 / 48.30 |
| Retina OCT | 93.14 / 38.55 | 79.46 / 74.84 |
| ClinicDB | 88.29 / 49.91 | 0.00 / 0.00 |
| ColonDB | 83.34 / 30.60 | 0.00 / 0.00 |
| Kvasir | 86.72 / 55.06 | 0.00 / 0.00 |
| Mean | 90.23 / 34.77 | 73.23 / 72.58 |

V3 best image checkpoint, epoch 12:

| Dataset | Pixel AUC / AP | Image AUC / AP |
|---|---:|---:|
| BrainMRI | 92.73 / 27.46 | 80.46 / 94.58 |
| Liver CT | 96.48 / 7.55 | 59.86 / 50.48 |
| Retina OCT | 89.06 / 28.60 | 81.91 / 77.65 |
| ClinicDB | 86.81 / 43.59 | 0.00 / 0.00 |
| ColonDB | 80.05 / 24.29 | 0.00 / 0.00 |
| Kvasir | 83.87 / 51.94 | 0.00 / 0.00 |
| Mean | 88.17 / 30.57 | 74.08 / 74.24 |

Decision update:

```text
Phase 1B SS2D g02 can be promoted from "ablation only" to a Phase 1B candidate
because it beats Phase 1A tau8 on best pixel-level mean AP and best image-level
mean AP.

Recommended checkpoint for segmentation/pixel reporting:
  phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth

Recommended checkpoint for image-level reporting:
  phase1_v3_attn_tau8_ss2d_g02/adapter_12.pth

Important caveat:
  V3 improves the mean, but Retina OCT and Kvasir pixel AP are still weaker than
  the strongest Phase 1A tau8 checkpoints. Report V3 as a mean-level improvement
  with a Retina/Kvasir trade-off.
```

Final report claim:

```text
Phase 1B SS2D achieves the best mean pixel-level result among our local Phase 1
variants, improving from 89.6/34.7 to 90.2/34.8 over Phase 1A tau8. It also
improves image-level mean performance from 72.7/73.6 to 74.1/74.2. However, the
improvement is modest and comes with dataset-level trade-offs, especially lower
pixel AP on Kvasir and Retina OCT.
```

Final Phase 1B follow-up order:

```text
1. Log diagnostics for Phase 1A tau8 vs Phase 1B SS2D g02 at the key checkpoint:
   Phase 1A tau8 pixel anchor: phase1_v2_attn_tau8/adapter_13.pth
   Phase 1B SS2D pixel anchor: phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth

   Diagnostics to collect:
     entropy normal/abnormal
     attention weight L1 distance
     SS2D residual norm ratio
     cosine similarity between the Phase 1A query and the Phase 1B query

2. If diagnostics show SS2D shifts attention too strongly:
   try V3c with a small residual on attention weights:
     weights_final = (1 - beta) * weights_phase1a + beta * weights_ss2d
     beta = 0.05

   Rationale:
     V3c directly limits how much SS2D can change group attention, so it is the
     most targeted fix for Kvasir/Retina regression while preserving some mean
     gains.

3. Train V3b with dfg_gamma_max=0.05 only if there is still time.

   Rationale:
     Smaller gamma may recover Kvasir/Retina, but it can also remove useful
     ClinicDB/Liver gains because it weakens the whole SS2D query branch.
```

Practical stop rule:

```text
If near deadline:
  stop training and write the analysis.

If there is extra time:
  prioritize V3c beta=0.05 over V3b gamma_max=0.05.
```

## Diagnostic: Phase 1A tau8 e13 vs Phase 1B SS2D e11

Command:

```bash
conda run --no-capture-output -n torchhuy python debug_attention_stats.py \
  --datasets Retina Colon_Kvasir Liver Colon_clinicDB \
  --batch_size 4 \
  --max_batches 2 \
  --num_workers 0 \
  | tee phase1_v3_attn_tau8_ss2d_g02/diagnostic_v2e13_vs_v3e11_attention_focused.txt
```

Compared checkpoints:

```text
Phase 1A tau8 reference: phase1_v2_attn_tau8/adapter_13.pth
Phase 1B SS2D candidate: phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth
```

Focused diagnostic summary:

| Dataset | Stage | Query cosine | L1 normal | L1 abnormal | SS2D ratio | Notes |
|---|---:|---:|---:|---:|---:|---|
| Retina OCT | 1 | -0.027 | 0.139 | 0.057 | 0.817 | mild weight shift |
| Retina OCT | 2 | -0.092 | 0.268 | 0.051 | 1.046 | strong normal-weight shift |
| Retina OCT | 3 | 0.116 | 0.410 | 0.132 | 1.154 | strongest normal-weight shift |
| Kvasir | 1 | -0.027 | 0.303 | 0.140 | 0.849 | strong shift from stage 1 |
| Kvasir | 2 | -0.093 | 0.524 | 0.257 | 1.395 | very strong shift |
| Kvasir | 3 | 0.107 | 0.150 | 0.205 | 1.283 | abnormal weights still shift |
| Liver CT | 1 | -0.027 | 0.143 | 0.052 | 0.810 | mild shift |
| Liver CT | 2 | -0.092 | 0.275 | 0.053 | 1.033 | moderate shift |
| Liver CT | 3 | 0.114 | 0.456 | 0.178 | 1.168 | strong stage-3 shift |
| ClinicDB | 1 | -0.027 | 0.303 | 0.138 | 0.838 | strong shift |
| ClinicDB | 2 | -0.094 | 0.550 | 0.328 | 1.697 | largest shift |
| ClinicDB | 3 | 0.107 | 0.151 | 0.232 | 1.315 | abnormal weights shift |

Interpretation:

```text
The diagnostic confirms that Phase 1B SS2D does not act as a tiny local tweak.
Even at epoch 11, the SS2D residual changes the DFG query direction strongly:
query cosine between Phase 1A and Phase 1B is near zero or negative for stages
1-2, and only about 0.11 for stage 3.

The largest attention drift is in stage 2:
  Retina normal L1: 0.268
  Kvasir normal L1: 0.524
  ClinicDB normal L1: 0.550

SS2D ratio is also large:
  Retina: about 0.82 / 1.05 / 1.15
  Kvasir: about 0.85 / 1.40 / 1.28
  ClinicDB: about 0.84 / 1.70 / 1.32

This supports the hypothesis that Kvasir/Retina AP drops are caused by the SS2D
branch shifting attention weights too aggressively, especially in stage 2 and
stage 3. It also explains why the same branch can improve the mean: it changes
the group weighting enough to help ClinicDB/Liver while hurting fine-grained
Retina/Kvasir localization.
```

Action after diagnostic:

```text
Prioritize V3c weight residual beta=0.05.

Do not make gamma_max=0.05 the first rescue run unless there is extra time,
because gamma scaling weakens the whole SS2D query branch and may erase
ClinicDB/Liver gains. V3c is more targeted because it limits the final attention
weight drift directly.
```

## Pre-V3c Rollback Snapshot: Phase 1B SS2D g02

Purpose:

```text
This snapshot records the current Phase 1B state before implementing V3c-beta010.
Use it to rollback if the new weight-residual fusion underperforms or introduces
implementation risk.
```

Architecture:

```text
run name: phase1_v3_attn_tau8_ss2d_g02
dfg_mode: attn
dfg_attn_dim: 256
dfg_attn_tau: 8.0
use_ss2d_dfg: true
dfg_gamma_max: 0.2
fusion behavior: feature/query residual

Formula:
  v_gap = GAP(V_i)
  v_ss2d = SS2D_branch(V_i)
  gamma = 0.2 * tanh(raw_gamma)
  v_global = v_gap + gamma * v_ss2d
  q = W_Q(v_global)
  w = softmax(q @ K / tau)
```

Current best local results:

```text
Pixel-level reporting checkpoint:
  phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth
  medical pixel-level mean over 6 datasets: 90.23 / 34.77

Image-level reporting checkpoint:
  phase1_v3_attn_tau8_ss2d_g02/adapter_12.pth
  medical image-level mean over 3 datasets: 74.08 / 74.24
```

Rollback commands:

```bash
bash train_phase1_v3_attn_tau8_ss2d_g02_n3.sh
bash test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh 11 12
```

Rollback arguments:

```text
--dfg_mode attn
--dfg_attn_dim 256
--dfg_attn_tau 8.0
--use_ss2d_dfg
--dfg_gamma_max 0.2
--n_groups 3
--img_size 518
--batch_size 6
--epoch 20
```

Files touched before V3c:

```text
model/adapter.py
model/adapter_modules.py
train.py
test.py
train_phase1_v3_attn_tau8_ss2d_g02_n3.sh
test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh
test_6medical_selected_epochs.sh
debug_attention_stats.py
parse_test_log.py
EXPERIMENT_LOG_PHASE1.md
PHASE1_DFG_ATTENTION_SS2D_PLAN.md
```

## V3c-beta010 Implementation: Attention-Weight Residual

Status:

```text
Implemented, not trained yet.
```

Run name:

```text
phase1_v3c_attn_tau8_ss2d_weightres_beta010
```

Architecture change:

```text
New CLI args:
  --dfg_ss2d_fusion {feature_residual,weight_residual}
  --dfg_beta 0.10

Existing V3 behavior is preserved by:
  --dfg_ss2d_fusion feature_residual

V3c-beta010 behavior is enabled by:
  --dfg_ss2d_fusion weight_residual
  --dfg_beta 0.10
```

V3c formula:

```text
v_gap = GAP(V_i)
v_ss2d = SS2D_branch(V_i)

q_gap = W_Q(v_gap)
q_ss2d = W_Q(v_ss2d)

w_gap_N = softmax(q_gap @ K_N / tau)
w_ss2d_N = softmax(q_ss2d @ K_N / tau)
w_final_N = 0.90 * w_gap_N + 0.10 * w_ss2d_N

w_gap_A = softmax(q_gap @ K_A / tau)
w_ss2d_A = softmax(q_ss2d @ K_A / tau)
w_final_A = 0.90 * w_gap_A + 0.10 * w_ss2d_A

The original CLIP-space normal and abnormal text descriptors remain the values.
No W_V is added.
```

Scripts:

```text
train:
  train_phase1_v3c_attn_tau8_ss2d_weightres_beta010_n3.sh

test:
  test_phase1_v3c_attn_tau8_ss2d_weightres_beta010_selected_epochs.sh
```

Expected train args:

```text
--dataset VisA
--n_groups 3
--dfg_mode attn
--dfg_attn_dim 256
--dfg_attn_tau 8.0
--use_ss2d_dfg
--dfg_gamma_max 0.2
--dfg_ss2d_fusion weight_residual
--dfg_beta 0.10
--img_size 518
--batch_size 6
--epoch 20
--amp
--grad_checkpointing
--num_workers 6
```

Metadata:

```text
Checkpoints now save:
  dfg_mode
  dfg_attn_dim
  dfg_attn_tau
  use_ss2d_dfg
  dfg_gamma_max
  dfg_ss2d_fusion
  dfg_beta

test.py checks these fields and raises a clear ValueError on mismatch before
loading adapter state_dicts.
```

V3c safety audit:

```text
Legacy checkpoint compatibility:
  phase1_v2_attn_tau8/adapter_13.pth has no use_ss2d_dfg, dfg_gamma_max,
  dfg_ss2d_fusion, or dfg_beta metadata. test.py treats missing use_ss2d_dfg as
  false and missing fusion/beta as feature_residual/0.10, so old V2 remains
  evaluable with use_ss2d_dfg=false.

  phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth has use_ss2d_dfg=true and
  dfg_gamma_max=0.2, but no dfg_ss2d_fusion or dfg_beta metadata. test.py falls
  back to feature_residual/0.10, so old V3 remains evaluable as the original
  feature-residual SS2D run.

Beta logging:
  train.py logs vars(args), which includes dfg_ss2d_fusion and dfg_beta at run
  start. V3c script passes --dfg_beta 0.10 explicitly.

Weight residual:
  weight_residual mode does not use v_gap + gamma * v_ss2d and does not scale
  v_ss2d by gamma before W_Q. gamma is only recorded for diagnostics/metadata in
  this mode.

Probability check:
  model diagnostics now log weights_normal_sum_error and
  weights_abnormal_sum_error. Dummy smoke test showed max error about 6e-08.

V3c diagnostics:
  debug_attention_stats.py now reads candidate checkpoint fusion/beta metadata
  and logs within-candidate drift:
    L1(w_final, w_gap)
    L1(w_ss2d, w_gap)
    effective_drift = beta * L1(w_ss2d, w_gap)
    cosine(q_gap, q_ss2d)
```

## V3c Betawarm010 FP32 Attention Stability Fix And Final Test Snapshot

Date: 2026-06-29

Run folder:

```text
phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3
```

Purpose:

```text
Keep V3c weight_residual betawarm010 unchanged architecturally, but compute
the weight_residual DFG q/k scores and softmax in fp32 to avoid the non-finite
loss instability observed after epoch 10 in the previous betawarm010 run.
```

Important note:

```text
Old V3c betawarm010 epoch 9/10 numbers were produced before the fp32 attention
stability fix. The clean fixed run should be reported separately as
V3c weight_residual betawarm010 + fp32 attention.
```

Best checkpoint from the fixed run:

```text
epoch 9
```

Mean medical zero-shot results:

| Variant | Epoch | Pixel mean, 6 datasets | Image mean, 3 datasets |
|---|---:|---:|---:|
| Baseline A0 | 19 | 88.51 / 34.05 | 71.70 / 72.65 |
| V1 tau4 | 15 | 87.45 / 31.47 | 73.04 / 74.07 |
| V2 / Phase 1A tau8 | 13 | 89.63 / 34.70 | 72.23 / 73.11 |
| V3 SS2D g02 | 11 | 90.23 / 34.77 | 73.23 / 72.58 |
| V3c betawarm010 old | 9 | 90.91 / 35.63 | 72.10 / 73.02 |
| V3c betawarm010 fp32 attention | 9 | 90.76 / 39.82 | 73.80 / 75.06 |

Pixel-level medical datasets at epoch 9:

| Dataset | Pixel AUC / AP |
|---|---:|
| ColonDB | 84.29 / 31.03 |
| ClinicDB | 89.66 / 52.87 |
| Kvasir | 88.28 / 60.50 |
| BrainMRI | 95.96 / 46.05 |
| Liver CT | 96.97 / 6.28 |
| Retina OCT | 89.39 / 42.20 |

Image-level medical datasets at epoch 9:

| Dataset | Image AUC / AP |
|---|---:|
| BrainMRI | 82.53 / 95.40 |
| Liver CT | 56.74 / 48.96 |
| Retina OCT | 82.12 / 80.82 |

Conclusion:

```text
V3c weight_residual betawarm010 + fp32 attention is the best local Phase 1
variant by medical pixel AP mean and image AP mean. The strongest gain is pixel
AP: 39.82 vs 34.70 for Phase 1A tau8 and 34.77 for V3 SS2D g02.

This is a stability fix, not a new architectural ablation. The recommended
Phase 1 anchor for the next step is:
  phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/adapter_9.pth
```
