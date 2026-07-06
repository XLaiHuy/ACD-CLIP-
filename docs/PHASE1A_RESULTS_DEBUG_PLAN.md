# Phase 1A Attention DFG Results and Debug Plan

## Current Protocol

All Phase 1A results below use the local baseline protocol:

```text
train dataset: VisA
n_groups: 3
test datasets: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir
metric_thresholds: none
pixel_stride: 4
TTA: no
```

Local reproduced baseline:

```text
A0 mainbase epoch 19 mean pixel AUC/AP = 88.51 / 34.05
```

Phase 1A attention DFG:

```text
dfg_mode: attn
dfg_attn_dim: 256
dfg_attn_tau: 4.0 fixed
W_Q/W_K: Linear(768 -> 256), no W_V
values: original CLIP text features
trainable parameters: 10,545,408
baseline trainable parameters: 9,960,978
extra trainable parameters: +584,430
```

The parameter increase is small. The current failure is more likely due to the
fusion behavior changing too much than simply having too many parameters.

## Phase 1A Results

Mean over six medical datasets:

```text
epoch  mean pixel AUC  mean pixel AP
15     87.45           31.47
16     87.19           31.09
17     86.85           30.03
18     87.07           30.01
19     86.47           28.94
20     87.46           30.75
```

Best Phase 1A checkpoint by mean AP:

```text
epoch 15: 87.45 / 31.47
```

Compared with A0:

```text
A0 baseline:       88.51 / 34.05
best Phase 1A:     87.45 / 31.47
difference:        -1.06 / -2.58
```

## Dataset-Level Pattern

Best or representative Phase 1A behavior:

```text
Liver improves:
  A0:        94.90 / 5.46
  A1 best:   96.56 / 6.21

Retina AUC improves but AP does not:
  A0:        90.17 / 46.89
  A1 best:   93.55 / 45.17

Brain AP can improve at epoch 15 but AUC drops:
  A0:        93.59 / 29.98
  A1 e15:    92.99 / 32.53

ColonDB and Kvasir degrade clearly:
  ColonDB A0:      84.32 / 31.64
  ColonDB A1 best: 80.23 / 26.22

  Kvasir A0:       85.21 / 53.84
  Kvasir A1 best:  78.87 / 42.06
```

Interpretation:

```text
Attention DFG is not collapsing completely.
It improves or partially improves several domains.
However, replacing the MLP gate with pure attention hurts medical polyp datasets,
especially AP, so Phase 1A is not yet a pass.
```

## Train Dynamics

Training loss decreases normally:

```text
epoch 1:  mean_loss=1.4085, mean_seg_loss=0.7217
epoch 10: mean_loss=0.9102, mean_seg_loss=0.4604
epoch 15: mean_loss=0.7684, mean_seg_loss=0.3826
epoch 20: mean_loss=0.7217, mean_seg_loss=0.3605
```

Non-finite gradient skips happened:

```text
epoch 6:  1 skip
epoch 8:  2 skips
epoch 9:  1 skip
epoch 18: 1 skip
epoch 19: 1 skip
```

This does not prove the run is broken because loss remains finite and decreases.
The baseline AMP run also showed non-finite skips. Still, if later ablations are
close, a no-AMP or lower-LR check is useful.

## Main Hypothesis

The current A1 does not fail because it lacks capacity. It likely fails because
it replaces the original learned MLP gate too aggressively.

Original DFG:

```text
v_gap -> MLP -> weights over text groups
```

Phase 1A:

```text
v_gap + text features -> attention weights
```

This changes the fusion function and text-score calibration at once. For
industrial-to-medical transfer, especially Colon/Kvasir, the original MLP gate
may provide a useful conservative prior. Pure attention can over-specialize to
VisA or select text levels in a way that improves ranking AUC but hurts AP.

## What To Debug Before Adding SS2D

### 1. Attention Entropy

Check whether normal/abnormal weights are:

```text
too uniform: entropy near log(3) = 1.099
too sharp: entropy near 0
biased: one text group dominates all datasets
unstable: normal and abnormal select inconsistent groups
```

Expected useful range is not known, but a healthy module should not collapse to
the same group for all images and both semantic states.

Initial sampled debug:

```text
Brain, epoch 15, 1 batch:
  stage 1: H_N=0.8091 H_A=0.8343 w_N=[0.0338, 0.5467, 0.4195] w_A=[0.0424, 0.5272, 0.4304]
  stage 2: H_N=0.8205 H_A=0.7824 w_N=[0.0399, 0.5572, 0.4030] w_A=[0.0249, 0.5573, 0.4178]
  stage 3: H_N=0.8951 H_A=0.9444 w_N=[0.0710, 0.5130, 0.4159] w_A=[0.0991, 0.4877, 0.4132]

Retina, epoch 17, 1 batch:
  stage 1: H_N=0.8136 H_A=0.8348 w_N=[0.0362, 0.5514, 0.4124] w_A=[0.0451, 0.5494, 0.4055]
  stage 2: H_N=0.8315 H_A=0.8045 w_N=[0.0471, 0.5709, 0.3820] w_A=[0.0340, 0.5636, 0.4024]
  stage 3: H_N=0.9713 H_A=0.9320 w_N=[0.1218, 0.4980, 0.3802] w_A=[0.0922, 0.5011, 0.4067]

Kvasir, epoch 15, 1 batch:
  stage 1: H_N=0.8275 H_A=0.8112 w_N=[0.0401, 0.5322, 0.4277] w_A=[0.0501, 0.6206, 0.3292]
  stage 2: H_N=0.9811 H_A=0.9365 w_N=[0.1689, 0.4412, 0.3899] w_A=[0.1539, 0.5027, 0.3433]
  stage 3: H_N=0.9477 H_A=0.9585 w_N=[0.1018, 0.4745, 0.4237] w_A=[0.1259, 0.5374, 0.3367]
```

Early read:

```text
max entropy log(3)=1.0986
attention is not fully collapsed, but group 1 is consistently under-used
and most mass goes to group 2/3 across datasets.
```

This supports trying a softer attention temperature or a hybrid baseline
residual before adding SS2D.

### 2. Text Descriptor Norm and Logit Scale

Attention path normalizes:

```text
T_final_N/A = normalize(weighted_sum(text_features))
```

Original MLP path does not explicitly normalize after weighted sum. This is
conceptually cleaner for cosine similarity, but it also changes logit scale. We
should inspect:

```text
norm(T_mlp_N/A)
norm(T_attn_N/A)
seg logit range before softmax
anomaly score min/max/mean
```

If attention logits are too sharp or too flat, AP can drop while AUC improves.

### 3. Learning Rate for New Attention

Current optimizer uses:

```text
image_lr = 0.001 for all image_adapter parameters
```

This means W_Q/W_K train at the same LR as Conv-LoRA/image adapters. Since
W_Q/W_K are newly initialized and control fusion directly, this may be too high.
Candidate check:

```text
dfg_attn_lr = 0.0001 or 0.0003
other image_adapter lr = 0.001
```

### 4. Hard Replacement vs Residual Fusion

Most likely rescue:

```text
T_final = (1 - lambda) * T_mlp + lambda * T_attn
lambda = sigmoid(raw_lambda) or bounded scalar
init lambda near 0
```

This starts from the reproduced baseline behavior and lets attention contribute
only if useful. It adds a small amount of complexity but is safer for zero-shot
transfer than pure replacement.

## Should We Accept More Parameters?

Yes, but only if the extra parameters preserve transfer behavior.

Good extra parameters:

```text
residual/hybrid attention gate initialized near baseline
low-rank or small W_Q/W_K
bounded mixing coefficient
separate lower LR for new fusion parameters
```

Risky extra parameters:

```text
larger d_attn without preserving MLP behavior
trainable temperature without bounds
adding SS2D before understanding why A1 hurts AP
dynamic/depthwise changes before DFG is stable
```

So the answer is:

```text
Accept a little more capacity if it makes the model more conservative, not more
aggressive.
```

## Recommended Next Plan

### Step A: Add Debug Logging, No Retraining

Inspect attention behavior for checkpoints 15 and 20:

```text
attention entropy per stage
mean attention weights for normal/abnormal
T_final norm
seg logit range
```

Do this on:

```text
VisA small train subset
Brain
Liver
Retina
Colon_clinicDB
Colon_Kvasir
```

### Step B: Cheap Ablation Without Changing Architecture Too Much

Run one conservative variant:

```text
A1-tau8:
  dfg_attn_dim=256
  dfg_attn_tau=8.0
```

Reason:

```text
If attention is too sharp, tau=8 softens selection.
It is cheaper than adding SS2D and keeps Phase 1A scope.
```

Run command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash train_phase1_v2_attn_tau8_n3.sh
```

After training, test selected late epochs:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash test_phase1_v2_attn_tau8_selected_epochs.sh 15 16 17 18 19 20
```

Decision rule:

```text
If mean AP is still clearly below 34.05, do not proceed to SS2D.
If Colon/Kvasir AP improves but mean is still slightly below baseline, consider
hybrid DFG before SS2D.
If tau8 matches or beats baseline, keep tau8 as A1 candidate and then test SS2D
as Phase 1B.
```

### Step C: Preferred Rescue If A1-tau8 Still Fails

Implement hybrid DFG:

```text
dfg_mode = hybrid
T_mlp = original MLP-fused text
T_attn = attention-fused text
lambda = 0.2 * sigmoid/raw bounded or initialized near 0
T_final = normalize((1 - lambda) * T_mlp + lambda * T_attn)
```

This is the best next architecture direction because it protects baseline
behavior and tests whether attention is useful as an additive correction.

### Step D: Only Then Add SS2D

Move to Phase 1B only after:

```text
A1 attention-only is understood, or
A1 hybrid gives a non-negative result vs baseline.
```

If pure attention remains negative, SS2D should be added to the hybrid path, not
to the pure attention path.
