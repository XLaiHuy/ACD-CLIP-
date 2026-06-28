# Phase 1 Plan: DFG Attention And SS2D Spatial Context

This note defines Phase 1 for the clean `ACD-CLIP-mainbase-newest` restart.
Phase 1 only changes the Dynamic Fusion Gateway (DFG). It does not add DoRA,
dynamic depthwise Conv-LoRA, few-shot memory, PaDiM, or PatchCore.

The goal is to test whether a more content-aware DFG improves zero-shot
industrial-to-medical transfer while preserving the ACD-CLIP prior.

## Scope

Training:

```text
source dataset: VisA
target eval: Brain, Liver, Retina, Colon_clinicDB, Colon_colonDB, Colon_Kvasir
n_groups: 3
```

Phase 1 is split into two ablations:

```text
Phase 1A: replace MLP DFG gating with dual-softmax visual-text attention
Phase 1B: add SS2D-style spatial context residual branch before DFG attention
```

Do not mix Phase 1 with:

```text
DoRA text adapter
static/dynamic depthwise Conv-LoRA
few-shot memory bank
PaDiM/PatchCore
class mass prior
new tau_seg calibration
new LR scheduler
```

## Baseline DFG In Current Code

Current segmentation DFG is in `model/adapter.py`:

```python
img_feat = vision_tokens[i]  # [B, patch_num, 768]
v_gap = img_feat.mean(dim=1, keepdim=True)
gate_weights = self.image_adapter["vision_text_gate"][i](v_gap)
gate_weights = gate_weights.view(B, self.n_groups, 2)
gate_weights = F.softmax(gate_weights, dim=1)
group_text_features = text_features.permute(1, 0, 2, 3)
group_text_features = group_text_features * gate_weights.unsqueeze(2)
group_text_features = group_text_features.sum(dim=1)
```

Important interpretation:

```text
v_global is GAP over segmentation patch tokens, not CLS/det_tokens.
```

Therefore Phase 1 should modify the patch-token DFG path, not the image-level
classification path.

## Lessons From Earlier Phase2 Repos

The previous Phase2 experiments produced useful negative evidence:

```text
1. Do not bundle many architectural changes in one run.
2. Do not select checkpoints by VisA train loss only.
3. Do not let new DFG modules dominate the original ACD-CLIP prior.
4. Do not use joint normal+abnormal text attention with strong global class bias.
5. Do not infer TTA/no-TTA from folder names; inspect logs.
6. Do not compare exact and binned metrics as equivalent.
7. Good source convergence can still mean poor medical zero-shot transfer.
```

Phase 1 is designed around these lessons:

```text
dual-softmax keeps normal and abnormal text states separate
SS2D-style scan is residual, zero-initialized, and bounded
training hyperparameters stay aligned with baseline
evaluation uses the same six medical datasets and same metric mode
```

## Phase 1A: Dual-Softmax Attention DFG

### Motivation

The original MLP gate uses only visual global context to choose text levels.
It does not directly compare visual content to text features.

Phase 1A replaces:

```text
weights = MLP(v_global)
```

with content-based attention:

```text
Q_v = W_Q(v_global)
K_N = W_K([T1_N, T2_N, T3_N])
K_A = W_K([T1_A, T2_A, T3_A])
```

Normal and abnormal features must have separate softmax operations:

```text
w_N = softmax(Q_v K_N^T / sqrt(d_attn) / tau_fusion)
w_A = softmax(Q_v K_A^T / sqrt(d_attn) / tau_fusion)
```

Then:

```text
T_final_N = normalize(sum_j w_N,j * T_j_N)
T_final_A = normalize(sum_j w_A,j * T_j_A)
```

### Why No W_V

Do not use a value projection:

```text
V_N = [T1_N, T2_N, T3_N]
V_A = [T1_A, T2_A, T3_A]
```

Reason:

```text
The final text descriptors are used for cosine/dot similarity with visual
patch features. A W_V projection may move text descriptors out of the CLIP
embedding space and damage zero-shot transfer.
```

Use `W_Q` and `W_K` only to compute compatibility scores. Keep values
unprojected.

### Tensor Shapes

Input tensors:

```text
vision_tokens: [n_groups, B, L, 768]
text_features: [n_groups, B, 768, 2]
```

For stage `i`:

```text
img_feat = vision_tokens[i]                 # [B, L, 768]
v_global = img_feat.mean(dim=1)             # [B, 768], no keepdim
group_text = text_features.permute(1,0,2,3) # [B, n_groups, 768, 2]
T_N = group_text[..., 0]                    # [B, n_groups, 768]
T_A = group_text[..., 1]                    # [B, n_groups, 768]
```

Attention:

```text
Q_v = W_Q(v_global)                         # [B, d_attn]
K_N = W_K(T_N)                              # [B, n_groups, d_attn]
K_A = W_K(T_A)                              # [B, n_groups, d_attn]
scores_N = einsum("bd,bnd->bn", Q_v, K_N)
scores_A = einsum("bd,bnd->bn", Q_v, K_A)
```

Outputs:

```text
T_final_N: [B, 768]
T_final_A: [B, 768]
group_text_features = stack([T_final_N, T_final_A], dim=-1) # [B, 768, 2]
fused_feature = matmul(img_feat_scaled, group_text_features)
```

### Hyperparameters

Initial V1 Phase 1A settings, historical:

```text
d_attn = 256
tau_fusion = 4.0 fixed
W_Q = Linear(768 -> 256)
W_K = Linear(768 -> 256)
W_V = none
normal/abnormal softmax = separate
```

Why these settings:

```text
d_attn=256 reduces capacity and overfit risk versus 768.
tau_fusion=4.0 keeps attention soft early and avoids hard text-level selection.
fixed tau_fusion keeps the first ablation clean.
```

Current A1 anchor for Phase 1B:

```text
d_attn = 256
tau_fusion = 8.0 fixed
reason: V2 tau=8 outperformed V1 tau=4 and A0 local baseline on mean
medical pixel-level and image-level metrics.
```

Later ablations only if Phase 1A is stable:

```text
tau_fusion = 2.0 / 4.0 / 8.0
trainable tau_fusion bounded to [1, 10]
d_attn = 128 / 256 / 768
```

Do not add MLP-attention residual mixing in the first run. If attention fails
badly, a rescue ablation can be:

```text
weights = (1 - lambda) * weights_mlp + lambda * weights_attn
lambda initialized near 0 and bounded to [0, 1]
```

This is not the main Phase 1A design because it makes attribution harder.

## Phase 1B: SS2D-Style Spatial Context Residual

### Motivation

The original DFG uses:

```text
v_gap = GAP(V_i)
```

Direct GAP can discard spatial relations before text fusion. Phase 1B adds an
SS2D-style branch so patch tokens exchange 2D spatial context before pooling.

### Architecture

For each stage `i`:

```text
V_i patch tokens [B, L, 768]
-> reshape [B, H, W, 768]
-> LayerNorm(768)
-> Linear(768 -> 768)
-> SiLU
-> lightweight four-direction SS2D-style scan
-> LayerNorm(768)
-> GAP over H,W
-> v_ss2d [B, 768]
```

Residual fusion:

```text
v_global = v_gap + gamma * v_ss2d
gamma = 0.2 * tanh(raw_gamma)
raw_gamma init = 0
```

At initialization:

```text
gamma = 0
v_global = v_gap
```

So the first step preserves the original DFG behavior.

### Shape And Layout Rules

Use patch tokens only:

```text
Current `seg_tokens` are already patch-only: [B, 1369, 768].
If using a tensor that still contains CLS, remove CLS before reshape.
```

For `img_size=518` and patch size 14:

```text
L = 1369
H = W = 37
```

LayerNorm rule:

```text
LayerNorm(768) is correct on [B, H, W, C].
Do not apply LayerNorm(768) directly to [B, C, H, W].
```

If the SS2D implementation expects channels-first:

```text
[B,H,W,C] -> LN -> Linear -> SiLU -> permute [B,C,H,W]
-> SS2D -> permute [B,H,W,C] -> LN -> GAP
```

### Hyperparameters

Initial Phase 1B settings:

```text
SS2D-style branch dim = 768
Linear = 768 -> 768
activation = SiLU
gamma_max = 0.2
gamma = 0.2 * tanh(raw_gamma)
raw_gamma init = 0
```

Why:

```text
768 -> 768 avoids dimension mismatch and extra capacity.
zero-init gamma preserves baseline behavior at step 0.
bounded gamma prevents SS2D from dominating GAP under VisA training.
gamma_max=0.2 is a conservative residual bound, heuristically aligned with
the small adapter intervention scale used in ACD-CLIP.
```

Later ablations only if Phase 1B is stable:

```text
gamma_max = 0.1 / 0.2 / 0.3
Linear 768 -> 1536 -> 768
add Mamba-style gate branch
```

Do not start with expansion or extra gating because it adds attribution noise.

## Segmentation Similarity

Keep the current ACD-CLIP mainbase segmentation logit scale in Phase 1.

Current code uses:

```python
img_feat = 10 * img_feat
fused_feature = torch.matmul(img_feat, group_text_features)
```

Do not introduce a new `tau_seg` in Phase 1. If paper-level notation mentions a
segmentation temperature, treat it as separate from `tau_fusion`.

Naming:

```text
tau_fusion: attention temperature for text-level selection
tau_seg: segmentation similarity temperature, not changed in Phase 1
```

Normalize before similarity:

```text
Use the same normalization behavior as the current mainbase. In the current
mainbase, `seg_tokens` are already normalized before DFG. If a future change
bypasses that path, explicitly normalize both visual and text descriptors.

img_feat = normalize(img_feat, dim=-1)       # only if not already normalized
T_final_N = normalize(T_final_N, dim=-1)
T_final_A = normalize(T_final_A, dim=-1)
```

Then use the same logit scale as the current code:

```text
img_feat_scaled = 10 * img_feat
```

## Training Setup

Keep baseline hyperparameters:

```text
optimizer = Adam
image_lr = 0.001
text_lr = 0.0005
StepLR step_size = 1
StepLR gamma = 0.9
epoch = 20
batch_size = 6
n_groups = 3
img_size = 518
image_adapt_weight = 0.2
text_adapt_weight = 0.2
conv_lora_rank = 8
conv_lora_alpha = 2.0
lora_rank = 16
lora_alpha = 2.0
conv_kernel_size_list = [3, 5]
```

Runtime settings used for local reproducibility:

```text
AMP = true
grad_checkpointing = true
num_workers = 6
CLIP backbone frozen
eps-safe adapter normalization = true
finite loss/grad guard = true
clip_grad_norm all adapters = 1.0
```

Do not change LR/scheduler in Phase 1. Changing optimizer settings would make
it hard to attribute changes to DFG attention or SS2D.

## Implementation Checklist

### Common

Add CLI/config options:

```text
--dfg_mode mlp|attn
--use_ss2d_dfg
--dfg_attn_dim 256
--dfg_attn_tau <explicitly set; use 8.0 for current Phase 1B>
--dfg_gamma_max 0.2
```

Default behavior must reproduce current mainbase:

```text
dfg_mode = mlp
use_ss2d_dfg = false
```

### Phase 1A

In `ACDCLIP.__init__`:

```text
if dfg_mode == "attn":
    add ModuleList W_Q under image_adapter, one per group
    add ModuleList W_K under image_adapter, one per group
    remove or bypass vision_text_gate only in the attention path
```

Do not create Phase 1 attention modules when `dfg_mode="mlp"` in the first
implementation. This keeps the default model closest to mainbase and avoids
baseline checkpoint key mismatches.

In `vision_text_fusion_gate_seg`:

```text
if dfg_mode == "mlp": use current code path
if dfg_mode == "attn": use dual-softmax attention path
```

Log optional diagnostics:

```text
attention entropy normal
attention entropy abnormal
mean/max attention weight per group
```

### Phase 1B

In `ACDCLIP.__init__`:

```text
if dfg_mode == "attn" and use_ss2d_dfg:
    add SS2D-style residual modules under image_adapter, one per group
    add raw_gamma parameters under image_adapter, one scalar or one per group
```

In DFG attention path:

```text
v_gap = img_feat.mean(dim=1)  # [B, 768]
if use_ss2d_dfg:
    v_ss2d = ss2d_branch(img_feat)
    gamma = gamma_max * tanh(raw_gamma)
    v_global = v_gap + gamma * v_ss2d
else:
    v_global = v_gap
```

Use SS2D only with the attention path in the first implementation:

```text
Phase 1B = dfg_mode="attn" + use_ss2d_dfg=true
```

Do not attach SS2D to the old MLP gate unless doing a separate ablation.

### Optimizer And Checkpoint Requirements

Any new Phase 1 parameters must be registered and preserved:

```text
W_Q modules
W_K modules
SS2D-style branch modules
raw_gamma parameters
```

Implementation requirements:

```text
1. Register all new modules under `model.image_adapter`.
2. Ensure they are included in `model.image_adapter.parameters()`.
3. Ensure `adapter_*.pth` saves them through `image_adapter.state_dict()`.
4. Ensure test/eval model construction creates the same modules before loading.
5. Ensure `test.py` exposes the same flags (`--dfg_mode`, `--use_ss2d_dfg`, etc.).
```

If a module is not in the checkpoint, training may appear to work while
evaluation silently uses random/default weights.

Checkpoint compatibility rule:

```text
Phase 1 uses conditional module creation.

dfg_mode="mlp":
    do not create W_Q, W_K, SS2D, or raw_gamma
    baseline checkpoints should load with the same keys as mainbase

dfg_mode="attn":
    create W_Q/W_K before loading the adapter checkpoint
    also create SS2D/raw_gamma before loading when use_ss2d_dfg=true
```

Avoid relying on `strict=False` as the default Phase 1 strategy. It can hide a
mistake where train saves attention/SS2D weights but test constructs the wrong
model variant and silently skips them.

## Experiment Folders

Suggested names:

```text
test_train_main_base
phase1a_attn_dfg_visa_n3
phase1b_attn_ss2d_dfg_visa_n3
```

Train command template:

```bash
conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --batch_size 6 \
  --epoch 20 \
  --amp \
  --grad_checkpointing \
  --num_workers 6 \
  --save_path <experiment_folder>
```

Phase 1A additions:

```bash
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0
```

Phase 1B additions:

```bash
  --dfg_mode attn \
  --use_ss2d_dfg \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --dfg_gamma_max 0.2
```

## Evaluation Protocol

Evaluate the six medical datasets:

```text
Brain
Liver
Retina
Colon_clinicDB
Colon_colonDB
Colon_Kvasir
```

Primary metrics:

```text
pixel AUROC
pixel AP
mean pixel AUROC/AP across six datasets
```

Secondary image-level metrics:

```text
Brain image AUROC/AP
Liver image AUROC/AP
Retina image AUROC/AP
```

Do not claim improvement unless:

```text
mean pixel AUROC and mean pixel AP beat the reproduced mainbase baseline
under comparable test settings.
```

Keep TTA/no-TTA explicit:

```text
Never infer TTA from folder names.
Always inspect args/logs.
```

## Expected Ablation Table

Minimum table:

```text
A0: reproduced ACD-CLIP mainbase, N=3
A1: A0 + dual-softmax attention DFG
A2: A1 + SS2D residual context
```

Do not add DoRA until Phase 1 is understood.

Phase 2 starts only after:

```text
Phase 1A has a valid train/test result
Phase 1B has a valid train/test result
medical transfer result is analyzed, not just VisA train loss
```

## Stop Conditions

Stop and debug if:

```text
loss becomes NaN/Inf repeatedly
attention entropy collapses to near 0 in early epochs
gamma saturates near +/- gamma_max early
medical mean pixel AP drops heavily while VisA loss improves
image-level scores improve only because max-pixel score dominates pcls
A1 improves image-level AUROC while pixel AP drops on most medical datasets
```

If Phase 1A underperforms badly:

```text
try tau_fusion=8.0
inspect normal/abnormal attention entropy
try MLP-attention residual lambda as a separate rescue ablation
```

If Phase 1B underperforms badly:

```text
reduce gamma_max to 0.1
inspect v_gap norm vs v_ss2d norm
keep attention-only A1 as the candidate for Phase 2
```

## Summary

Phase 1 changes only the DFG:

```text
Phase 1A replaces visual-only MLP text-level gating with dual-softmax
visual-text attention while keeping text values in CLIP space.

Phase 1B adds a zero-initialized, bounded SS2D-style spatial residual before
attention to enrich v_global without replacing GAP.
```

The design is intentionally conservative for zero-shot transfer from VisA
industrial anomalies to medical anomalies.

## Current A1 Anchor Before Phase 1B

After the tau sweep, Phase 1A tau=8 is the current A1 anchor.

```text
variant: V2 / Phase 1A pure dual-softmax attention DFG
folder: phase1_v2_attn_tau8
architecture:
  dfg_mode = attn
  W_Q/W_K = Linear(768 -> 256), bias=False
  W_V = none
  normal/abnormal attention = separate softmax
  dfg_attn_tau = 8.0 fixed
  SS2D-style branch = none
  DoRA = none
  dynamic/depthwise Conv-LoRA = none
  few-shot = none
best pixel checkpoint: adapter_13.pth
best image checkpoint: adapter_10.pth
```

Current local comparison against A0 baseline epoch 19:

```text
Medical pixel-level mean:
  A0 e19:    88.51 / 34.05
  A1 V2 e13: 89.63 / 34.70
  delta:    +1.12 / +0.65

Medical image-level mean:
  A0 e19:    71.70 / 72.65
  A1 V2 e10: 72.65 / 73.58
  delta:    +0.95 / +0.93
```

This is the rollback anchor:

```text
If Phase 1B is weaker, keep Phase 1A V2 tau=8 as the accepted Phase 1A result.
Do not delete phase1_v2_attn_tau8.
Do not overwrite adapter_13.pth or adapter_10.pth.
```

## Phase 1B Implementation Spec From Current A1

Phase 1B should change only the visual query used by attention DFG.

Keep from A1:

```text
dfg_mode = attn
dfg_attn_dim = 256
dfg_attn_tau = 8.0
dual-softmax normal/abnormal attention
no W_V
T_final_N/A normalization
existing img_feat scale: img_feat = 10 * img_feat
all optimizer/LR/scheduler/runtime settings
```

Add in Phase 1B:

```text
use_ss2d_dfg = true
dfg_gamma_max = 0.2
raw_gamma init = 0
SS2D-style residual branch per visual stage/group under image_adapter
```

The only mathematical change:

```text
A1:
  v_global = GAP(V_i)

A2 / Phase 1B:
  v_gap = GAP(V_i)
  v_ss2d = GAP(LN(SS2D_style_scan(SiLU(Linear(LN(V_i_2d))))))
  gamma = 0.2 * tanh(raw_gamma)
  v_global = v_gap + gamma * v_ss2d
```

At initialization:

```text
raw_gamma = 0
gamma = 0
v_global = v_gap
```

Therefore Phase 1B starts from the same DFG query as A1 and lets the SS2D
branch gradually contribute only if VisA training finds it useful.

Suggested names:

```text
folder: phase1_v3_attn_tau8_ss2d_g02
train script: train_phase1_v3_attn_tau8_ss2d_g02_n3.sh
test script: test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh
```

Expected train command:

```bash
conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --batch_size 6 \
  --epoch 20 \
  --amp \
  --grad_checkpointing \
  --num_workers 6 \
  --save_path phase1_v3_attn_tau8_ss2d_g02
```

Checkpoint metadata must include:

```text
dfg_mode
dfg_attn_dim
dfg_attn_tau
use_ss2d_dfg
dfg_gamma_max
```

V3 logging checklist:

```text
gamma per stage
raw_gamma per stage
||gamma * v_ss2d|| / ||v_gap||
attention entropy normal/abnormal
mean attention weight per group
non-finite gradient skip count
```

Interpretation:

```text
If gamma stays near 0, the SS2D-style branch is effectively not contributing and V3 is close to V2.
If gamma saturates near +/-0.2 early, the SS2D-style branch may dominate the query too strongly.
If VisA train loss improves but medical AP drops, rollback to V2 tau8.
```

Rollback/acceptance rule:

```text
If Phase 1B mean pixel AP and image AP are both lower than A1 V2:
  rollback to A1 V2 tau8.

If Phase 1B improves one metric group but hurts the other:
  keep both A1 and A2 as ablations, and report the trade-off.

If Phase 1B improves mean pixel AP or recovers AP on Brain/Liver/Retina without
destroying Colon/Kvasir gains:
  keep Phase 1B as the next candidate for Phase 2.
```

## Phase 1A Implementation Note

Implemented in `ACD-CLIP-base-new-phase1` as an explicit DFG mode:

```bash
--dfg_mode mlp   # original ACD-CLIP DFG, default
--dfg_mode attn  # Phase 1A dual-softmax attention DFG
```

Phase 1A changes:

```text
model/adapter.py:
  add ACDCLIP args: dfg_mode, dfg_attn_dim, dfg_attn_tau
  create original vision_text_gate only when dfg_mode="mlp"
  create vision_text_q and vision_text_k only when dfg_mode="attn"
  V1 historical run used dfg_attn_dim=256 and dfg_attn_tau=4.0
  current A1 anchor for Phase 1B uses dfg_attn_dim=256 and dfg_attn_tau=8.0
  initialize W_Q/W_K with Xavier for dfg_attn_dim=256
  replace MLP text fusion with dual-softmax attention when dfg_mode="attn"
  keep W_V absent; values remain original CLIP text features
  normalize fused T_final_N/T_final_A before existing segmentation matmul
  keep existing img_feat scale: img_feat = 10 * img_feat

train.py:
  add CLI args for dfg_mode/dfg_attn_dim/dfg_attn_tau
  pass them into ACDCLIP
  save DFG metadata in checkpoints

test.py:
  add matching CLI args
  pass them into ACDCLIP
  check checkpoint dfg_mode, n_groups, dfg_attn_dim, and dfg_attn_tau
  against eval args to avoid wrong eval mode

test_6medical_selected_epochs.sh:
  add env vars DFG_MODE, DFG_ATTN_DIM, DFG_ATTN_TAU
```

Historical V1 Phase 1A train command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
bash train_phase1_v1_attn_tau4_n3.sh
```

The script expands to:

```bash
conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 4.0 \
  --batch_size "${BATCH_SIZE:-6}" \
  --epoch "${EPOCH:-20}" \
  --amp \
  --grad_checkpointing \
  --num_workers "${NUM_WORKERS:-6}" \
  --save_path "${SAVE_PATH:-phase1_v1_attn_tau4}"
```

Phase 1A eval command using the same local baseline protocol:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
DFG_MODE=attn DFG_ATTN_DIM=256 DFG_ATTN_TAU=4.0 \
METRIC_THRESHOLDS=none PIXEL_STRIDE=4 SAVE_PATH=phase1_v1_attn_tau4 \
  bash test_6medical_selected_epochs.sh 19
python parse_test_log.py --log phase1_v1_attn_tau4/test.log
```

The `dfg_attn_dim=768`, `dfg_attn_tau=1.0` setting is only a debug/identity-init
variant and should not be used as the main A1 comparison.

Keep baseline comparison fixed to:

```text
A0 local baseline: ACD-CLIP-mainbase-newest, N=3, epoch 19,
no threshold, pixel_stride=4
mean pixel AUC/AP over six medical datasets = 88.51 / 34.05
```
