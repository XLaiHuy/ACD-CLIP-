# Gemini Prompt — Fast P1-v7 Audit, Root-Cause Decision Tree, and Minimal P1-v8 Fix

Repository:

```text
/home/ai4/caohuy/ACD-CLIP-phase4
```

Branch/source:

```text
phase4-progress1-cops-dynamic-prompt
```

Current completed run:

```text
runs/phase4/progress1_v7_full_seed0_ready3/
checkpoint:
runs/phase4/progress1_v7_full_seed0_ready3/train/adapter_12.pth
```

Use Gemini 3.1 Pro for audit, diagnosis, and architecture decisions.
Use Gemini 3.6 Flash only for repetitive test-fix loops after the design is fixed.

---

# 1. Goal

Find the real reason P1-v7 loses medical Pixel AP, while minimizing GPU time.

Research hypothesis:

```text
each patch should select the semantic factor that best explains that patch
```

Do not continue to Progress 2 until Progress 1:

```text
- passes metric parity;
- matches or beats a protocol-matched Phase2B baseline;
- has no functional factor/router/expert collapse.
```

---

# 2. Hard safety rules

Do not run:

```text
git add
git commit
git push
git reset
git clean
git restore
```

Do not overwrite or delete existing P1-v7 artifacts.

Do not load Phase2B or previous-progress checkpoints to initialize a new training run.

New training must start from:

```text
OpenAI pretrained CLIP
+
newly initialized Phase2B/H6 trainable modules
```

Do not launch:

```text
full 20-epoch training
exact medical test
```

unless explicitly enabled with:

```text
ALLOW_FULL_TRAIN=1
ALLOW_EXACT_TEST=1
```

Do not tune architecture on exact medical test data.

---

# 3. Strict time budget

Default maximum GPU work:

```text
Stage A — static audit and metric parity:
no training

Stage B — runtime profiling:
<= 150 measured training batches total

Stage C — inference triage:
small deterministic validation subset first

Stage D — wiring smoke:
1 epoch × 50 batches

Stage E — structural smoke:
maximum 3 epochs × 300 batches
hard abort as soon as collapse is confirmed

Stage F — full-data speed verification:
maximum 1 epoch

Do not automatically run 8 or 20 epochs.
```

Target after safe optimization:

```text
full epoch <= 12 minutes if hardware and VRAM permit
```

Do not sacrifice metric semantics merely to reach the target.

---


# 3A. NON-REGRESSION GUARANTEE FOR SPEED OPTIMIZATION

Runtime optimization is secondary to correctness and benchmark quality.

No speed change may be kept merely because it is faster.

A candidate optimization is accepted only when it passes all applicable gates below.

## Invariant configuration

The following must remain unchanged for final training/evaluation:

```text
model architecture and trainable parameter set
OpenAI CLIP initialization
dataset and full training samples
image resolution = 518
augmentations
seed
epoch count
effective batch size
number of optimizer updates per epoch
optimizer and parameter groups
learning rates
scheduler and scheduler-step semantics
loss definitions and loss weights
alpha/router/expert curricula
gradient clipping
metric definitions
Gaussian blur and post-processing
validation/test manifests
checkpoint-selection rule
```

Do not reduce final runtime by:

```text
using fewer training samples
lowering final image resolution
removing losses or modules
skipping optimizer steps
shortening the full training schedule
changing validation/test data
using approximate metrics
removing required forward branches
silently changing numerical precision
```

## Safe-by-design candidates

These may be optimized first because they should not change model mathematics:

```text
DataLoader workers, pin_memory, persistent_workers, prefetch_factor
non-blocking host-to-device transfers
caching tokenized static prompts and frozen anchors
reducing diagnostic frequency
moving expensive diagnostics outside the critical training path
reducing temporary checkpoint frequency
reusing cached predictions for metric-only recomputation
```

Even these changes must pass deterministic sample/order and output checks.

## Conditional candidates

These can alter numerical execution and must pass stronger equivalence tests:

```text
turning gradient checkpointing off
changing micro-batch size / gradient accumulation
torch.compile
changing autocast or precision behavior
fused optimizer/kernels
```

Do not keep a conditional candidate unless it passes:

### Gate 1 — forward parity

On the same frozen model and deterministic 30-batch input sequence:

```text
same sample IDs and order
same tensor shapes
all outputs finite
mean/max prediction differences within expected BF16 numerical tolerance
no change in evaluator output beyond 0.01 percentage point
```

### Gate 2 — gradient/update parity

From identical initialization and the same deterministic 30–100 batches:

```text
same optimizer-step count
same LR at every optimizer step
same effective batch
finite gradients
loss trajectory remains within a documented tolerance
gradient norms remain in the same range
no parameter group is skipped
```

Exact bitwise equality is not required under BF16, but meaningful divergence must reject the optimization.

### Gate 3 — one-epoch non-regression check

Run baseline and optimized configurations from identical initialization for one full epoch using the same seed and data order.

Compare on one fixed validation subset:

```text
Pixel AUROC delta >= -0.10 point
Pixel AP delta >= -0.10 point
Image AUROC delta >= -0.10 point where supported
Image AP delta >= -0.10 point where supported
training loss remains finite
router/factor/expert structural diagnostics do not worsen
```

If the difference is larger than the tolerance, reject the speed change.

This one-epoch check is a runtime-equivalence guard, not an architecture-quality conclusion.

### Gate 4 — final reproducibility

Record:

```text
baseline and optimized commands
environment
GPU
PyTorch/CUDA versions
sample-order hash
config fingerprint
runtime profile
parity results
accepted/rejected decision
```

## Micro-batch caution

Changing:

```text
batch_size=1, grad_accum=6
```

to:

```text
batch_size=2, grad_accum=3
or
batch_size=3, grad_accum=2
```

does not automatically guarantee identical training.

It may change:

```text
stochastic operations
batch-dependent reductions
loss averaging
router/load-bias updates
diagnostic and schedule update frequency
```

Only keep a micro-batch change if all state updates are proven to occur at the same semantic frequency and Gates 1–3 pass.

## Decision rule

For every proposed optimization report:

```text
speedup percentage
memory delta
metric/parity delta
ACCEPT or REJECT
reason
```

A speed optimization is accepted only when:

```text
speedup is measurable
AND
correctness parity passes
AND
no metric non-regression gate fails
```

When uncertain, keep the slower baseline.


# 4. Stage A — static audit before GPU work

Inspect:

```text
train.py
test.py
model/adapter.py
model/h6/*
tools/phase4*
scripts/phase4/*
P1-v7 diagnostics epochs 1, 3, 8, 12, 20
Phase2B config and checkpoint context
```

Verify from source/artifacts:

```text
1. global/local alpha schedule;
2. when the router starts receiving gradients;
3. when router teacher starts;
4. when expert assignment, advantage, and ETF start;
5. dense-to-sparse transition;
6. factor-aware center enabled/disabled;
7. state_projection and expert_B initialization;
8. patch-label generation method;
9. exact global and local scoring paths;
10. which diagnostics execute every batch.
```

Produce:

```text
runs/phase4/p1_fast_audit/static_audit.json
```

Do not modify architecture yet.

---

# 5. Stage A2 — metric parity

Use existing saved predictions when available.

Otherwise run only a tiny deterministic subset.

Compare identical predictions and masks using:

```text
current exact disk-backed metrics
vs
official-style torchmetrics AUROC/AP
```

Verify:

```text
medical Gaussian blur: kernel 9, sigma 1.5
bilinear resize
align_corners=True
mean logits across levels
softmax after aggregation
global pixel flattening
medical image score = 0.5 * cls + 0.5 * max pixel
```

Tolerance:

```text
<= 0.01 percentage point
```

Decision:

```text
if parity fails:
    fix evaluator only
    re-evaluate existing checkpoints
    STOP model changes
```

Save:

```text
runs/phase4/p1_fast_audit/metric_parity.json
```

---

# 6. Stage B — profile why one epoch increased from ~10 to ~20 minutes

Do not guess. Add lightweight timers around:

```text
data loading
visual CLIP forward
Phase2B adapters/DFG
H6 prototype + VAE
dynamic text encoding
router + experts
loss construction
expensive diagnostics
backward
optimizer step
checkpoint writing
```

Profile:

```text
20 warm-up batches
100 measured batches
same seed and batch order
```

Record:

```text
mean and p95 time per component
GPU allocated/reserved/peak memory
GPU utilization when observable
samples/second
optimizer steps/second
```

Save:

```text
runs/phase4/p1_fast_audit/runtime_profile_before.json
```

## Runtime decision tree

### R1. Data loading > 15% of step time

Benchmark only these safe loader variants:

```text
num_workers = 4, 6, 8
pin_memory = True
persistent_workers = True
prefetch_factor = 2 or 4
non_blocking transfer = True
```

Choose the fastest stable configuration.

### R2. Gradient checkpointing dominates and VRAM has >= 25% headroom

Benchmark:

```text
grad_checkpointing=True
vs
grad_checkpointing=False
```

Use the same 30 deterministic batches.

Require:

```text
same finite losses
same output shapes
max prediction difference within BF16 tolerance
no OOM
```

Prefer `False` if materially faster and memory is safe.

Gradient checkpointing saves memory but recomputes forward operations during backward, so it may explain a large part of the 10→20 minute increase.

### R3. Diagnostics consume > 10% of step time

Keep cheap counters every batch, but move expensive operations such as:

```text
SVD/effective-rank diagnostics
large pairwise matrices
gradient attribution
full factor geometry
quantiles
```

to:

```text
first batch
middle probe batch
last batch
```

or a configurable sample interval.

Default full training must use:

```text
h6_factor_grad_diagnostics = False
h6_drift_diagnostics = False
```

unless explicitly debugging.

### R4. Batch-size/accumulation overhead is large

Current effective batch is approximately:

```text
batch_size=1 × grad_accum=6
```

If VRAM permits, benchmark without changing effective batch:

```text
batch_size=2, grad_accum=3
batch_size=3, grad_accum=2
```

Use only a short throughput benchmark first.

Do not change the final effective batch, optimizer step count, or scheduler semantics.

### R5. Repeated text/token work is expensive

Cache only truly static data:

```text
tokenized prompt templates
frozen hard CLIP text anchors
dataset/class metadata
```

Do not cache image-conditioned dynamic text embeddings.

### R6. `torch.compile`

Treat as optional and disabled by default.

Only keep it if:

```text
compile succeeds
no graph-break explosion
same outputs within tolerance
measured epoch time improves after compile warm-up
```

Do not spend more than 10 minutes investigating compile failures.

After safe changes, repeat the same profile and write:

```text
runtime_profile_after.json
runtime_speedup_report.md
```

---

# 7. Stage C — protocol-matched triage before full validation

Create a deterministic triage manifest:

```text
Brain: up to 32 normal + 32 anomaly
Liver: up to 32 normal + 32 anomaly
Retina: up to 32 normal + 32 anomaly
each colon dataset: up to 64 images
```

Preserve masks and class distribution where possible.

Use the exact same triage manifest for:

```text
Phase2B best checkpoint
P1-v7 epoch 12
all inference ablations
```

Do not use triage scores as final benchmark numbers.

## C1. Protocol-matched Phase2B triage

Evaluate Phase2B and P1-v7 on the same triage manifest and evaluator.

Decision:

```text
if Phase2B does not clearly beat P1-v7:
    investigate protocol/split first

if Phase2B beats P1-v7 by >= 2 Pixel AP:
    architecture regression is likely
```

Only run full medical validation for Phase2B if the triage result is stable and architecture attribution requires it.

---

# 8. Stage C2 — ordered inference ablations

Use checkpoint epoch 12.

Run in this order:

```text
B0 full P1-v7
B5 pure Phase2B path inside P1 checkpoint
B3 rho=0 / local residual off
B4 Phase2B global text + H6 local branch
B2 forced dense routing
B1 paired experts off / pre-expert bank
```

Reason for order:

```text
B5 quickly checks shared-adapter regression
B3 checks false positives from local residual
B4 separates global and local changes
B2 checks sparse collapse
B1 checks whether experts matter
```

For each mode save:

```text
Pixel AUROC/AP
Image AUROC/AP where valid
inside-mask mean score
outside-mask mean score
false-positive mass outside mask
top 1%, 5%, 10% pixel precision
predicted anomaly area at 0.1, 0.3, 0.5
runtime
fingerprint
```

Stop rules:

```text
if one ablation restores >= 80% of the Phase2B AP gap:
    mark it as primary cause
    do not run unnecessary secondary combinations

if B1 differs from B0 by < 0.2 Pixel AP:
    experts are functionally inactive

if B2 improves >= 1 Pixel AP:
    sparse routing is harmful

if B3 improves >= 1 Pixel AP and outside-mask mass falls:
    local residual / coarse patch targets are harmful

if B4 improves >= 1 Pixel AP:
    dynamic global text is harmful

if B5 remains weak:
    H6 training losses altered shared Phase2B adapters
```

Only repeat the winning one or two ablations on full validation.

Do not run all ablations on full validation by default.

Save:

```text
p1_v7_triage_ablation.csv
p1_v7_full_validation_confirm.csv
p1_v7_decision.json
```

---

# 9. Architecture decision tree

Do not implement every proposed fix.

Select only branches supported by Stage C.

## Decision M — metric bug

```text
Fix evaluator only.
No retraining.
```

## Decision P — protocol/split issue

```text
Standardize Phase2B and P1 evaluation.
No architecture redesign yet.
```

## Decision S — shared adapter regression

Triggered when pure Phase2B path inside the P1 checkpoint is still weak.

Implement:

```text
staged optimization or gradient isolation;
protect Phase2B task path during early H6 formation;
audit H6 auxiliary-to-shared gradient ratios.
```

## Decision G — dynamic global regression

Triggered when Phase2B global text improves AP.

Implement:

```text
Phase2B global text remains the main branch;
H6 tests only patch-local routed residual.
```

## Decision L — local residual regression

Triggered when rho=0 improves AP and false-positive mass decreases.

Implement:

```text
soft patch anomaly fraction using adaptive_avg_pool2d;
bounded/learnable local fusion;
no binary max-pooled positive label from one anomalous pixel.
```

## Decision R — sparse router regression

Triggered when forced dense improves AP.

Implement:

```text
readiness-based dense-to-sparse transition;
no epoch-only sparse switch;
hard abort or remain dense when Top-K coverage collapses.
```

## Decision E — experts inactive/harmful

Triggered when expert-off is equal or better.

Do not delete FOFS/ETF source immediately.

Implement:

```text
delay expert training until router readiness;
state_projection small non-zero deterministic init;
expert_B remains zero for exact no-op;
function-level expert diagnostics;
disable expert contribution until functional readiness passes.
```

---

# 10. Minimal P1-v8 fixes shared by likely branches

Create:

```text
P1-v8-functional-router
```

Keep P1-v7 behavior unchanged under its version.

Mandatory corrections only when compatible with the selected decision:

```text
1. Router must not specialize while four factor functions are identical.
2. Separate global alpha and local factor exposure.
3. Local factors must be distinguishable before router teacher/assignment starts.
4. Enable factor-aware center with detached assignment.
5. Sparse mode requires readiness, not only epoch number.
6. Keep FOFS and ETF, but evaluate functional output, not only delta geometry.
7. Keep expert_B zero-init.
8. Initialize state_projection small non-zero so first expert gradients depend on CoPS state.
9. Keep Phase2B full-resolution focal+dice unchanged.
10. Soft patch targets apply only to H6 auxiliary losses.
```

Do not add new large modules.

---

# 11. Fast smoke strategy

## Smoke 1 — wiring

Run:

```text
1 epoch
50 batches
no medical validation
```

Check:

```text
finite forward/backward
all intended parameter groups receive gradients
checkpoint save/load
loss component sum
no shape errors
```

Expected duration:

```text
a few minutes
```

## Smoke 2 — structural

Run:

```text
maximum 3 epochs
300 batches per epoch
grad_accum preserving effective batch
no medical validation
```

Probe at:

```text
batch 50
batch 150
batch 300
```

Hard abort immediately if any condition persists across two probes:

```text
unique candidate Top-K pairs <= 1
a factor receives near-zero dense coverage
factor functions remain effectively identical
router teacher has no informative patches
NaN/Inf
state-conditioned expert gradient is absent
```

Do not require sparse routing to activate during this smoke.

Pass criteria:

```text
factor outputs differ
router candidate assignments remain diverse
no dense dead factor
state projection contributes
expert output remains bounded
```

Expected duration after speed fixes:

```text
approximately 10–20 minutes total
```

## Smoke 3 — full-epoch throughput verification

Only after Smoke 2 passes:

```text
run exactly 1 full epoch
measure wall-clock time
no medical validation
```

Target:

```text
<= 12 minutes when hardware permits
```

If still around 20 minutes, report the measured bottleneck rather than weakening training semantics.

Do not automatically continue to more epochs.

---

# 12. How to reduce total experiment time

Implement these workflow optimizations:

```text
- cache metric inputs/predictions for metric-only recomputation;
- use one deterministic triage manifest;
- run full validation only for top two hypotheses;
- abort structural smoke immediately on collapse;
- do not save every temporary checkpoint;
- save only final smoke checkpoint plus failure snapshot;
- do not compute expensive diagnostics every batch;
- do not evaluate every epoch during debugging;
- profile before changing performance-related flags;
- keep exact test completely outside the debug loop.
```

Do not reduce image resolution for final comparisons.

A lower resolution may be used only for a wiring smoke and must not be used to judge architecture quality.

---

# 13. Tests

Add focused tests for:

```text
metric parity
ablation switches
global/local alpha separation
factor-aware center
readiness-based sparse transition
soft patch targets
state_projection non-zero + exact expert no-op
functional expert diagnostics
P1-v7 checkpoint compatibility
P1-v8 checkpoint versioning
runtime diagnostic sampling
```

Run:

```bash
python -m compileall train.py test.py model tools
bash -n scripts/phase4/*.sh
git diff --check
pytest -q <focused test files>
```

---

# 14. Required final output

Report:

```text
1. metric parity result;
2. before/after runtime profile;
3. exact reason epoch time changed from ~10 to ~20 minutes;
4. safe speed changes kept;
5. triage Phase2B vs P1-v7 result;
6. ablation table;
7. selected decision branch;
8. minimal P1-v8 changes;
9. wiring smoke result;
10. structural smoke result;
11. full-epoch measured time;
12. whether an 8-epoch run is justified;
13. files modified;
14. new artifact paths;
15. tests/static checks;
16. git status --short;
17. confirmation no prohibited git command was run;
18. confirmation no exact medical test or full 20-epoch run was launched.
```

Final recommendation must be exactly one of:

```text
FIX_EVALUATOR
STANDARDIZE_PROTOCOL
FIX_SHARED_TRAINING
FIX_GLOBAL_BRANCH
FIX_LOCAL_BRANCH
FIX_ROUTER
DELAY_EXPERTS
READY_FOR_8_EPOCH_VALIDATION
```
