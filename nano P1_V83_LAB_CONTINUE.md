# P1-v8.3 LAB CONTINUATION — SYNC, FIX, DATA/GPU VALIDATION, SMOKE

You are continuing the implementation, audit, and controlled runtime validation of:

**Phase 4 → Progress 1 → candidate P1-v8.3**

on the LAB machine.

This is an implementation + verification session.

Do NOT run the final 20-epoch experiment or the full six-medical final evaluation.

---

# 0. MACHINE CONTEXT

Expected shell:

```text
(torchhuy) ai4@AI4:~/caohuy$
```

Use the existing Conda environment:

```text
torchhuy
```

Repository:

```text
https://github.com/XLaiHuy/ACD-CLIP-
```

Target branch:

```text
phase4-progress1-cops-dynamic-prompt
```

Historical local repo path is likely:

```text
~/caohuy/ACD-CLIP-phase4
```

but DO NOT blindly assume it.

First locate the existing repository under:

```text
~/caohuy
```

and verify its `origin` points to:

```text
XLaiHuy/ACD-CLIP-
```

Do NOT create a duplicate clone unless there is no usable local clone.

The lab machine is expected to already contain the required dataset and likely the
OpenAI CLIP ViT-L/14-336 checkpoint.

Do NOT download Med-VISA/Kaggle data unless explicitly instructed later.

---

# 1. AUTHORITATIVE SPECIFICATION

IMPORTANT:

There is NO `Prompt.txt` in this repository.

Do NOT:

- search for `Prompt.txt`;
- wait for `Prompt.txt`;
- consider missing `Prompt.txt` an error;
- infer the repository is incomplete because it does not exist.

For THIS session, **THIS FILE is the authoritative P1-v8.3 specification**.

Priority of truth:

```text
1. THIS FILE
   = authoritative target specification.

2. origin/phase4-progress1-cops-dynamic-prompt
   = latest implementation state.

3. P1_V83_IMPLEMENTATION.md
   = status summary only.

4. historical docs / runs / comments
   = evidence/context only.
```

If current source conflicts with this file:

1. identify the discrepancy;
2. determine whether it is a real implementation/config bug;
3. make the smallest correct change;
4. add/update a test;
5. record the result.

Do not preserve legacy behavior simply because old `h6` code implements it.

---

# 2. TERMINOLOGY — DO NOT CONFUSE VERSION NAMES

Architecture:

```text
Phase 4 → Progress 1 → candidate P1-v8.3
```

Do NOT call the architecture itself:

```text
H6
```

`h6` remains only a legacy internal code namespace / class / CLI prefix, e.g.:

```text
model/h6/*
H6Progress1
h6_*
```

Do not unnecessarily rename those internals during this task.

---

# 3. TRAINING INITIALIZATION CONTRACT

This point is CRITICAL.

Canonical P1-v8.3 training is:

```text
OpenAI CLIP pretrained
        ↓
ONE end-to-end P1-v8.3 training run
        ↓
epoch 1 ... epoch 20 checkpoints
        ↓
canonical epoch-20 checkpoint
        ↓
frozen six-medical exact evaluation
```

There is NO:

```text
load Phase2B checkpoint
→ continue/fine-tune Phase4
```

“Phase2B-style” means architecture/mechanism only:

```text
DFG + SS2D base/generalist branch
```

It does NOT mean inherited Phase2B weights.

Base/generalist branch and Phase4 local semantic branch train jointly in the same
P1-v8.3 run from OpenAI CLIP initialization.

Do not add code, comments, launcher arguments, or documentation implying a Phase2B
checkpoint is loaded.

---

# 4. NON-DESTRUCTIVE MACHINE POLICY

Do NOT run destructive repository/data commands such as:

```text
git reset --hard
git clean
rm -rf data
rm -rf runs
rm -rf ckpt
```

Do not overwrite:

```text
data/
runs/
checkpoints/
existing weights/
historical experiment artifacts/
```

Do not:

- reinstall NVIDIA drivers;
- reinstall CUDA globally;
- change GPU clocks;
- modify power limits;
- modify MIG;
- kill another user's processes;
- upgrade system packages;
- use `sudo` unless a proven prerequisite specifically requires it.

Do not use KaggleHub or Kaggle CLI in this session unless existing lab data is
proven absent and the user explicitly approves downloading later.

---

# 5. EXECUTION / TIMEOUT POLICY

Avoid busy polling.

For long commands:

- launch once;
- give a generous timeout;
- do not repeatedly `ps`, `tail`, `ls`, or poll every few seconds.

Suggested timeout classes:

```text
static/unit tests          ~1200 s max
dataset integrity          ~1200 s
archive/file inspection    ~1200 s
small GPU smoke            ~1800 s
optional 32–64 diagnostics ~3600 s
```

If Codex automatic approval/reviewer repeatedly stalls on a read-only command:

- do not retry the same large grep indefinitely;
- use smaller targeted commands;
- preserve completed work;
- continue from current state.

Do not restart the whole audit because one reviewer command times out.

Keep terminal narration concise.

Store detailed evidence under:

```text
runs/p1_v83_dev/
```

---

# 6. FIRST TASK — LOCATE AND UPDATE THE LAB REPOSITORY

Start from:

```bash
cd ~/caohuy
```

Locate candidate Git repos at shallow depth.

Identify the clone whose origin is:

```text
https://github.com/XLaiHuy/ACD-CLIP-
```

Enter it.

Before changing anything, record:

```bash
pwd
hostname
date
which python
python --version

git remote -v
git branch --show-current
git status --short
git rev-parse HEAD
git log --oneline --decorate -8
```

Create local evidence directory:

```text
runs/p1_v83_dev/lab_sync/
```

Do not commit this runs directory.

Save useful pre-sync information there.

---

# 7. SAFE GIT SYNC

Remote target:

```text
origin/phase4-progress1-cops-dynamic-prompt
```

Fetch first:

```bash
git fetch origin phase4-progress1-cops-dynamic-prompt
```

Inspect divergence:

```bash
git log --oneline --decorate \
  HEAD..origin/phase4-progress1-cops-dynamic-prompt
```

and:

```bash
git log --oneline --decorate \
  origin/phase4-progress1-cops-dynamic-prompt..HEAD
```

The remote branch should contain the recent P1-v8.3 checkpoint commits, including
the structured-prompt/utility-routing implementation and no-data tests.

Do NOT assume a specific SHA is still latest after fetch.

The remote HEAD after fetch is authoritative.

## Dirty worktree policy

If only untracked:

```text
data/
runs/
checkpoints/
local outputs/
```

exist:

leave them untouched.

If TRACKED source files have local changes:

1. save:

```bash
git diff
git diff --cached
```

to `runs/p1_v83_dev/lab_sync/`;

2. stash TRACKED changes only;

3. do NOT stash untracked data/runs;

4. do NOT automatically pop the stash after pulling.

Example conceptually:

```bash
git stash push -m "pre-p1-v83-lab-sync tracked changes"
```

without `-u`.

Then switch/update:

```bash
git checkout phase4-progress1-cops-dynamic-prompt

git pull --ff-only \
  origin phase4-progress1-cops-dynamic-prompt
```

Never use a destructive reset merely to make pull work.

If the branch cannot fast-forward safely:

STOP the sync step and report the divergence.

After successful pull record:

```bash
git rev-parse HEAD
git log --oneline --decorate -8
git status --short
```

All subsequent source work MUST begin from this newly pulled HEAD.

---

# 8. INSPECT CURRENT PULLED P1-v8.3 IMPLEMENTATION

Read:

```text
P1_V83_IMPLEMENTATION.md
```

if present.

It is only status documentation.

Then inspect the CURRENT pulled versions of:

```text
dataset/info.py

model/adapter.py
model/checkpoint_utils.py
model/clip.py

model/h6/model.py
model/h6/semantic_bank.py
model/h6/utility_routing.py

utils.py
train.py
test.py

tests/test_p1_v83_runtime.py
tests/test_p1_v83_structured_utility.py
tests/test_setup_med_visa_data.py

tools/setup_med_visa_data.py
```

Do not restart an enormous historical source audit.

Focus on:

1. known correctness bugs below;
2. critical runtime contracts;
3. real-data/GPU readiness.

---

# 9. KNOWN BUG — FIX G_local

Authoritative definition:

```text
G_local =
    (BaseLoss - OracleMultiLoss)
    /
    BaseLoss
```

This answers:

> If we could choose the best factor per patch, does the local factor mechanism
> provide any task value over Base at all?

The previously reviewed implementation appeared to compute approximately:

```text
Base - BestSingle
```

which is incorrect.

Fix it.

Use safe denominator handling.

`G_local` must be dimensionless.

Do not silently substitute BestSingle.

---

# 10. KNOWN BUG — FIX G_multi

Authoritative definition:

```text
G_multi =
    (BestSingleLoss - OracleMultiLoss)
    /
    BaseLoss
```

The previously reviewed implementation appeared to omit division by BaseLoss.

Fix it.

Engineering interpretation:

```text
G_multi < 0.02
    weak multi-mode evidence

0.02 <= G_multi <= 0.05
    borderline

G_multi > 0.05
    meaningful engineering evidence
```

These are engineering gates, NOT theoretical claims.

Do not hard-code them as scientific truth.

---

# 11. KNOWN BUG — FIX ROUTER CAPTURE

Authoritative definition:

```text
capture =
    (UniformLoss - SoftRoutedLoss)
    /
    (UniformLoss - OracleMultiLoss)
```

The previously reviewed implementation appeared to use Base instead of Uniform.

Fix it.

Also make denominator validity explicit.

Conceptually:

```text
capture_denominator =
    UniformLoss - OracleMultiLoss
```

Only treat capture as meaningful when the denominator is sufficiently positive.

Record diagnostics such as:

```text
capture
capture_denominator
capture_valid
```

If denominator is zero/tiny/non-positive:

do NOT silently report a huge meaningless value.

Return a safe diagnostic representation and mark:

```text
capture_valid = false
```

---

# 12. ADD EXACT FORMULA UNIT TESTS

Current tests should not merely assert:

```text
OracleMulti <= BestSingle <= Base
```

Add deterministic numerical tests that explicitly check:

```text
G_local exact formula
G_multi exact formula
capture exact formula
capture invalid-denominator behavior
```

Construct synthetic numbers where the old implementation would fail.

The tests must prove the equations, not just ordering.

---

# 13. PATCH TARGET / UTILITY CONTRACT

Utility supervision exists ONLY from VisA training masks.

Do NOT use medical GT during training.

Create patch supervision using area coverage / average pooling.

Concept:

```text
y_patch ∈ [0,1]
```

Use a valid-pixel/valid-area mask so augmentation-created invalid areas do not
contribute to utility supervision.

Exclude invalid/padded patches.

---

# 14. BASE MARGIN / UTILITY EQUATIONS

Reuse the already computed base DFG+SS2D abnormal-minus-normal margin.

Do NOT run a second base branch.

For each valid patch:

```text
z0 =
    base abnormal logit - base normal logit
```

Utility base loss:

```text
L0 =
    BCEWithLogits(stopgrad(z0), y_patch)
```

For factor `m`:

```text
l_m =
    10 *
    (
      similarity(patch, A_m)
      -
      similarity(patch, N_m)
    )
```

Then:

```text
z_m =
    stopgrad(z0)
    +
    rho * l_m
```

and:

```text
L_m =
    BCEWithLogits(z_m, y_patch)
```

Relative gain:

```text
gain_rel_m =
    (L0 - L_m)
    /
    clamp_min(L0, utility_denominator_floor)
```

Canonical initial denominator floor:

```text
0.10
```

Make actual runtime value configurable and persisted.

---

# 15. UTILITY TEACHER

Use:

```text
q_m =
    softmax(
        stopgrad(gain_rel_m)
        /
        tau_utility
    )
```

Canonical initial:

```text
tau_utility = 0.05
```

Responsibility:

```text
r_m =
    (1 - epsilon) * q_m
    +
    epsilon / M
```

with:

```text
M = 4
```

Exploration schedule:

```text
epsilon:
0.15 → 0.05
```

Factor loss:

```text
L_factor_patch =
    sum_m stopgrad(r_m) * L_m
```

Do NOT factor-balance.

In particular, do NOT force:

```text
F1 = 25%
F2 = 25%
F3 = 25%
F4 = 25%
```

---

# 16. NORMAL / ANOMALY EVIDENCE BALANCING

Balance semantic evidence, not factor usage.

For an anomaly-containing image approximately use:

```text
0.5 * mean(valid normal-region loss)
+
0.5 * mean(valid anomaly-region loss)
```

For a fully normal image:

```text
mean(valid normal-region loss)
```

Handle empty regions safely.

---

# 17. INFORMATIVE UTILITY TEACHER

Router teacher uses detached task utility.

Initial informative condition:

```text
best_gain_rel > 0.02
```

AND:

```text
normalized_entropy(q_utility) < 0.98
```

Make actual thresholds configurable and logged.

Record:

```text
informative fraction
all-harm fraction
best-second utility margin
teacher entropy
teacher max probability
winner shares F1/F2/F3/F4
```

---

# 18. ROUTER CONTRACT

Reuse the existing PatchRouter initially.

Main P1-v8.3 routing:

```text
DENSE
```

The router answers:

```text
Which latent factor should contribute to this patch?
```

Main P1-v8.3 must NOT use:

```text
Top-K prediction routing
load bias
equal routing balance
paired experts
cluster responsibility
```

Sparse/Top-K tensors may remain for legacy compatibility or diagnostics, but they
must NOT influence canonical P1-v8.3 prediction.

Router loss should match dense probabilities to the detached utility teacher on
informative valid patches.

Do not redesign the router unless runtime evidence demonstrates a router-specific
failure.

---

# 19. NO SELECTIVE USE GATE IN MAIN P1-v8.3

Do NOT add a use/no-op gate to the main P1-v8.3 architecture.

Canonical safety mechanism is:

```text
rho = 0.05 fixed
```

An optional future selective-use branch is allowed only after real runtime evidence
shows it is needed.

Do not proactively implement it.

---

# 20. MAIN LOSS CONTRACT

Main P1-v8.3 conceptually:

```text
L_total =
    L_task
    +
    lambda_factor * L_factor
    +
    lambda_router * L_router
    +
    necessary VAE/trust terms
```

Legacy experimental auxiliaries must be OFF in the canonical main path:

```text
hard semantic role losses
cluster responsibility
equal routing balance
load bias
paired experts
Top-K prediction
functional diversity
trainable rho
selective use gate
```

Do not accidentally activate old losses through stale nonzero defaults.

Audit the remaining VAE/trust terms separately.

Do not remove a necessary trust term merely because it is not one of the two new
utility losses.

---

# 21. GRADIENT ATTRIBUTION — COMPLETE THE MISSING P1-v8.3 AUDIT

Do NOT choose lambda values from raw loss magnitudes alone.

Reuse/extend existing gradient attribution machinery.

At a controlled diagnostic batch, perform a NO-OPTIMIZER-STEP gradient attribution
probe for at least:

```text
main task
factor utility auxiliary
router utility auxiliary
```

Inspect relevant parameter groups:

```text
shared semantic parameters
Text-LoRA
STATE path
VAE / CLASS path
router
```

Verify:

```text
rho.grad is None
```

exactly.

If the existing attribution dictionary still mainly contains legacy terms such as:

```text
main_task
assigned_expert
advantage
expert_anchor
center
dynamic_mean_anchor
```

add:

```text
utility_factor
utility_router
```

cleanly.

This diagnostic must NOT alter:

```text
forward outputs
main training loss
optimizer state
optimizer step count
scheduler state
```

Initial engineering guardrail:

```text
each auxiliary shared gradient
≈ 5–10% of main-task gradient

all auxiliaries together
≈ 20–30% of main-task gradient
```

This is only a guardrail.

Do NOT automatically force those ratios.

Current lambda candidates may remain unless runtime evidence shows severe
domination.

If domination is observed:

REPORT it first.

Do not silently tune the canonical experiment in the same run.

---

# 22. CORE STRUCTURED TEXT CONTRACT

P1-v8.3 structured prompt is conceptually:

```text
[C1][C2][C3][C4][STATE_m][CLASS][literal normal/abnormal][REAL_NAME]
```

Keep:

```text
4 learned context tokens
```

Separate semantic roles:

```text
CONTEXT
STATE
CLASS
lexical state
REAL_NAME
```

Do NOT implement STATE and CLASS merely by adding their vectors into all generic
context tokens.

The structured encoder should explicitly handle:

```text
context tokens
STATE token
CLASS token
lexical suffix
```

---

# 23. STATE CONTRACT

There are exactly four latent factors:

```text
F1 = (N1, A1)
F2 = (N2, A2)
F3 = (N3, A3)
F4 = (N4, A4)
```

STATE is factor-specific:

```text
STATE_m
```

and derives from the corresponding image-conditioned local visual semantic
prototype/state.

Do NOT assign fixed meanings such as:

```text
F1 = background
F2 = boundary
F3 = lesion core
F4 = normal
```

All four remain latent specialists.

Keep deterministic/lightweight symmetry-breaking mechanisms only where justified.

Do not proactively add stronger factor specialization modules before runtime
evidence.

---

# 24. CLASS CONTRACT

CLASS comes from:

```text
CLS24
  ↓
ClassVAE
  ↓
mu
  ↓
decoder(mu)
  ↓
CLASS token
```

Canonical prompt semantics must be deterministic.

Do NOT use random sampled `z` as CLASS in the canonical text prompt.

Sampling may remain for:

```text
reconstruction
KL
```

if the VAE implementation needs it.

Repeated identical model/input state should produce identical CLASS prompt
semantics.

---

# 25. DYNAMIC TEXT / TEXT-LoRA CONTRACT

Keep these three semantic banks distinct:

```text
hard_frozen

hard_adapted

dynamic_adapted
```

Definitions:

```text
hard_frozen
    original/frozen CLIP trust anchor

hard_adapted
    hard prompt bank through current text transformer + Text-LoRA

dynamic_adapted
    image-conditioned structured P1-v8.3 text
```

Dynamic structured prompts must use:

```text
the SAME CLIP text transformer
+
the SAME Text-LoRA modules
```

as hard_adapted.

No second text encoder.

For P1-v8.3 dynamic structured text:

```text
adapt_text = True
```

Verify gradients reach Text-LoRA.

---

# 26. HARD PROMPT BANK

Keep hard prompt semantics based on existing:

```text
REAL_NAMES
+
PROMPTS
```

Do not replace the hard prompt bank with dynamic prompts.

`hard_frozen` remains the frozen trust anchor.

---

# 27. GLOBAL / BASE TEXT CONTRACT

The BASE predictor remains the generalist:

```text
DFG + SS2D
```

trained jointly from OpenAI CLIP initialization.

Canonical P1-v8.3 should preserve the strong Phase2B-style global hybrid semantic
path unless an explicit controlled ablation changes it.

This means the global/base semantic branch must remain logically distinct from the
new local structured factor branch.

Do NOT interpret “Phase2B-style hybrid” as loading a Phase2B checkpoint.

Audit current source carefully so these concepts are not conflated:

```text
hard_frozen
hard_adapted
static learned/hybrid global prompt
dynamic structured factor text
```

The local dynamic factor bank is for the bounded local semantic residual.

It must NOT silently replace the stable global/base predictor.

If current pulled source/checkpoint metadata disagrees about the exact global
prompt mode, reconcile:

```text
model construction
train.py
test.py
checkpoint metadata
launcher/preflight
```

to the same canonical semantics.

Do not infer truth from an old parser default alone.

---

# 28. LOCAL FACTOR GEOMETRY

Canonical:

```text
local_factor_mode = center_spread
local_center_mix = 0.05
local_factor_spread = 0.10
```

These are critical contract fields.

Do NOT let them silently fall back through `**kwargs`.

Verify constructor plumbing end-to-end.

Center-spread is geometry / controlled symmetry breaking.

Do not claim it alone proves semantic specialization.

---

# 29. RHO CONTRACT

Canonical:

```text
rho = 0.05
```

Exactly.

Training:

```text
requires_grad = False
```

and rho must be absent from optimizer parameter groups.

Checkpoint compatibility with an old gate-shaped state is acceptable, but actual
P1-v8.3 runtime behavior must remain fixed `.05`.

A TEST-ONLY ablation override such as:

```text
rho = 0
```

may exist for diagnostics.

Never train rho.

---

# 30. FINAL LOCAL CORRECTION SEMANTICS

Base logits:

```text
[zN, zA]
```

Factor local evidence:

```text
l_m =
10 * (sim_A_m - sim_N_m)
```

Dense router probabilities:

```text
pi_m
```

Local correction:

```text
c =
0.05 * sum_m pi_m * l_m
```

Final:

```text
[zN, zA + c]
```

Signed correction is allowed.

Do NOT rectify correction to positive-only.

There must be NO second DFG pass after local correction.

---

# 31. CHECKPOINT CONTRACT — AUDIT CAREFULLY

Inspect:

```text
model/checkpoint_utils.py
model/h6/model.py
train.py
test.py
```

Checkpoint metadata must be sufficient to reconstruct actual P1-v8.3 semantics.

Persist/validate at minimum:

```text
progress_version = P1-v8.3

git SHA
seed

precision
TF32 policy
gradient checkpointing

n_groups
num_factors

DFG mode/dim/tau
SS2D config
beta schedule

global/base text mode

local factor mode
center mix
factor spread

rho=.05 fixed
rho_trainable=false

structured prompt version/layout
ctx_len
STATE enabled
CLASS enabled
dynamic Text-LoRA enabled

VAE config
decoder(mu) prompt semantics

utility denominator floor
tau_utility
epsilon schedule
gain threshold
entropy threshold

router mode
routing=dense
prediction temperature

metric protocol where relevant
```

Important known audit target:

legacy checkpoint code may still contain hardcoded metadata such as:

```text
use_hybrid_soft_prompt = True
```

or stale legacy variant names.

Do NOT blindly delete such fields.

Determine whether each field correctly describes the canonical current runtime.

Then make:

```text
model
train
test
checkpoint
```

consistent.

Also inspect P1-v8.3 variant metadata.

Do not save something like:

```text
p1_v6_structural_specialization
```

as the canonical v8.3 variant when a clean explicit label can be used.

Prefer something clear such as:

```text
p1_v8_3_structured_utility_routing
```

if appropriate.

Metadata cleanup must not alter model math.

Critical semantic train/test mismatch must hard-fail with a clear error.

---

# 32. CHECKPOINT ROUNDTRIP READINESS

Prepare a checkpoint roundtrip test/probe.

On a fixed deterministic small batch:

```text
forward
save
construct fresh model
load
forward again
```

Compare within appropriate FP32 tolerance:

```text
base logits
factor logits
router probabilities
routed correction
final prediction
```

Do not require cross-GPU bit identity.

If real data/model checkpoint roundtrip is too expensive before smoke, ensure at
least the implementation/test scaffolding is correct and run it at the earliest
permitted small real batch.

---

# 33. DATA ROOT PORTABILITY

Canonical data resolution should support:

```text
ACDCLIP_DATA_ROOT
```

Precedence:

```text
1. ACDCLIP_DATA_ROOT
2. <repo>/data
3. documented legacy fallback if necessary
```

All DATA_PATH values should derive from one resolved root.

Log resolved data root once.

Do not scatter hard-coded:

```text
/home/ai4/...
/workspace/...
```

through canonical source.

A documented legacy fallback is acceptable only where necessary for backward
compatibility.

---

# 34. DISCOVER EXISTING LAB DATA — DO NOT DOWNLOAD

Search the lab machine under reasonable likely locations, especially:

```text
~/caohuy
```

and known historical project/data directories.

Look for:

```text
VisA_20220922

MedAD/Brain_AD
MedAD/Liver_AD
MedAD/Retina_RESC_AD

Colon/CVC-ClinicDB
Colon/CVC-ColonDB
Colon/Kvasir
```

Expected canonical repo layout:

```text
<repo>/data/
├── VisA_20220922/
├── MedAD/
│   ├── Brain_AD/
│   ├── Liver_AD/
│   └── Retina_RESC_AD/
└── Colon/
    ├── CVC-ClinicDB/
    ├── CVC-ColonDB/
    └── Kvasir/
```

Avoid:

```text
data/data/
```

If complete data exists externally:

prefer symlink through:

```text
tools/setup_med_visa_data.py
```

using local/offline mode.

Do not duplicate several GB unnecessarily.

---

# 35. HARDEN tools/setup_med_visa_data.py

Audit the current pulled setup tool before using it.

It must support:

```text
--source-root
--data-root
--link-mode auto|symlink|copy
--verify
--force
```

Local/offline operation must NOT invoke Kaggle.

Download support may exist as an explicit fallback but must not run automatically.

Candidate root discovery must handle nesting robustly.

Do not assume archive/extracted depth.

---

# 36. KNOWN DATA VERIFICATION GAP — ANOMALY MASK

For every manifest record:

```text
label == 1
```

require:

```text
mask_path is present
mask_path is non-empty
mask file physically exists
```

Do NOT allow:

```text
label=1
mask_path missing
```

or:

```text
label=1
mask_path=""
```

to pass.

Normal-mask conventions may follow dataset metadata, but anomaly masks are
mandatory.

---

# 37. CLASS NAME VERIFICATION

Do not merely verify:

```text
class_name is a non-empty string
```

Validate class names against the authoritative repository dataset metadata /
expected class set.

Avoid creating an unnecessary second divergent hardcoded metadata source.

Use:

```text
dataset/info.py
manifest metadata
existing authoritative constants
```

where appropriate.

---

# 38. VERIFY ALL MANIFEST REFERENCES

Use the actual:

```text
dataset/hub/*.jsonl
```

manifests.

For every relevant record verify:

```text
JSON parses
label parses
class name valid
image exists
anomaly mask exists
```

Count:

```text
samples by dataset
label counts
class counts
missing images
missing masks
```

No medical validation split should be introduced.

---

# 39. EXPECTED SIX-MEDICAL COUNTS

Expected:

```text
Brain
total = 3715
normal = 640
anomaly = 3075

Liver
total = 1493
normal = 833
anomaly = 660

Retina
total = 1805
normal = 1041
anomaly = 764

Colon_clinicDB
total = 612
all anomaly

Colon_colonDB
total = 380
all anomaly

Colon_Kvasir
total = 1000
all anomaly
```

Total six-medical images:

```text
9005
```

If these manifest/data counts do not match:

STOP before GPU smoke and diagnose.

Save verification to:

```text
runs/p1_v83_dev/data_verification.json
```

---

# 40. TRAIN / MEDICAL DATA POLICY

Training:

```text
VisA ONLY
```

Medical datasets:

```text
TEST ONLY
```

No:

```text
medical validation
medical model selection
medical hyperparameter tuning
medical epoch selection
```

Do not scan multiple checkpoints on medical data and choose the best.

Canonical future checkpoint:

```text
epoch 20
```

---

# 41. OPENAI CLIP WEIGHT RESOLUTION

Canonical weight:

```text
ViT-L-14-336px.pt
```

Resolution priority should support:

```text
1. ACDCLIP_CLIP_VITL14_336

2. <repo>/model/ViT-L-14-336px.pt

3. documented lab/legacy fallback if available
```

The historical lab location may be:

```text
/home/ai4/.cache/clip/ViT-L-14-336px.pt
```

but do NOT reintroduce this as a mandatory hardcoded canonical path.

Check existing file.

Do NOT download model weights automatically.

Do NOT use an OpenAI API key.

Record:

```text
resolved path
file size
SHA256 if practical
```

If missing:

STOP before model smoke and report attempted paths.

---

# 42. STATIC / NO-DATA GATE AFTER FIXES

After fixing known implementation issues, run compilation:

```bash
python -m py_compile \
  tools/setup_med_visa_data.py \
  model/h6/utility_routing.py \
  model/h6/model.py \
  model/h6/semantic_bank.py \
  model/adapter.py \
  model/checkpoint_utils.py \
  model/clip.py \
  dataset/info.py \
  utils.py \
  train.py \
  test.py
```

Then:

```bash
git diff --check
```

Then focused tests:

```bash
PYTHONPATH=. pytest -q \
  tests/test_p1_v83_runtime.py \
  tests/test_p1_v83_structured_utility.py \
  tests/test_setup_med_visa_data.py
```

Previous checkpoint evidence was:

```text
20 passed, 1 warning
```

but new legitimate tests will increase the test count.

Do NOT require exactly 20.

Require:

```text
ALL focused tests PASS
```

If a test fails classify it as:

```text
1. implementation bug
2. invalid/stale test expectation
3. numerical tolerance/shape issue
```

Do not change intended equations merely to obtain a green test.

Save logs under:

```text
runs/p1_v83_dev/
```

---

# 43. NVIDIA / CUDA PREFLIGHT

Only after static tests and dataset integrity pass, inspect the actual GPU.

Run:

```bash
nvidia-smi
```

Then query useful fields if supported:

```bash
nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,compute_cap \
  --format=csv,noheader
```

Also inspect active compute processes if supported:

```bash
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

If multiple GPUs exist:

choose a suitable GPU based on actual:

```text
free VRAM
active processes
device availability
```

Do not assume GPU 0.

Do NOT kill another user's process.

Record selected:

```text
GPU index
GPU name
GPU UUID
compute capability
VRAM
driver version
```

---

# 44. PYTORCH / CUDA PREFLIGHT

Using the active `torchhuy` environment:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
print("device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(
        "device", i,
        "name=", torch.cuda.get_device_name(i),
        "cc=", f"{p.major}.{p.minor}",
        "VRAM_GiB=", p.total_memory / 2**30,
    )
PY
```

Do not reinstall PyTorch/CUDA merely because version strings differ from an old
machine.

First prove an incompatibility.

Store preflight under:

```text
runs/p1_v83_dev/preflight/
```

---

# 45. CANONICAL NUMERICAL POLICY

P1-v8.3 canonical runtime:

```text
precision = FP32

AMP = OFF
BF16 = OFF
FP16 = OFF

TF32 = OFF
```

Explicit:

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Gradient checkpointing:

```text
ON
```

Canonical train batch semantics:

```text
batch_size = 1
grad_accum_steps = 6
effective batch = 6
img_size = 518
```

Do not increase canonical training batch simply because the lab GPU has more
memory.

Do NOT use in canonical evidence:

```text
torch.compile
CUDA graphs
TF32
AMP
BF16
FP16
```

---

# 46. DATALOADER REPRODUCIBILITY

Verify worker seed:

```python
worker_seed = torch.initial_seed() % 2**32
random.seed(worker_seed)
np.random.seed(worker_seed)
```

Use a seeded:

```text
torch.Generator
```

for DataLoader ordering.

Training may still use:

```text
shuffle=True
```

but must be reproducible under equivalent environment/config/seed.

Log:

```text
seed
num_workers
pin_memory
persistent_workers
prefetch_factor
```

Do not optimize loader settings before simple reference correctness is proven.

---

# 47. MODEL CONFIG PRE-FLIGHT — HARD ASSERTIONS

Before real forward, assert/log:

```text
branch
git SHA

data root
VisA integrity

CLIP checkpoint

GPU / driver
PyTorch / torch CUDA

P1-v8.3

OpenAI CLIP initialization
NO Phase2B checkpoint load

FP32
TF32 OFF
AMP OFF

gradient checkpointing ON

batch_size = 1
grad_accum_steps = 6

center_spread
center = .05
spread = .10

rho = .05
rho requires_grad = false
rho absent optimizer

structured prompt ON
STATE ON
CLASS ON
CLASS = decoder(mu)

dynamic Text-LoRA ON

four factors

dense routing

experts OFF
load bias OFF
factor balance OFF
Top-K prediction OFF
cluster responsibility OFF
functional diversity OFF
selective-use gate OFF
```

Also assert the intended stable global/base semantic branch consistently across:

```text
model
train
test
checkpoint
```

If ANY critical contract is wrong:

FAIL EARLY.

Fix it before GPU smoke.

---

# 48. GPU FUNNEL — DO NOT SKIP AHEAD

Proceed only in this order:

```text
static/unit
→ data integrity
→ GPU preflight
→ model preflight
→ 1-batch forward
→ 1-batch backward
→ 8-batch smoke
→ 32–64 diagnostic only if useful
```

Do not launch a long run just because one small gate passes.

---

# 49. G0.5 — ONE REAL BATCH FORWARD

Run one deterministic VisA training batch.

No optimizer step required.

Verify:

```text
all relevant tensors finite

base logits finite

factor_patch_logits shape:
[G,B,P,4]

dense router probability shape:
[G,B,P,4]

router probabilities sum to 1 over factor dimension

STATE factor-specific

CLASS deterministic for repeated identical input/model state

hard_frozen stable

dynamic_adapted image-conditioned

rho exactly .05

factor outputs not exact-identical
```

Save compact diagnostics:

```text
tensor shapes
min/max/mean/std
factor differences
router entropy
utility tensors where applicable
```

Do not infer model quality from one forward batch.

---

# 50. ONE REAL BATCH BACKWARD

Run one controlled backward pass.

Do not use it as a final training measurement.

Verify finite gradients.

Required evidence:

```text
STATE path gradient > 0

VAE/CLASS path gradient > 0

Text-LoRA gradient > 0

factor-specific semantic path gradient > 0

base DFG/adapter path alive

router gradient > 0
when router utility diagnostic applies

rho gradient is exactly None
```

Also perform the utility-factor/router gradient attribution probe described above.

Do not silently tune lambda from this one batch.

Record ratios and report them.

---

# 51. 8-BATCH REAL TRAINING SMOKE

Only after one-forward and one-backward pass.

Run at most:

```text
8 batches
```

Canonical math/settings.

Because:

```text
grad_accum_steps=6
```

this smoke must exercise at least one actual optimizer step.

Verify:

```text
finite forward
finite task loss
finite factor utility loss
finite router utility loss
finite total loss

finite backward gradients

optimizer step succeeds

rho remains exactly .05

rho has no gradient

no unexpected GPU memory growth

structured text path remains alive

Text-LoRA remains alive

CLASS path remains alive

factor outputs do not exact-collapse

router remains dense

legacy auxiliaries remain OFF

experts remain OFF
```

Record GPU:

```text
allocated
reserved
peak allocated
peak reserved
```

Do not call `torch.cuda.empty_cache()` repeatedly in the train hot loop.

---

# 52. FACTOR COLLAPSE HARD GATE

Hard failure:

```text
factor outputs exactly identical
for two consecutive probes
```

Warning heuristic only:

```text
mean absolute factor-function correlation > .98
```

persisting across meaningful windows.

Do NOT automatically add functional diversity because correlation is high.

First determine WHY.

---

# 53. 32–64 BATCH DIAGNOSTIC — ONLY IF NEEDED

If 8-batch smoke is clean, use up to:

```text
32–64 batches
```

when needed to obtain more stable structural/value diagnostics.

Do NOT automatically run 200–300 batches.

Do not run one epoch unless this diagnostic evidence explicitly justifies it and
the user later approves.

Required representation/function metrics:

```text
factor embedding pairwise cosine
factor embedding L2
factor effective rank

STATE pairwise differences

factor_patch_logits:
    pairwise correlation
    max difference
    std across factors

factor gradient norms
factor gradient cosine
```

Utility metrics:

```text
L_base
L_per_factor
gain_rel per factor
best-second utility margin

teacher entropy
teacher max probability
informative fraction
all-harm fraction

winner shares:
F1
F2
F3
F4
```

Router:

```text
router top-1 agreement with utility winner
teacher-router KL
router entropy
router usage

query effective rank
query variance
```

Value diagnostics:

```text
Base/rho0
BestSingle
OracleMulti
Uniform
SoftRouted
HardRouted

G_local
G_multi

capture
capture_valid
capture_denominator
```

---

# 54. DECISION ORDER

Use this exact reasoning order.

## Step 1

Runtime/config correct?

If NO:

```text
fix runtime/config
```

Do not discuss factor science yet.

## Step 2

Factors differ in representation?

If NO:

```text
STATE / identity / wiring problem
```

## Step 3

Factor outputs differ functionally?

If NO:

```text
functional collapse candidate
```

## Step 4

Is:

```text
G_local > 0
```

?

Where:

```text
G_local =
(Base - OracleMulti) / Base
```

If NO:

```text
even oracle local factors do not improve Base
```

Do NOT tune the router.

## Step 5

Evaluate:

```text
G_multi =
(BestSingle - OracleMulti) / Base
```

If low:

distinguish:

```text
true low-mode structure
vs
factor collapse
```

using factor/function evidence.

Do not force diversity.

## Step 6

Is teacher informative?

If NO:

do not blame router.

## Step 7

Does router match teacher?

If NO:

router-specific problem.

## Step 8

Is SoftRouted better than Uniform?

If NO:

check:

```text
HardRouted good but SoftRouted weak
→ probability calibration candidate

HardRouted weak while Oracle good
→ router representation candidate
```

## Step 9

Does routed rho=.05 improve rho=0?

If NO:

local branch value remains unproven.

---

# 55. SPECIALIZATION RESCUE ORDER

Do NOT proactively create a rescue architecture.

If specialization fails, inspect in order:

```text
1. STATE is actually factor-specific

2. factor-specific gradients exist

3. structured prompt token placement is correct

4. deterministic symmetry breaking / identities are correct

5. utility temperature / exploration

6. only after evidence:
   strengthen a lightweight factor-specific mechanism

7. functional diversity LAST
```

Functional diversity is allowed only if evidence supports:

```text
G_local > 0
AND
multi-mode potential exists
AND
factor functions still collapse
```

Do not make it a default P1-v8.3 loss.

---

# 56. ROUTER RESCUE ORDER

If factors/teacher are useful but router fails:

```text
1. verify router gradients

2. router LR

3. utility/teacher temperature

4. query/key diagnostics

5. prediction probability calibration

6. only then consider a small detached factor-response projection
```

Do not redesign router before this evidence.

---

# 57. MEDICAL EVALUATION PROTOCOL — AUDIT ONLY

Do NOT execute full medical evaluation in this session.

Future final test datasets exactly:

```text
Brain
Liver
Retina
Colon_clinicDB
Colon_colonDB
Colon_Kvasir
```

Image AUROC/AP are valid for:

```text
Brain
Liver
Retina
```

Colon datasets may be one-class/all anomaly.

For unsupported one-class image metrics report:

```text
N/A
```

Do NOT report:

```text
0.0
```

Image macro includes only datasets with valid normal/anomaly image support.

Pixel metrics use all six.

---

# 58. FUTURE EXACT PIXEL PROTOCOL

Final future medical report:

```text
pixel_stride = 1

full resolution

external exact pixel metrics

all six medical datasets
```

Do not mix legacy stride-4 metrics into the same claimed protocol.

---

# 59. FUTURE MEDICAL PREDICTION PROTOCOL

Canonical medical segmentation:

```text
Gaussian blur:
kernel = 9
sigma = 1.5
```

Resize:

```text
bilinear
align_corners = True
```

Across feature levels:

```text
mean LOGITS first
then softmax
```

Do not average post-softmax probabilities across levels.

Image score:

```text
0.5 * cls
+
0.5 * pmax
```

Freeze this before medical final.

No tuning on medical.

---

# 60. FUTURE BOOTSTRAP / STATISTICS

If final statistical reporting is prepared later:

bootstrap by IMAGE.

Do NOT bootstrap individual pixels independently.

Suggested:

```text
2000 resamples = smoke
10000 resamples = final claim
95% CI
```

Do not execute expensive final statistics now.

---

# 61. SAFE PERFORMANCE WORK

Only after reference correctness.

Allowed candidates:

```text
pin_memory

non_blocking H2D

persistent_workers

prefetch_factor=2

num_workers benchmark:
4 / 8 / 12 as appropriate

torch.inference_mode() for evaluation

cache truly static hard text inference

test batch:
1 → 2 → 4 → maybe 8
```

Any optimization must preserve numerical/reference parity.

Do NOT use:

```text
torch.compile
CUDA graphs
TF32
AMP
BF16
FP16
```

for canonical evidence.

---

# 62. EMPTY_CACHE POLICY

If `test.py` still calls:

```python
torch.cuda.empty_cache()
```

inside normal inference hot loops:

do not remove it blindly.

First establish parity and stable VRAM.

Then benchmark:

```text
with empty_cache
vs
without empty_cache
```

Only remove from normal execution when:

```text
VRAM stable
predictions equivalent
metrics equivalent
```

This is not required before the initial P1-v8.3 smoke.

---

# 63. DO NOT RUN FINAL TRAINING

Do NOT run:

```text
20 epochs
```

Do not run:

```text
8 epochs
3 epochs
1 full epoch
```

unless a later explicit instruction approves it.

Current permitted funnel ends at:

```text
32–64 batch diagnostic
```

unless a smaller stage fails earlier.

---

# 64. DO NOT RUN FULL SIX-MEDICAL FINAL

Do NOT run full medical exact evaluation in this session.

Only:

```text
audit code
prepare launcher
verify protocol
```

---

# 65. PREPARE FINAL TRAIN LAUNCHER — DO NOT EXECUTE

Prepare/repair a future canonical P1-v8.3 training launcher.

It must hard-assert:

```text
dataset = VisA
epochs = 20

img_size = 518

batch_size = 1
grad_accum_steps = 6

FP32
TF32 OFF
AMP OFF

gradient checkpointing ON

P1-v8.3

OpenAI CLIP initialization only

no Phase2B checkpoint

correct stable global/base semantic path

structured text ON

center_spread
center=.05
spread=.10

rho=.05 fixed

dense routing

experts OFF
load bias OFF
balance OFF
Top-K prediction OFF
cluster responsibility OFF
functional diversity OFF
selective-use OFF
```

Do NOT execute it.

---

# 66. PREPARE FINAL MEDICAL LAUNCHER — DO NOT EXECUTE

Prepare/repair future medical exact launcher.

It must assert:

```text
canonical epoch-20 checkpoint

exactly six medical datasets

no medical validation

no medical checkpoint selection

pixel_stride=1

exact full-resolution metrics

correct unsupported-image-metric behavior

image score=.5 cls + .5 pmax
```

Do NOT execute it.

---

# 67. STATUS ARTIFACT

Maintain:

```text
runs/p1_v83_dev/STATUS.json
```

Create it if absent.

Track stages such as:

```text
git_sync
known_bug_fixes
static_compile
no_data_tests
data_discovery
data_integrity
clip_checkpoint
nvidia_preflight
pytorch_preflight
model_preflight
forward_1
backward_1
smoke_8
diagnostic_32_64
final_launcher_ready
medical_launcher_ready
```

Each stage should have:

```text
PASS
FAIL
SKIPPED
NOT_RUN
```

plus concise evidence paths/messages.

This directory is local runtime evidence.

Do not automatically commit `runs/`.

---

# 68. RECOMMENDED ARTIFACT TREE

Use compact artifacts:

```text
runs/p1_v83_dev/
├── STATUS.json
├── lab_sync/
├── preflight/
├── no_data_tests.log
├── compile.log
├── data_verification.json
├── 1batch_forward/
├── 1batch_backward/
├── 8batch_smoke/
└── 32_64batch_diag/
```

Only create directories actually needed.

Avoid giant redundant logs.

---

# 69. SOURCE SECURITY / PORTABILITY SCAN

Before committing any fixes, inspect changed source for:

```text
API keys
credentials
tokens
absolute machine paths
/tmp assumptions
/workspace assumptions
/home/ai4 hardcoding
large binary files
data
model weights
run outputs
```

No secrets or machine-only artifacts may enter a source commit.

---

# 70. FINAL SOURCE TEST GATE

After all code fixes:

```bash
git diff --check
```

Run affected tests first.

Then run full focused suite once:

```bash
PYTHONPATH=. pytest -q \
  tests/test_p1_v83_runtime.py \
  tests/test_p1_v83_structured_utility.py \
  tests/test_setup_med_visa_data.py
```

Compile critical modules again if source changed after the prior compile gate.

All must pass before local commit.

---

# 71. LOCAL COMMIT AFTER FIXES

If:

```text
known bugs fixed
static compilation PASS
focused tests PASS
real smoke source changes clean
```

then create one logical LOCAL checkpoint commit.

Suggested subject:

```text
fix(p1-v8.3): correct diagnostics and harden lab validation gates
```

Body should accurately mention only what was actually changed, e.g.:

```text
- correct normalized G_local and G_multi definitions
- use Uniform baseline for router capture
- add exact value-diagnostic unit tests
- extend utility gradient attribution
- harden anomaly-mask and class-name verification
- reconcile P1-v8.3 checkpoint metadata/runtime contracts
```

Do NOT include:

```text
data/
runs/
checkpoints/
CLIP weights/
cache/
large logs/
credentials/
this prompt file
```

Do NOT push automatically.

Report the local commit SHA.

---

# 72. DO NOT MODIFY THIS PROMPT INTO REPO SOURCE

This file is an execution instruction.

If it lives outside the repository:

leave it there.

Do not add it to Git.

If accidentally placed in the repository:

do not stage it unless the user explicitly asks.

---

# 73. HARD STOP CONDITIONS

STOP the current pipeline stage if:

```text
Git update requires destructive reset

required dataset is incomplete

manifest counts disagree materially

anomaly masks are missing

OpenAI CLIP checkpoint missing

CUDA unavailable

selected GPU unusable

model preflight contract mismatch

P1-v8.3 accidentally loads Phase2B checkpoint

AMP/BF16/FP16 enabled

TF32 enabled

rho trainable

rho receives gradient

rho changes from .05

Top-K affects canonical prediction

load bias active

equal factor balance active

paired experts active

cluster responsibility active

functional diversity active

selective-use gate active

structured STATE/CLASS wiring broken

dynamic text bypasses required Text-LoRA

non-finite forward

non-finite loss

non-finite gradients

factor outputs exactly identical for two consecutive probes

checkpoint critical semantic mismatch
```

Do NOT mask a hard failure by adding another auxiliary loss.

Diagnose the first failing gate.

---

# 74. FINAL REPORT

At the end, provide a concise report.

## Git

```text
repo path
branch
old local SHA
pulled remote SHA
final local SHA
local fix commit SHA if created
```

## Bugs fixed

For each:

```text
file
old behavior
new behavior
test/evidence
```

Especially report:

```text
G_local
G_multi
capture
gradient attribution
dataset verification
checkpoint/config metadata
```

## Tests

```text
focused test command
passed
failed
warnings
duration
py_compile result
git diff --check result
```

## Data

```text
resolved data root
VisA ready/not ready

Brain count
Liver count
Retina count
ClinicDB count
ColonDB count
Kvasir count

missing images
missing masks
invalid class names
```

## CLIP

```text
resolved checkpoint path
size
hash if computed
```

## NVIDIA

```text
GPU index
GPU name
GPU UUID
compute capability
VRAM
driver
```

## PyTorch

```text
torch version
torch CUDA version
cuDNN
CUDA available
```

## Runtime contract

Report PASS/FAIL for:

```text
OpenAI-only initialization

FP32
TF32 off
AMP off
gradient checkpointing

batch1 / accum6

center-spread .05/.10

rho fixed .05/no-grad/not optimizer

structured STATE
deterministic CLASS
shared Text-LoRA

dense routing

experts/load-bias/balance/top-k/cluster/func-div/use-gate OFF
```

## 1-batch forward

```text
PASS/FAIL
key tensor shapes
finite status
factor differentiation
```

## 1-batch backward

```text
PASS/FAIL

STATE grad
CLASS/VAE grad
Text-LoRA grad
factor grad
router grad
rho grad
```

## 8-batch smoke

```text
PASS/FAIL
optimizer step executed
finite status
peak VRAM
collapse status
```

## 32–64 diagnostic, if run

Report:

```text
Base
BestSingle
OracleMulti
Uniform
SoftRouted
HardRouted

G_local
G_multi

capture
capture_valid
capture_denominator

informative fraction
all-harm fraction

winner F1/F2/F3/F4

teacher/router agreement
teacher-router KL

factor correlation
factor effective rank
collapse status
```

## Remaining unproven items

Explicitly state what is NOT yet proven.

---

# 75. CLAIM POLICY

Do NOT call the architecture fully validated after a small smoke.

If all permitted runtime gates pass, acceptable wording is:

```text
P1-v8.3 implementation/runtime smoke PASS
```

Do NOT yet claim:

```text
P1-v8.3 final architecture PASS
```

Final architecture evidence still requires later:

```text
longer stability evidence
canonical 20-epoch VisA training
frozen epoch-20 checkpoint
exact six-medical one-shot evaluation
```

Do not claim real:

```text
G_local
G_multi
medical gain
router capture
factor specialization
```

unless measured on actual runtime data.

---

# 76. IMMEDIATE EXECUTION ORDER

START NOW.

Do exactly this high-level sequence:

```text
1. Locate existing lab repo.

2. Record pre-sync Git state.

3. Fetch/pull latest:
   phase4-progress1-cops-dynamic-prompt

4. Inspect the newly pulled P1-v8.3 implementation.

5. Fix known bugs:
   - G_local
   - G_multi
   - capture
   - exact formula tests
   - utility gradient attribution
   - anomaly-mask verification
   - class-name verification
   - checkpoint/runtime metadata inconsistencies

6. Run static compile + focused no-data tests.

7. Discover existing lab dataset.
   DO NOT download.

8. Verify manifests/data/counts.

9. Resolve existing OpenAI CLIP checkpoint.

10. Run NVIDIA + PyTorch CUDA preflight.

11. Run strict P1-v8.3 model/config preflight.

12. Run one real forward batch.

13. Run one real backward batch.

14. Run at most 8 real training batches.

15. If useful and all earlier gates pass:
    run 32–64 batch diagnostics.

16. Prepare final train/medical launchers.
    DO NOT execute them.

17. Commit clean source fixes locally.

18. DO NOT push automatically.

19. Update STATUS.json.

20. Give the final concise evidence report.

STOP before final training or full medical evaluation.
```