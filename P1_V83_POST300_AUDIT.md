# P1-v8.3 POST-300 AUDIT
## anomaly semantics + canonical schedule + grad-accum remainder + lambda gradient probe

Continue from the CURRENT LAB MACHINE state.

Project:

```text
Phase 4 → Progress 1 → candidate P1-v8.3
```

Expected repo:

```text
/home/ai4/caohuy/ACD-CLIP-phase4
```

Verify path and Git state rather than blindly assuming it.

Current relevant commits:

```text
parent implementation/fix:
12d0b7fbb045b2c549e4be2c6c6daf85dae342f2

diagnostics commit:
2019e5ada8f30166ab70c8a859de35c487ecc1c1
```

Do NOT pull/rebase/reset unless an actual repository problem requires it.

Do NOT push.

Do NOT run final20.

Do NOT run medical evaluation.

Do NOT run a full VisA epoch in this task.

This is a focused audit/probe session only.

---

# 1. CURRENT EVIDENCE

The canonical 300-batch probe completed:

```text
300 batches
50 optimizer steps
seed 0
FP32
rho=.05 fixed
no exact factor collapse
```

Main findings:

```text
final recent-window G_local ≈ 0.08325
final recent-window G_multi ≈ 0.00333

F1 winner share ≈ 99.36%

factor effective rank ≈ 1.0187
factor embedding cosine mean ≈ 0.9967
factor function correlation mean ≈ 0.9985
```

Router teacher eventually became active:

```text
informative fraction ≈ 4.44%
router top1 agreement ≈ 95.2%
capture ≈ 3.67%
```

But anomaly-region behavior was concerning:

```text
anomaly informative fraction = 0

anomaly all-harm fraction = 1.0

anomaly best_gain_rel_mean < 0

anomaly gain_rel_mean < 0
```

Normal-region utility was positive.

Gradient attribution at the end also showed:

```text
utility_factor / main shared grad ≈ 22.6%

utility_router / main shared grad ≈ 87.9%

total auxiliary / main shared grad ≈ 110.5%
```

This task must determine whether these observations come from:

```text
A. implementation/semantic bug

B. canonical schedule/runtime mismatch

C. excessive auxiliary scaling

D. genuine current learning behavior
```

Do not assume which one is true.

---

# 2. AUTHORITATIVE CONTRACT

For this task, use the current P1-v8.3 implementation/spec already encoded in the
repo plus these explicit contracts.

Patch target:

```text
y_patch = anomaly coverage
```

Expected polarity:

```text
normal ≈ 0
anomaly ≈ 1
```

Base margin:

```text
z0 = abnormal_logit - normal_logit
```

Therefore:

```text
positive z0
→ more abnormal

negative z0
→ more normal
```

Factor local evidence:

```text
l_m =
10 * (
    similarity(patch, A_m)
    -
    similarity(patch, N_m)
)
```

Therefore:

```text
positive l_m
→ factor pushes toward abnormal

negative l_m
→ factor pushes toward normal
```

Candidate factor margin:

```text
z_m =
stopgrad(z0)
+
rho * l_m
```

with:

```text
rho = .05 fixed
```

Utility:

```text
L0 =
BCEWithLogits(z0.detach(), y_patch)

Lm =
BCEWithLogits(z_m, y_patch)
```

Relative gain:

```text
gain_rel =
(L0 - Lm)
/
clamp_min(L0, floor)
```

Positive gain means the factor helps.

Negative gain means the factor harms.

Do not change these equations unless an actual implementation contradiction is
proved.

---

# 3. FIRST: GIT / SOURCE SAFETY

Run:

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4

git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -6
```

Preserve all historical dirty/untracked artifacts.

Do NOT use:

```text
git reset --hard
git clean
git stash -u
```

Create:

```text
runs/p1_v83_dev/post300_audit/
```

Store all audit evidence there.

---

# 4. DO NOT RERUN LONG TRAINING

Do NOT rerun:

```text
300 batches
1 epoch
3 epochs
20 epochs
medical test
```

This task should primarily use:

```text
source inspection
existing 300-batch artifacts
fixed-batch forward probes
no-optimizer-step autograd probes
small unit tests
```

A few deterministic real batches are allowed only when necessary to prove
semantics.

---

# 5. AUDIT A — PATCH TARGET POLARITY

Trace the exact code path from:

```text
VisA GT mask
→ augmentation
→ valid mask
→ patch target
→ BCE target
```

Verify empirically and in source:

```text
normal pixels/patches → target near 0

anomaly pixels/patches → target > 0
and strongly anomalous patch → target near 1 where appropriate
```

Check:

```text
mask dtype
mask normalization
255→1 conversion
logical inversion
interpolation mode
area/average pooling
valid-mask multiplication
thresholding if any
```

Pay special attention to any convention such as:

```text
0=defect
1=normal
```

or accidental inversion.

Do not trust variable names alone.

For a few real VisA anomaly images:

record:

```text
raw mask min/max/unique

pooled y_patch:
min
max
mean

normal-region sample y values

anomaly-region sample y values
```

Expected result:

```text
normal y ≈ 0
anomaly y > 0
```

If polarity is reversed:

HARD FAIL.

Do not continue to lambda probing until fixed and retested.

---

# 6. AUDIT B — BASE LOGIT POLARITY

Trace the exact model output convention.

Prove which channel/index means:

```text
normal
abnormal
```

Then verify that utility code computes:

```text
z0 = abnormal - normal
```

not the reverse.

For a fixed real batch, log:

```text
normal_logit
abnormal_logit
z0
y_patch
```

for selected:

```text
normal patches
anomaly patches
```

Do not require the model to already classify perfectly.

This audit is only about sign convention.

Add a deterministic unit test if channel order is not already protected by one.

---

# 7. AUDIT C — FACTOR N/A POLARITY

Trace factor pair construction and ensure:

```text
N_m = normal semantic endpoint
A_m = abnormal semantic endpoint
```

Verify local evidence is exactly:

```text
l_m = 10 * (sim_A - sim_N)
```

not:

```text
sim_N - sim_A
```

and not accidentally swapped due to tensor ordering.

For a fixed batch, log per factor:

```text
sim_N
sim_A
l_m
```

for selected normal/anomaly patches.

Again, do not demand learned behavior yet.

Only prove semantic polarity.

---

# 8. AUDIT D — DIRECT ANOMALY PATCH TRACE

This is critical.

Choose a small deterministic set of real anomaly patches from VisA.

Prefer at least:

```text
8–16 anomaly patches
```

across several images/classes if practical.

For each selected patch record:

```text
class_name
image identifier
patch coordinate/index
y_patch

zN
zA
z0

for m=1..4:
    sim_N_m
    sim_A_m
    l_m
    rho*l_m
    z_m
    L0
    Lm
    gain_rel_m
```

Also record:

```text
best factor
best gain
all_harm boolean
```

Write results to something like:

```text
runs/p1_v83_dev/post300_audit/anomaly_patch_trace.json
```

Main question:

```text
WHY are anomaly factors all-harm?
```

Classify observed behavior:

### SIGN_BUG

Example:

```text
anomaly y=1
but positive abnormal evidence increases loss due to reversed margin/target
```

### MODEL_BEHAVIOR

Example:

```text
signs are mathematically correct
but learned factors genuinely produce negative abnormal correction on anomaly patches
```

### TARGET/VALID_MASK_BUG

Example:

```text
selected anomaly region is labeled/weighted incorrectly
```

Do not speculate without this direct trace.

---

# 9. AUDIT E — NORMAL/ANOMALY BALANCING IMPLEMENTATION

The factor utility loss is intended to prevent background dominance.

For anomaly-containing images, verify implementation behaves approximately as:

```text
0.5 * mean(valid normal-region utility loss)
+
0.5 * mean(valid anomaly-region utility loss)
```

For fully normal images:

```text
mean(valid normal-region utility loss)
```

Audit source AND one fixed batch.

Log:

```text
normal valid patch count
anomaly valid patch count

normal-region mean factor loss
anomaly-region mean factor loss

balanced combined factor loss
```

Check for bugs such as:

```text
anomaly patches absent from balancing

normal patches counted twice

region masks inverted

soft y_patch incorrectly split

invalid/padded patches included

normal/anomaly branch chosen using wrong image label
```

If anomaly signal is effectively underweighted despite intended balancing:

fix it before any longer run.

---

# 10. AUDIT F — EXPLORATION EPSILON SCHEDULE

Determine the exact epsilon schedule used by:

```text
the canonical 20-epoch training path
```

and separately:

```text
the completed 300-batch diagnostic runner
```

The intended initial schedule is approximately:

```text
epsilon = .15 early
→ .05 late
```

Do not assume the 300-batch probe used the right value.

Inspect code/config and report exact:

```text
current epoch
total schedule epochs
computed epsilon
```

for:

```text
canonical epoch 1 / 20

canonical epoch 20 / 20

300-batch probe
```

Explicitly test for this failure mode:

```text
diagnostic runner sets:
current_epoch=1
total_epochs=1

therefore schedule immediately returns .05
```

If the 300-batch probe used `.05` instead of canonical early `.15`:

mark:

```text
NONCANONICAL_EARLY_EXPLORATION
```

Do NOT discard the existing results.

Explain whether that likely biases specialization toward less exploration.

Fix the diagnostic runner / schedule plumbing if necessary.

Add an exact unit test for:

```text
epoch1/20 ≈ .15
epoch20/20 ≈ .05
```

---

# 11. AUDIT G — GRADIENT ACCUMULATION REMAINDER

300 batches had:

```text
300 % 6 = 0
```

so the completed 300-batch probe itself has no remainder problem.

The future full VisA epoch has:

```text
2162 samples
batch_size=1
grad_accum_steps=6

2162 % 6 = 2
```

Audit exactly what canonical `train.py` does with the final 2 micro-batches.

Classify behavior as:

### CARRY

accumulation continues correctly across epoch boundary

### RESCALED_FLUSH

partial group is flushed with correct normalization

### DROP

remainder is intentionally dropped and explicitly documented

### UNDERWEIGHT_BUG

loss was divided by 6 for only 2 samples, then optimizer stepped without correcting
the scale

### OTHER

explain precisely

Do not guess from apparent optimizer step count.

Create a tiny deterministic unit test using a toy scalar model.

Test at least:

```text
6 microbatches
8 microbatches
12 microbatches
```

Compare accumulated update against a mathematically equivalent reference.

Do not alter main training behavior unless test proves a bug.

---

# 12. AUDIT H — NULL GRADIENT DIAGNOSTICS

Classify these fields:

```text
class_to_context_grad_norm

factor_generator_context_grad_norm

factor_generator_head_grad_norms

factor_generator_identity_grad_norm

factor_id_projection_grad_norm
```

Each must be labeled:

```text
EXPECTED_NA
```

if the corresponding legacy/noncanonical module is intentionally unused in
P1-v8.3,

or:

```text
BUG
```

if current canonical P1-v8.3 expects that path to participate.

Do NOT modify architecture just to make a legacy diagnostic non-null.

Where appropriate, update diagnostic output from ambiguous:

```text
null
```

to explicit metadata:

```text
status: EXPECTED_NA
reason: ...
```

only if that can be done without invasive changes.

---

# 13. GRADIENT ATTRIBUTION — VERIFY WEIGHTED VS RAW SEMANTICS

The previous report showed roughly:

```text
utility_factor/main shared grad ≈ .226

utility_router/main shared grad ≈ .879

total auxiliary/main ≈ 1.105
```

Before changing lambdas, verify exactly what these ratios mean.

Inspect gradient attribution code.

Determine whether ratios are based on:

```text
RAW auxiliary loss gradient
```

or:

```text
ACTUAL weighted contribution:
lambda * auxiliary_loss
```

This distinction is CRITICAL.

For each component report:

```text
raw loss value

lambda

raw shared gradient norm

weighted shared gradient norm

weighted/main ratio
```

for:

```text
main
utility_factor
utility_router
```

If current diagnostics already include weights but ratio calculations ignore them,
that is a diagnostic bug.

Fix the diagnostic before tuning lambdas.

Add a deterministic test proving:

```text
changing lambda by factor k
changes reported weighted gradient contribution by approximately factor k
on identical forward state
```

Do not tune lambda before this is proven.

---

# 14. FIXED-STATE LAMBDA GRADIENT-ONLY PROBE

ONLY if all semantic/sign audits pass.

Do NOT train.

Do NOT optimizer.step().

Use ONE deterministic fixed model state and fixed real batch/probe set.

Prefer a checkpoint/state where router utility is active, e.g. the end or late state
from the completed 300-batch run, if available.

If the 300-batch runner did not save a usable state:

use the smallest reproducible late-state mechanism available.

Do not rerun 300 batches solely for this unless unavoidable.
If unavoidable, STOP and report rather than launching automatically.

Evaluate gradient attribution under a small lambda grid.

Canonical current:

```text
lambda_factor = .10
lambda_router = .10
```

Candidate diagnostic grid:

```text
lambda_factor:
.10
.05
.03

lambda_router:
.10
.03
.01
```

You do NOT need all 9 combinations if separability lets you derive scaling exactly.

Because gradients of:

```text
lambda * L
```

scale linearly with lambda on a fixed forward graph, prefer computing raw component
gradient once and analytically reporting candidate weighted ratios.

Do not perform redundant backward passes when linear scaling is sufficient.

Report for each candidate:

```text
factor weighted/main shared grad ratio

router weighted/main shared grad ratio

total aux/main shared grad ratio
```

Also report parameter-group-specific ratios for:

```text
shared semantics

STATE path

router

Text-LoRA
```

where meaningful.

---

# 15. GRADIENT GUARDRAIL

Use the existing engineering guardrail only as a guide:

```text
each auxiliary shared contribution:
roughly 5–10% of main

total auxiliary:
roughly 20–30% of main
```

Do not force exact equality.

Do not optimize solely to hit these numbers.

The final recommendation must also consider:

```text
factor specialization purpose

router teacher sparsity

anomaly-region behavior

main-task preservation
```

Recommend at most ONE lambda pair for the next experiment.

Do not execute it.

---

# 16. DO NOT CHANGE ARCHITECTURE

This task may fix proven bugs in:

```text
sign convention
mask/target plumbing
epsilon schedule plumbing
gradient accumulation
diagnostic weighting
```

But do NOT introduce:

```text
functional diversity

orthogonal loss

new factor identities

new router architecture

selective-use gate

trainable rho

Top-K

load balancing

new semantic roles
```

If no implementation bug is found:

leave architecture unchanged.

---

# 17. SOURCE CHANGE POLICY

If a real bug is found:

make the smallest focused fix.

For every source fix:

```text
add/update deterministic test
run affected test
```

Then once at end run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_p1_v83_runtime.py \
  tests/test_p1_v83_structured_utility.py \
  tests/test_setup_med_visa_data.py
```

Also:

```bash
git diff --check
python -m py_compile \
  model/h6/utility_routing.py \
  model/h6/model.py \
  model/adapter.py \
  train.py \
  test.py
```

Do not commit runtime artifacts.

---

# 18. OPTIONAL SMALL REAL PROBE AFTER BUG FIX

If and ONLY if a semantic implementation bug is found and fixed:

run the minimum real-data probe needed to prove the fix.

Maximum:

```text
8–16 batches
```

No long training.

Do NOT rerun 300 batches in this task.

Do NOT run one epoch.

For an anomaly-related bug, directly verify:

```text
anomaly target polarity correct

anomaly gain no longer structurally wrong due to sign/plumbing
```

Do not require the model to become good immediately after a bug fix.

---

# 19. OUTPUT ARTIFACTS

Use:

```text
runs/p1_v83_dev/post300_audit/
```

Create:

```text
audit_summary.json

semantic_polarity.json

anomaly_patch_trace.json

na_balance_audit.json

epsilon_schedule_audit.json

grad_accum_audit.json

null_gradient_classification.json

gradient_weighting_audit.json

lambda_gradient_probe.json

tests.log
```

Only create files that are meaningful.

---

# 20. STATUS UPDATE

Update:

```text
runs/p1_v83_dev/STATUS.json
```

Add:

```text
post300_semantic_audit

epsilon_schedule_audit

grad_accum_remainder_audit

gradient_weighting_audit

lambda_gradient_probe
```

Use:

```text
PASS
FAIL
NOT_RUN
```

Do not change the overall claim to final architecture PASS.

---

# 21. COMMIT POLICY

If source was modified and all tests pass:

create ONE local commit.

Suggested:

```text
fix(p1-v8.3): harden post-300 training contracts
```

or a more precise subject based on actual fixes.

Do NOT amend prior commits.

Do NOT push.

If no source changes are necessary:

do not create an empty commit.

---

# 22. SLEEP / EXECUTION POLICY

Most tasks here should be short.

Do NOT busy poll.

For static tests / small probes:

use a foreground timeout around:

```text
600–1200 seconds
```

For any permitted 8–16 batch GPU probe:

use:

```text
1200–1800 seconds
```

If asynchronous execution is absolutely necessary:

```text
sleep 300
```

before the first check.

Then:

```text
sleep 180
```

between later checks.

Do NOT poll every few seconds.

Do NOT use:

```text
watch
tail -f
while true
sleep 5
sleep 10
```

---

# 23. HARD STOP CONDITIONS

STOP and report immediately if any of these are found:

```text
mask target polarity reversed

normal/abnormal output channels reversed

factor N/A ordering reversed

utility BCE target incorrect

valid mask removes anomaly regions incorrectly

normal/anomaly balancing not actually implemented as intended

rho changes from .05

rho gets gradient

canonical epoch1 epsilon unexpectedly equals final-schedule value due to runner bug

gradient accumulation remainder mathematically underweighted

gradient attribution ratios ignore lambda weights

checkpoint/state needed for lambda probe is unavailable and would require rerunning
300 batches
```

Do not hide any of these by tuning lambdas.

---

# 24. FINAL DECISION REPORT

Return a compact decision report.

## A. Semantic correctness

```text
patch target polarity: PASS/FAIL

base logit polarity: PASS/FAIL

factor A-N polarity: PASS/FAIL

BCE utility semantics: PASS/FAIL

valid mask semantics: PASS/FAIL

normal/anomaly balancing: PASS/FAIL
```

## B. Why anomaly all-harm happened

Classify:

```text
SIGN/IMPLEMENTATION BUG

TARGET/MASK BUG

BALANCING BUG

or

CURRENT MODEL BEHAVIOR
```

Provide evidence from direct anomaly patch traces.

## C. Schedule/runtime

Report:

```text
canonical epoch1 epsilon

300-batch epsilon

canonical epoch20 epsilon

300-batch canonical/noncanonical exploration?
```

## D. Accumulation

Report exact semantics for:

```text
2162 samples
accum=6
remainder=2
```

and whether a fix is required before one epoch/final20.

## E. Gradient weighting

Report actual weighted:

```text
factor/main ratio
router/main ratio
total aux/main ratio
```

at current `.10/.10`.

Clearly distinguish:

```text
raw vs weighted
```

## F. Lambda probe

If valid, report compact candidates such as:

```text
factor_lambda
router_lambda
factor/main
router/main
total_aux/main
```

Recommend exactly ONE pair for the next controlled experiment.

Do NOT apply it to training yet.

## G. Next experiment

Recommend exactly one of:

```text
1. rerun a short corrected probe
   if a real implementation bug was found

2. run one full VisA epoch with unchanged config
   if all audits pass and scaling is acceptable

3. run one full VisA epoch with ONE explicitly recommended lambda pair
   if only auxiliary scaling requires correction
```

Do NOT execute the recommendation.

---

# 25. START NOW

Execute in this order:

```text
1. Verify Git state.

2. Audit patch target polarity.

3. Audit base normal/abnormal logit polarity.

4. Audit factor N/A polarity.

5. Produce direct anomaly patch trace.

6. Audit normal/anomaly balancing.

7. Audit epsilon schedule.

8. Audit full-epoch grad accumulation remainder.

9. Classify legacy null gradient diagnostics.

10. Verify raw-vs-weighted gradient attribution semantics.

11. If all semantic audits pass:
    perform fixed-state lambda gradient-only probe.

12. Fix only proven implementation bugs.

13. Run affected tests and focused suite if source changed.

14. Update artifacts and STATUS.json.

15. Create one local commit only if source changed.

16. Do NOT push.

17. Recommend exactly one next experiment.

18. STOP.

No one-epoch training.
No final20.
No medical evaluation.
```