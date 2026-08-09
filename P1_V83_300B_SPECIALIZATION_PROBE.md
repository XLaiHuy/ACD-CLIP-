# P1-v8.3 — 300-BATCH SPECIALIZATION / UTILITY / ROUTER VALUE PROBE

Continue from the CURRENT LAB MACHINE state.

Current project:

```text
Phase 4 → Progress 1 → candidate P1-v8.3
```

Repository is the existing lab clone of:

```text
https://github.com/XLaiHuy/ACD-CLIP-
```

Expected repo path is currently:

```text
/home/ai4/caohuy/ACD-CLIP-phase4
```

but verify it rather than blindly assuming it.

Current local implementation checkpoint:

```text
12d0b7fbb045b2c549e4be2c6c6daf85dae342f2
```

Nothing from that local commit has been pushed yet.

DO NOT pull/rebase/reset unless current Git inspection proves that is necessary.
This is a continuation of the already validated lab state.

Historical dirty/untracked artifacts must remain untouched.

---

# 1. CURRENT VERIFIED BASELINE

The previous bounded validation produced:

```text
P1-v8.3 implementation/runtime smoke PASS
```

Verified:

```text
Focused tests:
28 passed

Data:
VisA = 2162 images
six-medical = 9005 images
zero missing images/masks/classes/count mismatches

1-batch forward:
PASS

1-batch backward:
PASS

8-batch smoke:
PASS
2 optimizer steps
peak reserved VRAM ≈ 4.31 GiB

32-batch diagnostic:
PASS
6 optimizer steps

factor exact collapse:
NO

rho:
0.05 fixed
no gradient

STATE:
alive

CLASS/VAE:
alive

Text-LoRA:
alive

factor gradients:
alive

legacy auxiliaries:
OFF
```

The most important 32-batch diagnostic was approximately:

```text
Base        = 0.31954044
BestSingle  = 0.31192261
OracleMulti = 0.31149936

Uniform     = 0.31239092
SoftRouted  = 0.31239232
HardRouted  = 0.31240085

G_local     = 0.04426996
            ≈ 4.43%

G_multi     = 0.00164435
            ≈ 0.164%

capture     ≈ -0.000886
```

Utility:

```text
gain_rel_mean        ≈ 0.03964

informative_fraction = 0.0

teacher_entropy      ≈ 0.99891

teacher_max_prob     ≈ 0.26656

router utility loss  = 0.0
```

Structure:

```text
factor embedding effective rank
≈ 1.0053

factor embedding cosine mean
≈ 0.99925

factor patch correlation mean
≈ 0.8925

factor patch correlation max
≈ 0.9535

factor patch max difference
≈ 0.3204

STATE pairwise L2 mean
≈ 0.195
```

Interpretation:

```text
local residual mechanism shows positive value

BUT

meaningful multi-factor specialization is not yet proven

AND

utility teacher has not yet become informative

THEREFORE

router utility training has effectively not started
```

Do NOT interpret:

```text
router_top1_agreement = 0
capture ≈ 0
```

as a router architecture failure while:

```text
informative_fraction = 0
```

The teacher currently has no accepted informative target.

---

# 2. OBJECTIVE OF THIS SESSION

Do NOT modify architecture first.

Run ONE controlled:

```text
300 training-batch specialization/value probe
```

from a clean canonical initialization.

The purpose is to answer:

```text
Does P1-v8.3 naturally develop useful multi-factor specialization
when given more than 32 batches?
```

Specifically determine whether over 300 batches:

```text
factor representations separate

factor functions remain meaningfully different

utility differences sharpen

teacher entropy decreases

informative patches appear

G_local remains positive

G_multi grows

router utility supervision becomes active

SoftRouted begins to outperform Uniform
```

This is NOT a final training run.

This is NOT a medical evaluation.

---

# 3. IMPORTANT: START A FRESH CONTROLLED 300-BATCH RUN

Do NOT continue the optimizer state from the previous 32-batch diagnostic.

The 300-batch diagnostic should start from the same canonical initialization used
for a new P1-v8.3 train run:

```text
OpenAI CLIP pretrained
→ construct fresh P1-v8.3
→ fresh optimizer
→ fresh scheduler state
→ fixed canonical seed
→ run batches 1..300
```

Do NOT load:

```text
32-batch probe weights
Phase2B checkpoint
medical checkpoint
old P1 checkpoint
```

The reason is that the trajectory:

```text
batch 32
64
128
192
256
300
```

must belong to ONE coherent run.

Use the SAME canonical seed/config used by the validated lab smoke.

Inspect the existing evidence/config to recover the exact seed rather than guessing.

---

# 4. CANONICAL TRAINING CONTRACT

Preserve canonical math:

```text
P1-v8.3

OpenAI CLIP pretrained initialization only

FP32

TF32 OFF

AMP OFF
BF16 OFF
FP16 OFF

gradient checkpointing ON

img_size = 518

batch_size = 1

grad_accum_steps = 6

effective batch = 6

rho = .05 fixed

center_spread

center = .05
spread = .10

M = 4 factors

structured STATE/CLASS text

CLASS = decoder(mu)

dynamic text through SAME Text-LoRA

dense routing
```

Main v8.3 must keep OFF:

```text
Top-K prediction routing
load bias
equal factor balance
paired experts
cluster responsibility
functional diversity
trainable rho
selective-use/no-op gate
legacy semantic role losses
```

Do NOT change:

```text
tau_utility
entropy threshold
gain threshold
epsilon schedule
lambda_factor
lambda_router
router temperature
```

during the primary 300-batch run.

This run is diagnostic, not tuning.

---

# 5. GIT / SOURCE SAFETY BEFORE RUN

Before doing anything expensive:

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4

git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -5
```

Confirm the P1-v8.3 local fix commit is present.

Do NOT:

```text
git reset
git clean
git pull
git rebase
git stash -u
```

unless an actual repository problem is discovered.

Do not modify historical dirty artifacts.

If only runtime logs/runs are dirty/untracked:

continue.

---

# 6. DO NOT RERUN ALREADY-PASSED EXPENSIVE GATES

Do not rerun:

```text
data discovery
full data integrity
NVIDIA preflight
1-forward
1-backward
8-batch smoke
32-batch smoke
```

unless their artifacts are missing or a source change invalidates them.

Do not rerun the 28-test suite merely for ceremony if no source changes occur.

If you make SOURCE changes to diagnostics:

run:

```text
affected test(s)
```

then:

```bash
PYTHONPATH=. pytest -q \
  tests/test_p1_v83_runtime.py \
  tests/test_p1_v83_structured_utility.py \
  tests/test_setup_med_visa_data.py
```

once before the 300-batch run.

---

# 7. REUSE THE EXISTING SMOKE / DIAGNOSTIC RUNNER

Inspect the runner/controller that produced:

```text
runs/p1_v83_dev/8batch_smoke_retry2/smoke_summary.json

runs/p1_v83_dev/32_64batch_diag/smoke_summary.json
```

Reuse it.

Do NOT write a second independent training implementation.

If the current runner already supports:

```text
max_batches
```

or equivalent:

use it.

If it only supports 32/64:

make the MINIMUM instrumentation change required to support:

```text
300 batches
```

and milestone reporting.

Do not alter model/training math.

---

# 8. OUTPUT DIRECTORY

Use a new directory:

```text
runs/p1_v83_dev/300batch_specialization_probe/
```

Do not overwrite:

```text
8batch_smoke_retry2
32_64batch_diag
```

Save:

```text
config.json
run.log
trajectory.json
trajectory.csv
final_summary.json
```

and compact milestone artifacts.

---

# 9. MILESTONES

Capture diagnostics at:

```text
batch 32
batch 64
batch 128
batch 192
batch 256
batch 300
```

Because:

```text
grad_accum_steps = 6
```

the complete run should contain exactly:

```text
300 / 6 = 50 optimizer steps
```

unless the implementation has an explicitly documented different accumulation
flush policy.

Record actual optimizer step count at every milestone.

---

# 10. REPORT BOTH CUMULATIVE AND WINDOW METRICS

This is important.

At each milestone save:

## A. cumulative metrics

Example:

```text
1..32
1..64
1..128
...
1..300
```

## B. recent-window metrics

Example:

```text
1..32

33..64

65..128

129..192

193..256

257..300
```

Do not rely only on cumulative means.

Early near-uniform teacher behavior may otherwise hide later specialization.

---

# 11. STRUCTURAL METRICS AT EACH MILESTONE

Log:

```text
factor embedding pairwise cosine mean/max/min

factor embedding pairwise L2 mean/min/max

factor embedding effective rank

STATE pairwise L2 mean/min/max

factor_patch_logits pairwise correlation mean/max/min

factor_patch_logits max pairwise difference

factor_patch_logits std across factors

factor exact-collapse flag
```

Also preserve:

```text
factor gradient norms

factor gradient cosine mean/max/min
```

at selected milestone probes when available.

Do not run unnecessarily expensive full SVD every batch.

Only at the milestone batches.

---

# 12. UTILITY DIAGNOSTICS — ADD MORE RESOLUTION

Current:

```text
informative_fraction = 0
```

is insufficient to determine WHY the teacher rejects patches.

At every milestone report separately:

```text
gain_threshold_pass_fraction

entropy_threshold_pass_fraction

informative_fraction
```

where:

```text
informative =
gain condition
AND
entropy condition
```

Also log:

```text
best_gain_rel mean
best_gain_rel std

best_gain_rel p50
best_gain_rel p75
best_gain_rel p90
best_gain_rel p95
best_gain_rel p99

best-second utility margin:
mean
p50
p90
p95
p99
```

Log normalized utility teacher entropy:

```text
mean
std
p01
p05
p10
p50
p90
```

Log:

```text
teacher max probability:
mean
p50
p90
p95
p99
```

This is needed to distinguish:

```text
gain too weak
```

from:

```text
entropy gate too strict
```

from:

```text
factors genuinely equivalent
```

---

# 13. UTILITY VALUE METRICS

At each milestone calculate the CORRECT definitions:

```text
Base

BestSingle

OracleMulti

Uniform

SoftRouted

HardRouted
```

Then:

```text
G_local =
(Base - OracleMulti) / Base
```

and:

```text
G_multi =
(BestSingle - OracleMulti) / Base
```

Router capture:

```text
capture_denominator =
Uniform - OracleMulti
```

If denominator is valid:

```text
capture =
(Uniform - SoftRouted)
/
(Uniform - OracleMulti)
```

Otherwise:

```text
capture_valid = false
```

Never substitute Base for Uniform.

---

# 14. ADD NORMAL / ANOMALY BREAKDOWN

Where practical, report utility evidence separately for:

```text
normal-region valid patches

anomaly-region valid patches
```

At least record:

```text
gain_rel mean

best_gain_rel mean

teacher entropy mean

informative fraction

all-harm fraction
```

for each side.

This helps identify whether:

```text
background dominates
```

or:

```text
anomaly factors lack specialization
```

Do NOT factor-balance.

---

# 15. WINNER DISTRIBUTION

At each milestone log:

```text
winner share F1
winner share F2
winner share F3
winner share F4
```

But do NOT treat:

```text
25% each
```

as a target.

A dominant factor may be legitimate if:

```text
BestSingle ≈ OracleMulti
```

The distribution is diagnostic only.

---

# 16. ROUTER METRICS

At each milestone report:

```text
router entropy

router usage F1/F2/F3/F4

router top-1 agreement with utility winner

teacher-router KL

utility router loss

number/fraction of router-supervised informative patches
```

IMPORTANT:

When:

```text
informative_fraction == 0
```

record explicitly:

```text
router utility objective inactive due to teacher gate
```

Do not claim router failure.

---

# 17. GRADIENT HEALTH

Do not perform expensive gradient attribution every batch.

At minimum inspect at:

```text
batch 32
batch 128
batch 300
```

or nearby optimizer-step-aligned probes.

Record:

```text
main task gradient scale

utility factor gradient scale

utility router gradient scale

Text-LoRA gradient

STATE path gradient

VAE mu gradient

VAE decoder gradient

router gradient

factor gradients

rho gradient
```

rho must remain:

```text
None
```

exactly.

If router utility loss remains zero:

distinguish:

```text
router total/model gradient exists from another path
```

from:

```text
router utility objective gradient
```

Do NOT interpret a nonzero generic router gradient as proof the utility router
teacher is active.

---

# 18. GPU / MEMORY

At every milestone record compact GPU metrics:

```text
allocated

reserved

peak allocated

peak reserved
```

No need to call `nvidia-smi` every milestone.

Do not busy poll GPU state.

Hard stop on:

```text
OOM
non-finite tensors
non-finite loss
non-finite gradient
unexpected rho change
```

---

# 19. LONG-RUN EXECUTION / SLEEP POLICY

THIS IS IMPORTANT.

Do NOT busy-poll the 300-batch run.

Preferred behavior:

## If Codex can execute the training command in the foreground with a long timeout

Use one long timeout:

```text
3600 seconds
```

and simply let the command run.

Do NOT open another polling loop.

Do NOT repeatedly run:

```text
ps
nvidia-smi
tail
ls
cat
```

while the command is still executing.

## If Codex must launch the task asynchronously/background

After confirming the process started successfully:

FIRST WAIT:

```bash
sleep 600
```

That is 10 minutes.

Then perform ONE status check.

If still running and healthy:

```bash
sleep 300
```

before the next status check.

Use approximately:

```text
first wait:        600 s
subsequent waits: 300 s
```

Do not check more frequently than every 5 minutes.

Do not exceed roughly 3–4 status checks unless the run genuinely lasts much
longer than expected.

If output is clearly still progressing:

WAIT.

Do not cancel simply because no terminal line appeared recently.

Do NOT use:

```bash
watch ...
tail -f ...
while true ...
sleep 5 ...
sleep 10 ...
sleep 30 ...
```

No busy polling.

## If the run appears stalled

Only classify it as stalled when there is evidence such as:

```text
process alive but no new progress for a long interval

GPU idle unexpectedly

no file timestamp/progress change across widely separated checks

or explicit error
```

One quiet period is not enough.

---

# 20. DO NOT MODIFY ARCHITECTURE DURING THE PRIMARY RUN

The 300-batch primary probe must be one controlled architecture/config.

Do NOT respond mid-run to weak metrics by changing:

```text
tau
epsilon
threshold
lambda
router
factor identity
diversity loss
rho
```

Finish the primary 300-batch evidence first.

---

# 21. OPTIONAL POST-RUN TEACHER SENSITIVITY AUDIT

ONLY AFTER the canonical 300-batch run finishes.

ONLY if:

```text
informative_fraction remains near 0
```

perform a NO-OPTIMIZER-STEP / OFFLINE sensitivity audit.

This must NOT train or alter the saved canonical trajectory.

Use the same utility/gain tensors or a fixed deterministic probe batch.

Evaluate what the teacher WOULD look like under a tiny diagnostic grid such as:

```text
tau_utility:

0.05   canonical
0.03
0.02
```

and entropy thresholds:

```text
0.98   canonical
0.99
0.995
```

Do NOT select a new canonical value automatically.

For each diagnostic combination report:

```text
teacher entropy

teacher max probability

informative fraction

winner shares
```

The purpose is only to distinguish:

```text
A. true factor utility ambiguity

vs

B. teacher temperature/filter calibration bottleneck
```

Do not train under these alternate values.

Do not change architecture.

---

# 22. DECISION RULES AFTER 300 BATCHES

Do not use one metric alone.

Compare trajectory.

## CASE A — specialization is emerging

Evidence could look like:

```text
G_local stays positive

factor outputs remain non-collapsed

effective rank increases

factor correlation decreases or meaningful functional differences persist

best-second utility margin increases

teacher entropy decreases

informative_fraction becomes > 0

router utility loss becomes active

G_multi increases materially
```

Then:

```text
P1-v8.3 specialization is emerging
```

and a longer controlled stability run may be justified.

Do NOT automatically launch final20.

---

## CASE B — local branch useful but task appears low-mode

Evidence:

```text
G_local clearly > 0

BestSingle ≈ OracleMulti

G_multi remains very low

one factor may dominate

factor outputs can differ but per-patch oracle selection adds little
```

Then report:

```text
local semantic residual is useful,
but evidence for four useful task modes is weak
```

Do NOT artificially force equal factor specialization.

---

## CASE C — factors differ, but teacher calibration blocks routing

Evidence:

```text
functional factors differ

best_gain_rel meaningful

but canonical q remains near-uniform

entropy gate rejects nearly everything

offline lower-tau sensitivity creates informative teacher
```

Then report:

```text
utility-teacher calibration candidate
```

Do NOT change architecture yet.

A later controlled config probe may test tau/threshold.

---

## CASE D — factor specialization itself is weak

Evidence:

```text
rank remains ≈1

embedding cosine remains ≈1

factor function correlations approach collapse

best-second utility margin remains tiny

G_multi remains tiny

teacher remains uniform
```

Then follow specialization rescue order.

Do NOT tune router first.

---

# 23. PROVISIONAL ENGINEERING GATES

Use:

```text
G_local <= 0
→ local factor mechanism itself lacks oracle value

G_local > 0
→ continue analysis
```

For G_multi:

```text
< 0.02
→ weak multi-mode evidence

0.02–0.05
→ borderline

> 0.05
→ meaningful engineering evidence
```

These are heuristics, not theorems.

Do not turn them into scientific claims.

---

# 24. DO NOT RUN MEDICAL EVALUATION

Do NOT run:

```text
Brain
Liver
Retina
ClinicDB
ColonDB
Kvasir
```

during this probe.

Medical data remains frozen test-only.

---

# 25. DO NOT RUN FINAL 20 EPOCHS

Regardless of the result:

DO NOT execute:

```text
scripts/run_p1_v83_final20.sh
```

in this task.

Also do NOT execute:

```text
scripts/test_p1_v83_6medical_epoch20_exact.sh
```

They are prepared launchers only.

---

# 26. DO NOT PUSH

Do not push anything.

Current local P1-v8.3 fix commit remains:

```text
12d0b7fbb045b2c549e4be2c6c6daf85dae342f2
```

If NO source change is needed:

do not create another source commit just for runtime results.

If instrumentation source changes ARE needed:

1. make the smallest diagnostics-only change;
2. run affected tests;
3. run focused suite once;
4. commit locally with a clear message.

Example:

```text
chore(p1-v8.3): add specialization trajectory diagnostics
```

Do NOT push.

Do not commit:

```text
runs/
data/
checkpoints/
weights/
this prompt file
```

---

# 27. UPDATE STATUS

Update:

```text
runs/p1_v83_dev/STATUS.json
```

Add a stage such as:

```json
"specialization_300": {
  "status": "PASS | FAIL",
  "evidence": "runs/p1_v83_dev/300batch_specialization_probe/final_summary.json"
}
```

PASS here only means:

```text
300-batch diagnostic executed correctly
```

It does NOT mean the architecture scientifically passed.

Add separate interpretation fields for:

```text
local_value
multi_mode_evidence
teacher_informativeness
router_utility_activity
factor_collapse
```

---

# 28. FINAL REPORT

At completion return a concise report with:

## Execution

```text
git SHA
seed
300 batches executed
optimizer steps
runtime duration
peak VRAM
```

## Trajectory table

Produce one compact table:

```text
batch
optimizer_steps

G_local
G_multi

Base
BestSingle
OracleMulti
Uniform
SoftRouted
HardRouted

capture

gain_pass_fraction
entropy_pass_fraction
informative_fraction

teacher_entropy
teacher_max_probability

best_second_margin

factor_effective_rank
factor_embedding_cos_mean
factor_patch_corr_mean

all_harm_fraction
```

for:

```text
32
64
128
192
256
300
```

Prefer recent-window values alongside cumulative values when useful.

## Final batch-300 structural state

Report:

```text
factor exact collapse?
effective rank
embedding cosine
function correlation
STATE separation
```

## Utility teacher

Report:

```text
gain condition pass fraction

entropy condition pass fraction

both/informative fraction
```

Explicitly state which gate is blocking teacher activity.

## Router

Report:

```text
utility router loss
router utility gradient
router agreement
SoftRouted vs Uniform
capture
```

Do not call it router failure if the teacher remains inactive.

## Decision

Classify evidence as one of:

```text
A. specialization emerging

B. useful local branch but weak/low multi-mode evidence

C. teacher calibration bottleneck candidate

D. factor specialization weakness
```

More than one may apply, but explain the dominant diagnosis.

## Recommendation

Recommend exactly ONE next experimental step.

Do not launch it.

---

# 29. HARD STOP CONDITIONS

Stop and diagnose immediately if:

```text
NaN / Inf

OOM

rho receives gradient

rho changes from .05

legacy auxiliary unexpectedly activates

Top-K affects prediction

expert activates

load bias activates

balance activates

functional diversity activates

selective-use gate activates

dataset unexpectedly changes

Phase2B checkpoint is loaded

factor outputs become exactly identical on two consecutive probes
```

Do not repair these by adding new losses during the same run.

---

# 30. START NOW

Immediate sequence:

```text
1. Verify current local Git state.

2. Read existing 32-batch evidence/config.

3. Reuse the existing diagnostic runner.

4. Add only missing trajectory diagnostics if required.

5. If source changes:
   affected tests
   → focused 28+ suite once.

6. Start ONE fresh canonical P1-v8.3 run from OpenAI CLIP.

7. Run exactly 300 training batches.

8. Capture milestones:
   32 / 64 / 128 / 192 / 256 / 300.

9. Use long command timeout.

10. If asynchronous:
    sleep 600 before first check,
    then sleep 300 between checks.
    NO busy polling.

11. Finish canonical run before any sensitivity audit.

12. If teacher remains inactive:
    run optional OFFLINE teacher sensitivity analysis only.

13. Update STATUS.json.

14. Report evidence and one recommended next step.

15. STOP.

Do NOT run final20.
Do NOT run medical evaluation.
Do NOT push.
```