You are continuing the ACD-CLIP P1 research workflow after the P1-v8.3
EXIT_FOR_DISCUSSION decision.

This task is an evidence-driven architecture rescue focused ONLY on the local
factor mechanism.

The objective is:

1. determine exactly WHY P1-v8.3 produces:
   - useful normal correction,
   - harmful anomaly correction,
   - almost-identical factors,
   - zero informative router supervision;

2. validate whether TRUE RESIDUAL FACTORS + an explicit ACT/NO-ACT mechanism
   solve the root cause;

3. only if needed, add the minimum factor-specific capacity required to create
   genuine functional specialization;

4. validate development candidates through:
   no-training forensic audit
   → 8-batch smoke
   → fresh corrected 300B
   → fresh 1 epoch
   → fresh 3 epochs;

5. stop immediately if evidence requires a fundamentally different scientific
   hypothesis.

IMPORTANT:

DO NOT COMMIT.
DO NOT PUSH.
DO NOT git add.
DO NOT git commit.
DO NOT git cherry-pick.
DO NOT git reset/rebase/clean/stash destructively.

Leave all source modifications UNCOMMITTED for user review.

DO NOT run final20.
DO NOT run ANY medical evaluation.
DO NOT use medical datasets for tuning or validation.

============================================================
0. CURRENT STATE
============================================================

Repository:

https://github.com/XLaiHuy/ACD-CLIP-

Current isolated worktree:

/home/ai4/caohuy/ACD-CLIP-p1v83-autopilot

Original lab worktree:

/home/ai4/caohuy/ACD-CLIP-phase4

DO NOT modify the original lab worktree.

Current branch:

autopilot/p1-v83-root-cause-d58b84bc

Historical Phase-4 base:

d58b84bcecb9c4d22bdef321ea0ca28bd3b6745b

Existing local development commits include:

4b57b24
primary-anchored factor surgery

459e546
corrected-300B exit decision

These are already historical context for this task.

DO NOT create another commit.

First run:

cd /home/ai4/caohuy/ACD-CLIP-p1v83-autopilot

pwd
git branch --show-current
git status --short
git log --oneline -6

Preserve all existing artifacts.

============================================================
1. AUTHORITATIVE P1-v8.3 RESULT
============================================================

Corrected 300B attempt1:

fresh OpenAI CLIP only

300 batches
50 optimizer steps
batch1
grad_accum6

runtime:
~294 seconds

Optimization:

PASS

Primary-anchored factor surgery:

PASS

MAIN exact-change max:
0

correction reconstruction error:
0

raw main/factor conflicts:
19 / 50 windows

weighted factor/main:

median:
0.167

p95:
0.713

Therefore optimizer scale/conflict is NOT considered the remaining primary
root cause.

------------------------------------------------------------
SEMANTIC FAILURE
------------------------------------------------------------

At batch300:

G_local:
3.43%

G_multi:
0.303%

BestSingle:
0.075662

OracleMulti:
0.075426

factor patch correlation:
0.998699

factor embedding effective rank:
1.0055

anomaly all-harm:
99.90%

anomaly best gain:
-3.19%

normal best gain:
+2.94%

teacher normalized entropy:
~0.99949

teacher max probability:
~0.259

router informative fraction:
0%

router utility loss:
0

Normal valid patches:
~1,064,118

Anomaly valid patches:
~6,258

Therefore:

- local correction helps normal patches;
- almost every anomaly patch is harmed by every factor;
- four factors produce nearly the same function;
- teacher cannot confidently distinguish factor identities;
- router therefore receives zero utility supervision.

============================================================
2. CURRENT ROOT-CAUSE HYPOTHESES
============================================================

Do NOT assume these hypotheses are true.

Test them.

------------------------------------------------------------
H1 — COMMON-MODE / ABSOLUTE-CORRECTION PROBLEM
------------------------------------------------------------

Current local factor evidence is approximately:

l_m =
10 * (
  similarity(patch, abnormal_factor_m)
  -
  similarity(patch, normal_factor_m)
)

Candidate:

z_m =
z_base + rho * l_m

rho=.05

But factor banks are constructed around a strong shared center.

Conceptually:

factor_m =
COMMON + delta_m

Current correction therefore behaves approximately like:

rho * (COMMON + delta_m)

instead of:

rho * delta_m

If COMMON is a normalizing direction:

normal patch:
helpful

anomaly patch:
harmful

then all four factors can simultaneously harm anomaly patches even if their
small residual differences are not identical.

This may explain:

normal gain positive
+
anomaly gain negative
+
factor correlation ~.999.

------------------------------------------------------------
H2 — FORCED-ACTION PROBLEM
------------------------------------------------------------

Current dense router satisfies:

sum_m pi_m = 1

Therefore the model always uses a weighted combination of active factors.

It cannot explicitly say:

"none of these corrections should be applied."

If all four factor candidates are harmful:

the router still produces a correction.

------------------------------------------------------------
H3 — SYMMETRY / SPECIALIZATION DEADLOCK
------------------------------------------------------------

Current factor utility teacher is approximately uniform:

q ≈ [.25,.25,.25,.25]

Then responsibility is also nearly uniform.

Therefore all four factors receive almost the same utility objective.

At the same time the dense router is also approximately uniform.

This can create:

similar factor
→ similar utility
→ uniform teacher
→ similar gradients
→ even more similar factors

which prevents specialization.

------------------------------------------------------------
H4 — FACTOR-SPECIFIC SIGNAL MAY DIE INSIDE THE PROMPT PATH
------------------------------------------------------------

P1-v8.3 structured contexts share much of their machinery.

The STATE token is factor-specific, but:

shared contexts
shared text encoder
shared normal/abnormal semantics
shared fusion center

may make factor-specific differences too weak by the time they reach patch
functions.

Need locate EXACTLY where collapse occurs.

============================================================
3. DO NOT IMPLEMENT ANYTHING YET
============================================================

First perform a NO-TRAIN forensic audit on the completed corrected-300B
checkpoint.

No optimizer step.

No backward needed unless explicitly required for diagnostics.

No source behavior changes.

No medical data.

Use VisA training/development evidence only.

Create a dedicated directory such as:

runs/p1_v83_dev/v84_forensic_audit/

Do not delete historical artifacts.

============================================================
4. FORENSIC AUDIT A — EXACT NO-OP REFERENCE
============================================================

Identify the exact local-path no-op/reference bank.

Current source already exposes an object equivalent to:

expected_noop_pre_expert_bank

Verify mathematically and by source trace what it represents.

Do NOT assume it equals z_base.

These are different concepts:

z_base:
base segmentation margin used by the utility teacher

l_ref:
local semantic logit produced by the exact no-op/reference local bank

Compute:

l_ref =
10 * (
 sim(patch, abnormal_noop)
 -
 sim(patch, normal_noop)
)

For every active factor:

l_m =
current factor patch margin

Define counterfactual residual evidence:

delta_m =
l_m - l_ref

Then candidate residual logits:

z_residual_m =
z_base + rho * delta_m

rho stays:

0.05

No trainable rho.

============================================================
5. FORENSIC AUDIT B — THREE ORACLES
============================================================

Evaluate on the SAME stored/evaluated VisA patch support:

------------------------------------------------------------
CURRENT ORACLE
------------------------------------------------------------

min loss over:

z_base + rho*l_1
...
z_base + rho*l_4

------------------------------------------------------------
ORACLE + NO-OP
------------------------------------------------------------

min loss over:

z_base
z_base + rho*l_1
...
z_base + rho*l_4

------------------------------------------------------------
RESIDUAL ORACLE + NO-OP
------------------------------------------------------------

min loss over:

z_base

z_base + rho*(l_1-l_ref)
...
z_base + rho*(l_4-l_ref)

============================================================
6. REPORT NORMAL AND ANOMALY SEPARATELY
============================================================

For BOTH normal and anomaly patches report:

count

mean l_ref

for each factor:
mean l_m

mean delta_m

std delta_m

P(l_m > 0)
P(l_m < 0)

P(delta_m > 0)
P(delta_m < 0)

best absolute-factor gain

best residual-factor gain

current all-harm fraction

residual all-harm fraction

no-op selected fraction

residual-no-op selected fraction

Base loss

Current Oracle loss

Oracle+NoOp loss

ResidualOracle+NoOp loss

Do not hide class imbalance by reporting only a global average.

============================================================
7. FORENSIC AUDIT C — COMMON-MODE DECOMPOSITION
============================================================

For each patch define:

factor_mean =
mean_m(l_m)

factor_residual_m =
l_m - factor_mean

Measure:

variance explained by common factor mean

variance across factor residuals

sign agreement across F1..F4

pairwise correlation of:

absolute l_m

residual delta_m

mean-subtracted factor evidence

If absolute correlation≈1 but residual correlation falls substantially:

COMMON-MODE DOMINANCE is supported.

If residual correlation also≈1:

true factor functional collapse is supported.

============================================================
8. FORENSIC AUDIT D — STAGE-BY-STAGE COLLAPSE TRACE
============================================================

Trace factor identity through:

concept slots
↓
normal/abnormal queries
↓
normal/abnormal prototypes
↓
state_delta_raw
↓
state_delta_generated
↓
state_delta_with_identity
↓
STATE tokens
↓
dynamic contexts
↓
dynamic text raw
↓
dynamic text normalized
↓
factor bank
↓
abnormal-normal factor direction
↓
patch logits
↓
residual patch logits

At each meaningful stage report:

pairwise cosine mean/min/max

pairwise L2

effective rank

factor-wise std

where appropriate:
functional correlation

Determine the first stage where:

distinct factors
→ almost identical factors.

Classify the bottleneck as one of:

STATE_GENERATION_COLLAPSE

PROMPT_CONTEXT_COLLAPSE

TEXT_ENCODER_COLLAPSE

CENTER_SPREAD_FUSION_COLLAPSE

PATCH_FUNCTION_COLLAPSE

NO_SINGLE_STAGE_COLLAPSE

============================================================
9. FORENSIC DECISION TREE
============================================================

------------------------------------------------------------
CASE F-A
RESIDUAL ORACLE MATERIALLY IMPROVES ANOMALY
------------------------------------------------------------

Evidence pattern:

current absolute factors harmful

but:

residual candidate set produces positive anomaly gains
and/or materially reduces anomaly all-harm

Interpretation:

COMMON absolute component is a major root cause.

Action:

authorize P1-v8.4-A:

TRUE RESIDUAL FACTORS
+
ACT/NO-ACT mechanism.

------------------------------------------------------------
CASE F-B
NO-OP HELPS BUT RESIDUALIZATION DOES NOT CREATE USEFUL ANOMALY MODES
------------------------------------------------------------

Evidence:

Oracle+NoOp protects Base strongly

but residual factors remain mostly harmful/redundant.

Interpretation:

forced-action is real,
but factor generator still lacks useful anomaly modes.

Action:

still implement ACT/no-op for safety,
but mark specialization unresolved.

P1-v8.4-A may be run to determine whether training with explicit abstention changes
the learned factor geometry.

Do NOT claim residualization alone solves S2.

------------------------------------------------------------
CASE F-C
RESIDUAL FACTORS BECOME FUNCTIONALLY DISTINCT
------------------------------------------------------------

Evidence:

residual correlation drops substantially

Residual G_multi increases

some factors win different patches

Interpretation:

absolute COMMON mode was hiding meaningful factor differences.

Action:

P1-v8.4-A only.

Do NOT add extra capacity yet.

------------------------------------------------------------
CASE F-D
RESIDUAL FACTORS REMAIN NEAR-IDENTICAL
------------------------------------------------------------

Evidence:

residual corr remains near 1

Residual G_multi remains near 0

effective rank remains near 1

Interpretation:

factor generation itself lacks specialization.

Do NOT immediately implement a large MoE.

First complete the stage-by-stage collapse trace.

If the collapse location is clear:

prepare the minimum factor-specific residual capacity branch.

This becomes P1-v8.4-B, but only AFTER v8.4-A evidence if v8.4-A remains
scientifically plausible.

------------------------------------------------------------
CASE F-E
NO-OP AND RESIDUAL ORACLES BOTH SHOW ESSENTIALLY NO POTENTIAL
------------------------------------------------------------

Interpretation:

the current local factor hypothesis is not supported.

EXIT_FOR_DISCUSSION.

Do not train another architecture automatically.

============================================================
10. P1-v8.4-A — TRUE RESIDUAL FACTORS
============================================================

If authorized by forensic evidence:

change local correction semantics from:

factor absolute margin

to:

factor residual relative to exact no-op local reference.

For each factor:

delta_m =
factor_patch_logit_m
-
noop_reference_patch_logit

Final active local correction:

rho * delta_m

NOT:

rho * factor_patch_logit_m

No-op must satisfy EXACTLY:

delta_noop = 0

Therefore:

ACT=0
→ correction exactly zero
→ final local branch preserves Base.

Add explicit diagnostics:

noop_reference_logit

factor_absolute_logits

factor_residual_logits

residual_factor_correlation

residual_effective_rank

absolute_vs_residual_variance

residual sign distribution

============================================================
11. P1-v8.4-A — SEPARATE ACT FROM WHICH-FACTOR
============================================================

Do NOT represent no-op simply by forcing factor router probabilities to zero.

Introduce an explicit ACT gate:

a(p) ∈ [0,1]

Factor router remains:

pi_m(p)

sum_m pi_m = 1

Final local correction:

correction(p) =
a(p)
*
rho
*
sum_m pi_m(p) * delta_m(p)

This cleanly separates:

ACT:
should local correction be applied?

ROUTER:
if yes, which factor mixture is useful?

============================================================
12. ACT TEACHER
============================================================

Build ACT supervision from the detached RESIDUAL utility teacher.

For each valid patch:

best_residual_gain =
max_m(
  relative BCE improvement from residual factor m
)

Use three zones:

POSITIVE ACT SUPPORT:

best_residual_gain > +0.02

target_act = 1

NEGATIVE ACT SUPPORT:

best_residual_gain <= 0

target_act = 0

AMBIGUOUS:

0 < best_residual_gain <= .02

do not supervise ACT strongly.

Reason:

existing .02 utility threshold is already part of the development contract.

Do not invent many new thresholds.

ACT teacher is detached.

============================================================
13. ACT LOSS MUST NOT REINTRODUCE CLASS IMBALANCE
============================================================

Normal patches greatly outnumber anomaly patches.

Do NOT use a naive global mean BCE for ACT if it causes normal support to dominate.

Audit ACT target distributions separately by:

normal/anomaly

positive/negative/ambiguous.

Prefer the already validated effective-number region weighting principle:

beta=.999

for valid ACT support if class imbalance would otherwise dominate.

Do not reuse the OLD hard 50/50 region-mean implementation.

Use patch-level effective-number weights + one normalized weighted mean.

Report ACT gradient/main distributions before accepting it.

============================================================
14. ACT MODEL
============================================================

Use the minimum-capacity gate that reuses existing router/local patch features.

Do NOT add a second vision encoder.

Do NOT rerun CLIP.

Prefer a small head on already-computed patch/router features.

Example conceptual form:

existing patch/router representation
→ small LayerNorm/Linear/MLP
→ scalar act logit
→ sigmoid

Keep parameter overhead small and explicit.

Do not add a deep gate network unless measured evidence requires it.

============================================================
15. FACTOR ROUTER TEACHER AFTER ACT DECOMPOSITION
============================================================

Factor router supervision should answer ONLY:

"which factor?"

Do not require the router to decide whether local should be used.

Router supervision support:

ACT-positive patch
AND
factor utility identity sufficiently distinguishable.

Use existing factor utility q and entropy logic only inside ACT-positive support.

If factors are tied:

ACT may still learn ON/OFF

while factor router may remain unsupervised.

This is intentional.

============================================================
16. P1-v8.4-A INFERENCE
============================================================

Use continuous soft ACT probability initially:

a = sigmoid(act_logit)

Do not introduce a hard threshold into final prediction without calibration evidence.

Prediction:

delta_soft =
sum_m pi_m * delta_m

correction =
a * rho * delta_soft

final local logits =
base + correction

No-op identity invariant:

if a=0:

final local logits == base

within floating-point tolerance.

============================================================
17. PRESERVE EVERYTHING ELSE
============================================================

Do NOT automatically change:

OpenAI CLIP initialization

DFG

SS2D

global text mode

phase2b_hybrid

structured prompt layout

CLASS token semantics

STATE semantics

rho=.05

number of factors=4

dense factor routing

effective-number beta=.999

primary-anchored factor surgery

lambda_factor=.03

legacy auxiliary losses

medical scoring

test protocol.

Only change what is required for:

true residual semantics
+
ACT/no-act.

============================================================
18. UTILITY FACTOR LOSS FOR RESIDUAL FACTORS
============================================================

After changing factor semantics:

the factor utility teacher MUST evaluate:

z_base + rho*delta_m

not the old:

z_base + rho*l_m.

Recompute:

loss_per_factor

gain_rel

q_utility

responsibility

best gain

all-harm

G_local

G_multi

Oracle

using the residual candidate semantics.

Do not mix old absolute utility teacher with new residual predictions.

============================================================
19. PRIMARY-ANCHORED FACTOR SURGERY
============================================================

Keep the validated primary-preserving surgery.

MAIN gradient remains unchanged.

Only harmful auxiliary factor shared-semantic component may be projected.

Because factor utility semantics changed:

redo no-step gradient magnitude/conflict diagnostics before 300B.

Do not assume old .03 remains ideal.

However:

start from lambda_factor=.03

and change it only if current residual-gradient evidence proves scale changed
materially.

Router lambda remains .10 unless evidence changes.

ACT loss needs its own lambda calibration.

============================================================
20. NEW NO-STEP GRADIENT CALIBRATION
============================================================

Before GPU training:

use natural six-microbatch optimizer windows.

Measure:

main

residual factor utility

router utility

ACT utility

for relevant shared parameter groups.

Report:

raw norm/main

weighted norm/main

median
p75
p90
p95
max

cos(main,factor)

cos(main,act)

true combined aux/main.

Calibrate ACT lambda analytically.

Do NOT brute-force train lambdas.

Choose the largest stable ACT lambda whose common-window gradient is meaningful
but does not systematically dominate main.

If ACT gradient scale cannot be made stable with static weighting:

EXIT_FOR_DISCUSSION.

Do not automatically introduce GradNorm.

============================================================
21. UNIT / SOURCE TESTS
============================================================

Add tests for at least:

1.
no-op residual exactly zero.

2.
factor identical to reference:
delta=0.

3.
ACT=0:
final prediction exactly equals Base.

4.
ACT=1:
final uses routed residual.

5.
soft ACT interpolation correct.

6.
residual teacher uses residual candidates, not absolute factor logits.

7.
negative anomaly residual gain produces ACT target 0.

8.
positive gain >.02 produces ACT target 1.

9.
ambiguous 0..02 handled according to contract.

10.
router teacher only supervises factor identity where ACT-positive and informative.

11.
ACT loss class/support normalization correct.

12.
rho fixed .05.

13.
primary MAIN gradient unchanged.

14.
grad_accum=6 remains exact.

15.
checkpoint/config metadata reflects P1-v8.4-A semantics.

16.
old P1-v8.3 checkpoints remain readable if backward compatibility is expected.

17.
P1-v8.3 path unchanged when new mechanism disabled.

Run:

focused pytest

py_compile

bash -n changed scripts

git diff --check

DO NOT COMMIT.

============================================================
22. VERSIONING
============================================================

If mechanism changes as above, research candidate becomes:

P1-v8.4-A

Do not call it H6.

Legacy h6 code namespace may remain.

P1-v8.3 must remain reproducible.

Do not silently change P1-v8.3 behavior.

Use explicit version/config flags.

============================================================
23. 8-BATCH SMOKE
============================================================

Only after CPU/no-step gates PASS.

Fresh OpenAI CLIP initialization.

8 batches.

batch1
grad_accum6

FP32
TF32 OFF
AMP OFF

gradient checkpointing ON.

Verify:

2 optimizer windows:
6 + remainder2

ACT forward/backward works

residual correction works

ACT=0 identity path is numerically exact in unit test

no NaN

no gradient duplication

MAIN exact-change=0

rho=.05

checkpoint metadata correct.

============================================================
24. GPU OCCUPANCY
============================================================

Before long GPU commands:

nvidia-smi

Do NOT kill another user's process.

If GPU is actively occupied:

STOP with:

WAITING_FOR_GPU

Do not restart CPU audits.

============================================================
25. LONG-RUN TIMEOUT POLICY
============================================================

Use long foreground timeouts.

Minimum recommended:

8-batch:
5 min

300B:
20 min

1 epoch:
90 min

3 epochs:
240 min

A tool timeout is NOT evidence that the training process failed.

If a foreground call times out:

check:

process
GPU
log
RUN_DIR

before doing anything.

Never start a duplicate training run unless the original process is confirmed
dead/failed.
============================================================
25B. ADAPTIVE WAIT / POLLING POLICY
============================================================

For long-running commands, choose the waiting/polling interval based on the
EXPECTED remaining runtime.

Goal:

- avoid wasting tokens by polling too frequently;
- avoid unnecessarily waiting a long time after a command has already finished;
- never restart a healthy process merely because there was no recent output.

------------------------------------------------------------
A. ESTIMATE RUNTIME FIRST
------------------------------------------------------------

After the first few batches/iterations, estimate:

seconds_per_iteration

estimated_remaining_seconds =
remaining_iterations * seconds_per_iteration

Update this estimate only when useful.

Do NOT repeatedly recompute/report ETA every few seconds.

------------------------------------------------------------
B. ADAPTIVE POLL INTERVALS
------------------------------------------------------------

Use approximately:

Expected remaining time <= 2 minutes:
    poll every 20–30 seconds

2–10 minutes:
    poll every 45–60 seconds

10–30 minutes:
    poll every 2–3 minutes

30–90 minutes:
    poll every 5 minutes

90 minutes–4 hours:
    poll every 8–10 minutes

>4 hours:
    poll every 10–15 minutes

These are guidelines, not rigid timing requirements.

If the process emits useful progress continuously in the foreground:
prefer simply waiting for the command instead of additional polling.

------------------------------------------------------------
C. NEAR-COMPLETION ADJUSTMENT
------------------------------------------------------------

When estimated remaining runtime becomes short, reduce the poll interval.

Example:

3-epoch run initially has ~2 hours remaining:
    poll every ~8–10 min

when ~25 min remain:
    poll every ~3 min

when ~5 min remain:
    poll every ~45–60 sec

when <2 min remain:
    poll every ~20–30 sec

This avoids both excessive token use and unnecessary post-completion waiting.

------------------------------------------------------------
D. POLL QUIETLY
------------------------------------------------------------

A polling turn should be minimal.

Prefer checking only:

- process alive/dead;
- latest progress counter;
- latest loss finite/nonfinite;
- RUN_DIR/log modification;
- GPU state only when necessary.

Do NOT reread full logs on every poll.

Do NOT repeatedly summarize unchanged metrics.

Do NOT rerun expensive diagnostics during waiting.

Do NOT repeatedly call nvidia-smi if the training process is clearly progressing.

Only perform full analysis AFTER the stage completes or when a failure signal
appears.

------------------------------------------------------------
E. FAILURE SIGNALS THAT JUSTIFY EARLY INSPECTION
------------------------------------------------------------

Immediately inspect rather than waiting for the next scheduled poll if any of
these appear:

- NaN / Inf;
- CUDA OOM;
- fatal traceback;
- process exit;
- progress stops for substantially longer than expected;
- RUN_DIR/log stops changing unexpectedly;
- GPU utilization drops to idle while the process should still be computing;
- optimizer/batch counters violate the expected contract.

Otherwise, do not interrupt a healthy run.

------------------------------------------------------------
F. TOOL TIMEOUT VS POLLING
------------------------------------------------------------

Prefer a sufficiently long foreground timeout whenever supported.

Polling is a fallback for tool/session limitations, not the default if a command
can safely remain attached.

If the interface/tool times out while the underlying process may still be alive:

1. do NOT relaunch;
2. check the existing process/session;
3. inspect only enough log/progress state to confirm health;
4. continue waiting using the adaptive interval above.

------------------------------------------------------------
G. TOKEN-EFFICIENCY RULE
------------------------------------------------------------

Do not spend reasoning/token budget while nothing has changed.

During a healthy long run:

WAIT > POLL LIGHTLY > ANALYZE ON COMPLETION.

Do not generate intermediate research conclusions from partial metrics unless
they indicate an immediate stop/failure condition.

For 300B / 1e / 3e, perform the expensive decision-tree analysis only after the
required horizon has completed.

------------------------------------------------------------
H. EXAMPLE FOR THIS PROJECT
------------------------------------------------------------

If corrected 300B runs at ~1 sec/batch:

expected runtime ~5 minutes

→ wait/poll approximately every 45–60 seconds,
→ tighten to ~20–30 seconds near completion.

If 1 epoch is estimated at ~35 minutes:

→ initially poll around every 5 minutes,
→ around the last 10 minutes poll every ~2 minutes,
→ near the final 2 minutes poll every ~20–30 seconds.

If 3 epochs are estimated at ~100–120 minutes:

→ poll every ~8–10 minutes initially,
→ then ~5 minutes,
→ ~2 minutes near the end,
→ ~20–30 seconds only when almost complete.

Do not blindly use these example runtimes if measured throughput differs.
Always adapt to the current measured rate.
============================================================
26. P1-v8.4-A FRESH 300B
============================================================

Only after all prior gates PASS.

Fresh OpenAI CLIP only.

Never resume from P1-v8.3 checkpoint.

Canonical:

VisA
img518
seed0
batch1
grad_accum6
FP32
TF32 OFF
AMP OFF
gradient checkpointing ON

300 batches
50 optimizer steps.

Keep DFG/SS2D and base protocol frozen.

============================================================
27. REQUIRED P1-v8.4-A 300B METRICS
============================================================

Report legacy-comparable metrics:

Base
BestSingle
OracleMulti
Uniform
SoftRouted
HardRouted

G_local
G_multi
capture

normal gain
anomaly gain

normal all-harm
anomaly all-harm

factor correlation
factor effective rank
winner shares.

Also report NEW metrics:

absolute factor correlation

residual factor correlation

residual effective rank

no-op fraction

ACT probability:

overall
normal
anomaly

ACT target fractions:

positive
negative
ambiguous

normal ACT mean

anomaly ACT mean

ACT AUROC against detached utility target if meaningful

ACT loss

ACT gradient/main

factor residual sign agreement

Residual Oracle

Residual BestSingle

Residual G_multi

fraction where:
Base wins
F1 wins
F2 wins
F3 wins
F4 wins

============================================================
28. V8.4-A DECISION TREE
============================================================

------------------------------------------------------------
CASE A1 — S4 FIXED + S2 IMPROVES
------------------------------------------------------------

Desired pattern:

anomaly all-harm falls materially from 99.9%

anomaly best gain improves toward/above zero

ACT is lower on harmful anomaly patches

residual factors become less correlated

G_multi rises

Oracle separates from BestSingle.

Action:

P1-v8.4-A PASS.

Proceed fresh 1 epoch.

------------------------------------------------------------
CASE A2 — S4 FIXED BUT S2 REMAINS
------------------------------------------------------------

Pattern:

ACT/no-op protects anomaly

anomaly harm greatly reduced

BUT:

residual factor corr still ~1

G_multi still deeply <2%

Oracle≈BestSingle.

Interpretation:

abstention solved forced-harm,
but factor generator still has no specialization.

Authorize P1-v8.4-B:

minimum factor-specific residual capacity.

Do NOT proceed to 1e under A if multi-factor hypothesis remains clearly failed.

------------------------------------------------------------
CASE A3 — RESIDUAL FACTORS GOOD BUT ACT FAILS
------------------------------------------------------------

Pattern:

Residual Oracle is clearly useful

multiple factors have positive residual utility

but ACT fails to distinguish harmful/useful patches.

First diagnose:

ACT feature quality
ACT target support
ACT class imbalance
ACT gradient magnitude.

If implementation/optimization issue:
fix exact root cause.

If ACT cannot predict utility from available features:

EXIT_FOR_DISCUSSION.

Do not make ACT network arbitrarily large.

------------------------------------------------------------
CASE A4 — ACT WORKS BUT ROUTER FAILS
------------------------------------------------------------

Pattern:

ACT correctly turns local on/off

Residual G_multi meaningful

OracleMulti significantly > BestSingle

but SoftRouted fails to approach Oracle/Hard.

Then router becomes the bottleneck.

Perform no-training router temperature/calibration analysis.

Do not increase lambda blindly.

If simple calibration explains failure:
one bounded router fix may be attempted.

Otherwise:

EXIT_FOR_DISCUSSION.

------------------------------------------------------------
CASE A5 — RESIDUALIZATION DOES NOT HELP
------------------------------------------------------------

Pattern:

anomaly residual gain remains negative

residual all-harm remains very high

Residual Oracle≈Base

and factors remain redundant.

Interpretation:

current factor generator cannot create anomaly-useful directions.

Proceed to B ONLY if forensic stage identifies a concrete capacity bottleneck.

Otherwise EXIT.

============================================================
29. P1-v8.4-B — MINIMUM FACTOR-SPECIFIC CAPACITY
============================================================

Use ONLY if A2/A5 provides direct evidence.

Do NOT create four separate CLIP encoders.

Do NOT add large independent experts.

Prefer existing repository infrastructure where appropriate:

factor_id_embedding

factor_id_to_context

factor_output_heads

or an equivalent SMALL residual specialization path.

Goal:

shared semantic base
+
small factor-specific residual transformation.

Conceptually:

shared context
        ↓
factor-specific residual head m
        ↓
distinct Δprompt_m
        ↓
text encoder
        ↓
residual factor m.

Keep the common representation shared.

============================================================
30. CAPACITY BRANCH RULE
============================================================

The added factor capacity must:

be factor-specific

be small relative to shared path

start near identity/no-op

preserve stable OpenAI CLIP semantics

operate as residual capacity

not replace the entire text encoder.

Add diagnostics:

per-head gradient norm

per-head residual norm

pairwise head output cosine

residual effective rank

functional residual correlation.

Do NOT add diversity loss initially.

Let distinct utility + assignment determine specialization first.

============================================================
31. SPECIALIZATION OBJECTIVE POLICY
============================================================

Do NOT automatically add:

orthogonality loss

load balance

equal-use loss

entropy minimization

functional diversity penalty.

Reason:

forcing factors to LOOK different is not evidence that they become USEFULLY
different.

Specialization is validated by:

OracleMulti > BestSingle

G_multi

different factor winners

functional patch residual correlation

different positive utility regions.

Representation-angle diversity alone is insufficient.

============================================================
32. V8.4-B 300B
============================================================

If B is authorized:

tests
↓
no-step gradient audit
↓
8B smoke
↓
fresh OpenAI CLIP 300B

Do not inherit v8.4-A training weights.

Compare A vs B on identical development protocol.

B passes only if factor-specific capacity produces FUNCTIONAL specialization.

============================================================
33. B DECISION TREE
============================================================

------------------------------------------------------------
B1 — SPECIALIZATION EMERGES
------------------------------------------------------------

Evidence:

Residual G_multi increases materially

OracleMulti separates from BestSingle

residual correlation decreases

different factors show distinct useful patch regions

ACT protects harmful cases.

Action:

PASS.

Proceed fresh 1e.

------------------------------------------------------------
B2 — REPRESENTATIONS DIFFER BUT FUNCTION REMAINS SAME
------------------------------------------------------------

Embedding/state cosine falls

BUT:

patch residual corr remains ~1

Oracle≈BestSingle

G_multi≈0.

Interpretation:

cosmetic diversity only.

EXIT_FOR_DISCUSSION.

Do NOT add orthogonality.

------------------------------------------------------------
B3 — FACTORS SPECIALIZE BUT ANOMALY STILL HAS NO USEFUL MODE
------------------------------------------------------------

Different factors exist

BUT anomaly Residual Oracle≈Base or harmful.

Interpretation:

conditioning signal is not anomaly-aware enough.

EXIT_FOR_DISCUSSION.

Future candidate may require stronger local/anomaly visual conditioning.

Relevant conceptual families to discuss:

conditional prompt generation

local visual-conditioned prompting

anomaly-aware prompt/adaptation

CoCoOp-like instance conditioning

AdaCLIP / anomaly-aware ZSAD approaches.

Do not auto-implement this branch.

------------------------------------------------------------
B4 — FACTORS SPECIALIZE BUT ROUTER STARVES THEM
------------------------------------------------------------

If useful factor modes exist but router fails to train/use them:

router/expert assignment becomes the real problem.

MoE / Expert-Choice-style mechanisms become relevant.

This changes routing semantics substantially.

EXIT_FOR_DISCUSSION.

Do not auto-enable load balancing.

============================================================
34. MAXIMUM ARCHITECTURE ITERATION
============================================================

This task permits at most:

P1-v8.4-A
and
P1-v8.4-B

development candidates.

Do NOT automatically create v8.4-C.

If B fails:

EXIT_FOR_DISCUSSION.

No uncontrolled architecture search.

============================================================
35. FRESH 1-EPOCH GATE
============================================================

Only after a 300B candidate passes.

Run fresh from OpenAI CLIP.

Do NOT resume the 300B checkpoint.

VisA only.

No medical.

Same frozen accepted config.

Analyze:

Base/task losses

G_local

G_multi

Residual G_multi

normal/anomaly gains

normal/anomaly all-harm

ACT normal/anomaly behavior

no-op selection

factor residual correlation

factor rank

winner shares

Oracle vs BestSingle

router capture

gradient stability.

If 1e contradicts 300B materially:

EXIT_FOR_DISCUSSION.

============================================================
36. FRESH 3-EPOCH GATE
============================================================

Only after fresh1e PASS.

Run fresh 3 epochs from OpenAI CLIP.

Do NOT continue from the 1e checkpoint.

No medical.

No validation.

No config changes between epochs.

Analyze trajectories at:

epoch1
epoch2
epoch3.

Need to ensure:

ACT does not collapse to always ON

ACT does not collapse to always OFF

anomaly protection persists

normal benefit persists

G_local remains useful

G_multi does not collapse

residual factor correlation does not return toward 1

factor specialization remains functional

main task remains stable.

============================================================
37. ACT FAILURE CASES DURING 1E/3E
============================================================

------------------------------------------------------------
ACT ALWAYS ON
------------------------------------------------------------

Likely:

negative/no-op supervision insufficient

class imbalance

gate objective too weak.

Audit ACT targets and gradient scale.

Do not simply increase lambda.

------------------------------------------------------------
ACT ALWAYS OFF
------------------------------------------------------------

Likely:

local residual candidates not sufficiently useful

or negative supervision dominates.

Check Residual Oracle.

If Oracle itself weak:

EXIT.

Do not force ACT ON.

------------------------------------------------------------
ACT NORMAL ON / ANOMALY OFF
------------------------------------------------------------

This may be correct under current training data IF anomaly factors remain harmful.

But desired long-term result is not merely:

"never use local on anomalies."

Check whether anomaly residual experts eventually become useful.

If ACT only protects Base but factors never learn anomaly modes:

S4 safety is fixed
but multi-mode anomaly modeling remains unresolved.

Do not overclaim.

============================================================
38. METRIC INTERPRETATION
============================================================

Remember why this matters for final zero-shot anomaly detection.

A useful local mechanism should improve separation:

normal:
lower anomaly score when appropriate

anomaly:
preserve/increase anomaly score when appropriate.

Current v8.3:

normal improved
but anomaly almost universally harmed.

This can damage especially:

pixel AP

pixel AUROC

and indirectly:

image AP
image AUROC

through anomaly map / peak scoring.

However:

DO NOT evaluate medical during development.

Do not claim medical improvement before final frozen evaluation.

============================================================
39. SUCCESS CRITERIA
============================================================

A candidate is not successful just because anomaly harm is hidden by no-op.

Strong candidate should ideally demonstrate BOTH:

SAFETY:
harmful local corrections can be suppressed

AND

UTILITY:
multiple residual factors provide genuinely different useful corrections.

Evidence:

anomaly all-harm strongly improved

normal gain retained

Residual Oracle useful

G_multi improved

Oracle > BestSingle

residual functional corr reduced

effective rank improved

router/ACT capture useful modes.

============================================================
40. DEVELOPMENT SUCCESS
============================================================

After fresh3e PASS:

mark:

BEST VALIDATED P1-v8.4 DEVELOPMENT CANDIDATE

not:

globally optimal.

Freeze the accepted settings in a summary/config artifact.

Do NOT run final20.

Do NOT run medical.

============================================================
41. ARTIFACTS
============================================================

Maintain one discussion file, e.g.:

P1_V84_AUTOPILOT_DISCUSSION.md

and compact JSON summaries.

Include:

source provenance

uncommitted diff status

v8.3 baseline

forensic results

oracle comparisons

collapse stage

A implementation

A 300B

B if used

1e

3e

all decisions

all stopped branches

all relevant metrics.

============================================================
42. NO COMMIT / NO PUSH POLICY
============================================================

This is absolute.

At all times:

NO git add

NO git commit

NO git push

NO PR

NO cherry-pick.

At the end show:

git status --short

git diff --stat

git diff --check

Leave code and artifacts in the worktree for user review.

============================================================
43. FINAL OUTPUT — FORENSIC EXIT
============================================================

If forensic evidence says neither residual nor no-op has credible potential:

DECISION:
EXIT_FOR_DISCUSSION

Root cause:

Evidence:

Current Oracle:

Oracle+NoOp:

ResidualOracle+NoOp:

Collapse stage:

Recommended next hypothesis:

300B:
NOT RUN

1e:
NOT RUN

3e:
NOT RUN

FINAL20:
NOT RUN

MEDICAL:
NOT RUN

COMMIT:
NONE

PUSH:
NONE

============================================================
44. FINAL OUTPUT — WAITING FOR GPU
============================================================

If source/tests are ready but GPU busy:

DECISION:
WAITING_FOR_GPU

Candidate:
P1-v8.4-A or B

CPU gates:
PASS

Next:
8B smoke or fresh300B

COMMIT:
NONE

PUSH:
NONE

============================================================
45. FINAL OUTPUT — ARCHITECTURE EXIT
============================================================

If A/B proves another semantic redesign is needed:

DECISION:
EXIT_FOR_DISCUSSION

Candidate:

Stage:

Measured failure:

Numbers:

What was ruled out:

What was fixed:

Remaining root cause:

Paper/similar-method families relevant:

Exact recommended next experiment:

Do not implement it.

COMMIT:
NONE

PUSH:
NONE

============================================================
46. FINAL OUTPUT — DEVELOPMENT PASS
============================================================

If 3e passes:

DECISION:
DEVELOPMENT_GATES_PASS

Candidate:

Source state:
UNCOMMITTED WORKTREE

Mechanism:

true residual factors:
YES

ACT/no-act:
YES

factor-specific capacity:
YES/NO

rho:
.05

factor loss:

ACT loss:

router loss:

lambda values:

300B summary:

1e summary:

3e summary:

v8.3 comparison:

G_local:

G_multi:

anomaly best gain:

anomaly all-harm:

residual factor corr:

factor effective rank:

ACT normal/anomaly behavior:

FINAL20:
NOT RUN

MEDICAL:
NOT RUN

NO MEDICAL VALIDATION

COMMIT:
NONE

PUSH:
NONE

============================================================
47. EXECUTION ORDER
============================================================

Follow exactly:

1.
verify worktree/provenance

2.
read current source

3.
NO-TRAIN forensic audit

4.
calculate:
Current Oracle
Oracle+NoOp
ResidualOracle+NoOp

5.
normal/anomaly decomposition

6.
stage-by-stage collapse trace

7.
classify forensic case

8.
if evidence supports:
implement P1-v8.4-A
true residual + ACT/no-act

9.
CPU tests

10.
no-step gradient calibration

11.
GPU availability

12.
8B smoke

13.
fresh A 300B

14.
analyze full decision tree

15.
if A solves root causes:
fresh1e → fresh3e

16.
if A fixes safety but specialization still failed:
implement only evidence-backed P1-v8.4-B small factor-specific residual capacity

17.
tests → smoke → fresh B 300B

18.
if B passes:
fresh1e → fresh3e

19.
if B fails:
EXIT

20.
never final20

21.
never medical

22.
never commit

23.
never push.

============================================================
48. RESEARCH PRINCIPLE
============================================================

The priority order is:

correct correction semantics
>
ability to abstain safely
>
factor utility
>
factor functional specialization
>
router capture
>
additional capacity.

Do NOT try to fix the router before useful factor modes exist.

Do NOT force factor diversity before proving that diversity is useful.

Do NOT interpret balanced router usage as specialization.

Do NOT interpret different embeddings as functional specialization.

Do NOT use medical results to choose architecture.

Every automatic modification must answer:

What exact measured failure does it solve?

Why is this a root cause?

What evidence supports the mechanism?

How will the next experiment falsify it?

If those questions cannot be answered:

EXIT_FOR_DISCUSSION.

Start now.