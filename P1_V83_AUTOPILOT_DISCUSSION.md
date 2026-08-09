# P1-v8.3 root-cause autopilot discussion

## Frozen provenance

- Authoritative upstream branch: `origin/phase4-progress1-cops-dynamic-prompt`
- Audited upstream commit: `d58b84bcecb9c4d22bdef321ea0ca28bd3b6745b`
- Isolated worktree: `/home/ai4/caohuy/ACD-CLIP-p1v83-autopilot`
- Historical 300-batch checkpoint SHA-256: `3ad425decfbb0a36bdad4c4157dbc2f84ef555e2ba6e0d575e0ecbacbaa62cf7`
- Original dirty worktree is preserved and is not used for source edits.
- No push, history rewrite, final-20 training, or medical evaluation is allowed.

## Source and architecture reconciliation

The current runtime, tests, canonical smoke launcher, final-20 launcher, and exact
medical launcher consistently require `phase2b_hybrid` global text for P1-v8.3.
Older `hard_anchor` launchers belong to P1-v8.2/triage paths and are not the
P1-v8.3 canonical configuration.  P1-v8.3 initializes from OpenAI CLIP only;
it does not load a Phase2B training checkpoint.

The current utility sign path is internally consistent:

- patch target 1 means anomaly foreground;
- base margin is abnormal minus normal;
- factor evidence is `10 * (similarity_abnormal - similarity_normal)`;
- candidate margin is detached base plus fixed `rho=.05` factor evidence;
- positive relative gain means the factor lowers anomaly-target BCE.

The historical auxiliary gradient ratios were lambda-weighted.  Current
diagnostics expose raw and weighted fields separately.

## Root-cause hypothesis under test

Two loss reductions can amplify sparse support before lambda calibration:

1. F0 averages normal and anomaly region means 50/50.  A tiny anomaly region
   can therefore give each anomaly patch far more influence than each normal
   patch.
2. R0 averages CE only over informative patches.  One or two informative
   patches therefore produce a full-size router loss.

Stage B evaluates F0 against inverse-effective-number patch weighting at
`beta={.99,.999,.9999}`, and R0 against CE masked by informativeness but divided
by all valid support.  All comparisons use the same natural seed-0 forward
states in six-microbatch optimizer windows.

## Current stage

`SMOKE_PASS_READY_FOR_CORRECTED_300B`: Stage B, the primary-preserving source
change, CPU gates, and the fresh 8-batch runtime smoke are complete.  The
corrected 300B has not yet started.

## Stage B optimizer-window evidence

The audit consumed 66 natural seed-0 windows (396 consecutive microbatches):

- normal-only: 3 windows;
- anomaly-containing: 63 windows;
- router-informative: 65 windows;
- optimizer constructed/steps: 0/0;
- parameter hash before/after: identical;
- all audited `.grad` fields remained empty.

On shared-semantic parameters, raw F0 factor/main had median `9.644`, p90
`38.651`, p95 `52.261`, and max `76.742`.  F1 beta `.999` reduced these to
median `3.422`, p90 `14.075`, p95 `19.332`, and max `26.940`, while retaining
median `21.85%` of total effective patch weight for anomaly support.  Beta
`.99` reduced scale more strongly but retained only about `3%` anomaly weight;
beta `.9999` remained too close to F0.  Therefore beta `.999` is selected.

R0 router/main had median `3.215`, p90 `7.655`, and p95 `11.635`.  Dividing the
masked CE by all valid support (R1) reduced these to median `.139`, p90 `.508`,
and p95 `.611`; gradients remained nonzero on all 65 informative windows.

The analytical static grid selected factor/router lambdas `.03/.10`.  Its
weighted true-combined/main distribution is median `.103`, p90 `.422`, p95
`.582`, max `.814`, with zero windows above main.  Factor `.05` produced four
repeated windows above main, so `.03` is the largest stable factor value in the
preregistered grid.  Router `.10` is the largest preregistered value and is not
inflated beyond the grid after support normalization.  Magnitude is
`STATIC_OK`; GradNorm is not eligible.

After normalization, main-factor cosine remains negative in `63.64%` of
windows with median `-.426`.  This is persistent optimizer-window conflict.
Router cosine is numerically near zero.  The minimum supported solution is
therefore:

1. inverse-effective-number factor patch weighting, beta `.999`;
2. support-normalized router loss;
3. static lambdas `.03/.10`;
4. primary-anchored auxiliary surgery for accumulated main/factor gradients
   only on shared-semantic parameters; MAIN and router remain unprojected.

Symmetric PCGrad was rejected for the development candidate because it also
projects MAIN.  The selected implementation removes only the factor component
parallel to and opposing accumulated MAIN, leaving MAIN exactly unchanged.
This is a custom main-preserving auxiliary projection informed by the
primary/auxiliary gradient-surgery literature; it is not represented as
paper-faithful PCGrad.

## Source/test gate

- focused suite: `41 passed` (prior baseline: `35 passed`);
- changed modules compile;
- changed shell launchers pass `bash -n`;
- `git diff --check` passes;
- fresh 8-batch smoke: **PASS**, 2 optimizer steps (6 + remainder 2), all
  metrics finite, peak allocated GPU memory about 4.12 GiB;
- smoke surgery artifact: MAIN exact-change norm `0` in both windows,
  correction reconstruction error `0`, router never projected, rho fixed `.05`;
- both smoke optimizer windows were aligned, so the real no-projection path is
  runtime-validated; the conflicting projection path remains deterministic
  unit-test evidence until a natural conflicting training window occurs;
- corrected-300B launcher is prepared but not executed;
- final-20 and medical launchers remain prepared-only and were not executed.

## Exact local-commit scope

Intended reviewable files are:

- `P1_V83_AUTOPILOT_DISCUSSION.md`;
- `train.py`;
- `model/checkpoint_utils.py`;
- `model/h6/utility_routing.py`;
- `scripts/run_p1_v83_smoke.sh`;
- `scripts/run_p1_v83_final20.sh`;
- `scripts/run_p1_v83_corrected_300b.sh`;
- `tests/test_p1_v83_runtime.py`;
- `tests/test_p1_v83_loss_counterfactuals.py`;
- `tools/audit_p1_v83_optimizer_windows.py`;
- `tools/analyze_p1_v83_optimizer_windows.py`;
- `tools/preflight_p1_v83_final_checkpoint.py`;
- compact Stage-B summaries under
  `runs/p1_v83_dev/root_cause_optimizer_audit/` (`audit_summary.json`,
  `static_decision.json`, `window_distributions.json`, `lambda_grid.csv`).

Excluded are the raw `optimizer_windows.json`, temporary `progress.json`, the
unstarted `root_fix_smoke8/` GPU-preflight folder, all checkpoints/weights,
data/symlinks, caches, and unrelated historical run folders.

## Corrected-300B attempt 1 decision

**DECISION: EXIT_FOR_DISCUSSION**

Fresh run:

- directory: `runs/p1_v83_dev/corrected_300b_primary_anchored_attempt1`;
- source: `4b57b2484927b08fe2cd08b2c3d8a04cbbe91ffa`;
- OpenAI CLIP object SHA-256:
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`;
- exactly 300 batches / 50 optimizer steps;
- all milestones `32,64,128,192,256,300` completed;
- runtime 294.05 seconds; peak allocated/reserved GPU memory about
  4.14/4.65 GiB;
- model preflight and final runtime summary: **PASS**;
- no NaN/Inf, rho remained fixed `.05`, and the effective-number/support-aware
  configuration and primary-anchored provenance were exact.

Optimizer-window behavior is healthy enough to reject scale/conflict as the
remaining root cause:

- 50 windows, 44 anomaly-containing, 0 router-informative;
- weighted factor/main median `.167`, p75 `.365`, p90 `.583`, p95 `.713`, max
  `1.323` (one window above main, not systematic domination);
- raw main/factor conflicts in 19/50 windows;
- safe factor/main median `.105`, p95 `.713`, max `1.323`;
- MAIN exact-change max `0`;
- correction reconstruction error max `0`;
- conflicting-window `abs(dot(main,safe_factor))` max about `3.81e-15`.

The semantic/specialization gate fails in two independent, architecture-level
ways:

1. **S2 — local useful but Oracle approximately BestSingle.** Cumulative
   `G_local` is positive and rises from `2.52%` at batch 32 to `3.43%` at batch
   300, but `G_multi` only rises from `.175%` to `.303%`, deeply below the 2%
   meaningful multi-mode region. At batch 300 BestSingle/OracleMulti losses are
   `.075662/.075426`, factor patch-function correlation is `.998699`, factor
   effective rank is `1.0055`, and winner shares are
   `[.351,.148,.030,.470]`. Current candidates therefore do not demonstrate
   multiple useful modes.
2. **S4 — anomaly all-harm persists.** Anomaly all-harm is `96.61%` at batch
   32 and worsens monotonically to `99.90%` at batch 300. Cumulative anomaly
   best gain remains negative (`-3.19%`) while normal best gain is positive
   (`+2.94%`). With sign/target semantics, exploration, normalization,
   magnitude, and gradient direction already validated, this means the local
   candidate set lacks a beneficial anomaly correction rather than suffering
   from an optimizer-scale bug.

The router auxiliary is inactive because the canonical entropy gate accepts no
patches. A no-training sensitivity audit shows that changing the temperature
and entropy threshold could create support (up to about `16.1%` for
`tau=.02`, threshold `.995`), but this does not resolve either the weak
multi-mode evidence or anomaly all-harm. Increasing `lambda_router` cannot
create a gradient when support is zero, and changing teacher semantics now
would be a separate semantic intervention. No lambda change is recommended.

Scientifically defensible discussion options include an explicit identity/no-op
candidate or local abstention gate for harmful anomaly corrections, and stronger
factor-specific conditional capacity for genuine multi-mode candidates. Both
change architecture semantics and therefore require user discussion before any
implementation. Load balancing, forced orthogonality, a larger router lambda,
or a brute-force threshold sweep are not justified by this evidence.

Per the decision tree, fresh 1e and 3e are **NOT RUN** after this S2/S4 failure.
Final20 and all medical evaluation remain **NOT RUN**. No run directory was
deleted on EXIT, and nothing was pushed.

### Evidence labels

- **PROVEN:** sign/target path, F0/R0 imbalance behavior, Stage-B distributions,
  parameter immutability, and persistent normalized main-factor conflict.
- **IMPLEMENTED:** effective-number factor loss, support-normalized router,
  static `.03/.10`, checkpoint metadata, and primary-anchored factor surgery.
- **TRAINING-VALIDATED (SMOKE + 300B):** runtime accumulation, optimizer execution,
  aligned-window no-projection behavior, MAIN preservation, factor/router
  gradient handling, rho invariants, and stable primary-anchored conflict
  removal.
- **PROVEN DEVELOPMENT FAILURE:** S2 weak multi-mode evidence and S4 persistent
  anomaly all-harm require architecture-semantic discussion.
- **NOT RUN BY DECISION:** one epoch, three epochs, final20, and six-medical
  evaluation.
