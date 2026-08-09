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

### Evidence labels

- **PROVEN:** sign/target path, F0/R0 imbalance behavior, Stage-B distributions,
  parameter immutability, and persistent normalized main-factor conflict.
- **IMPLEMENTED:** effective-number factor loss, support-normalized router,
  static `.03/.10`, checkpoint metadata, and primary-anchored factor surgery.
- **TRAINING-VALIDATED (SMOKE):** runtime accumulation, optimizer execution,
  aligned-window no-projection behavior, MAIN preservation, factor/router
  gradient liveness, and rho invariants.
- **NOT YET VALIDATED BY DEVELOPMENT HORIZON:** whether the source improves a
  fresh corrected 300B trajectory. Corrected 300B, one epoch, three epochs,
  final20, and six-medical evaluation are all **NOT RUN**.
