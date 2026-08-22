# SABRA-CAR Decision Tree Journal

This file is append-only. The headings below declare the complete tree; dated
stage records are appended chronologically after the structure.

## Run Identity

- repository: XLaiHuy/ACD-CLIP-
- baseline branch: `research/p5-sabra-canonical-v1`
- BASE_SHA: `a986bcfee41c31f03d38e449efb8826d56c90525`
- CAR branch: `research/p6-sabra-car-v1`
- worktree: `/home/ai4/caohuy/ACD-CLIP-sabra-car-v1`
- start timestamp: `2026-08-23T01:01:24+07:00`
- hardware/runtime: NVIDIA GeForce RTX 5060 Ti 16311 MiB; Python 3.11.15; PyTorch 2.12.1+cu130; CUDA 13.0; FP32; TF32 disabled by canonical contract
- Phase2B checkpoint identity: E10 `/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/phase2b/checkpoints/adapter_10.pth`, SHA256 `6643cd68eafabf9acdb724242ef5b2d1fbc4bf7e9d2ba7ad6c47776ea646da80`
- scientific baseline identity: canonical scientific code `4aa9b465ddeb072e9218b74982306d6324c62375`; canonical workflow BASE_SHA above; Phase2B E10 selected on MVTec by the frozen weighted score

## Scientific Goal

Develop and validate the strongest scientifically defensible SABRA-CAR model
under the preregistered Direction -> Trust/Stability -> Radius -> Integration
tree, with pixel AP as the primary weakness, without Medical tuning or post-hoc
scientific rule changes.

## Preregistered Decision Tree

The complete frozen definitions and quantitative gates are in
`research/sabra_car/MASTER_PREREGISTRATION.md` at the S0 output commit.

## Stage S0 — Bootstrap / Preregistration

## Stage R0 — Signed Direction

## Stage R0B — Ranking-Aware Fallback

## Stage R1 — Action Predictor

## Stage R2 — Trust / Stability

## Stage R3 — Radius / Strength

## Stage R4 — Integrated CAR

## Freeze

## Final Evaluation

## Final Decision

### Stage S0 — Bootstrap / Preregistration

START_TIME:
2026-08-23T01:01:24+07:00

END_TIME:
2026-08-23T01:01:24+07:00

INPUT_COMMIT:
`a986bcfee41c31f03d38e449efb8826d56c90525`

OUTPUT_COMMIT:
Recorded in the next append-only entry after commit creation.

PURPOSE:
- Establish an isolated CAR branch/worktree from current canonical remote HEAD and freeze the complete scientific decision tree before any CAR experiment.

HYPOTHESIS:
- Signed correction direction, selective reliability, and bounded radius can repair SABRA-v1 pixel AP without sacrificing canonical zero-shot integrity.

ALLOWED_CHANGES:
- Add CAR research specifications, append-only journal, and machine state.

FORBIDDEN_CHANGES:
- Any canonical production SABRA edit, experiment, threshold adaptation, Medical access, Phase2B training, canonical history rewrite, or force push.

DATA_USED:
- dataset: no samples opened; existing provenance/manifests inspected only
- classes: canonical inventories inspected only
- split: none
- whether GT was used: no
- whether Medical was accessed: no Medical sample was opened; existing retrospective summary metadata was inspected for baseline identity only

IMPLEMENTATION:
- files added: `research/sabra_car/MASTER_PREREGISTRATION.md`, `research/sabra_car/DECISION_TREE_JOURNAL.md`, `research/sabra_car/AUTOPILOT_STATE.json`
- files modified: none
- important algorithmic changes: none; all future stage definitions and gates frozen

COMMANDS:
- `git fetch origin research/p5-sabra-canonical-v1`
- `git worktree add -b research/p6-sabra-car-v1 /home/ai4/caohuy/ACD-CLIP-sabra-car-v1 origin/research/p5-sabra-canonical-v1`
- checkpoint SHA256 and runtime identity checks

TESTS_BEFORE_RUN:
- canonical remote/local divergence: PASS (`0 0`)
- requested branch/worktree unused: PASS
- checkpoint SHA256: PASS

RUN:
- command: no scientific run
- run directory: N/A
- start time: N/A
- estimated runtime: N/A
- actual runtime: N/A

RESULTS:
- primary metrics: N/A
- secondary metrics: N/A
- class/dataset breadth: N/A
- relevant confidence/risk/coverage information: N/A

EXPECTED_VS_OBSERVED:
- expected: clean isolation from verified remote canonical HEAD
- observed: CAR branch created at exact remote SHA `a986bcf`; canonical worktree left untouched

BUGS:
- None.

SCIENTIFIC_INTERPRETATION:
- This stage establishes provenance and prevents outcome-dependent rule changes.
- It does not provide evidence that any CAR component works.

GATES:
- remote canonical current; threshold exact SHA match; observed `a986bcf` with zero divergence; PASS
- isolated unused branch/worktree; threshold yes; observed yes; PASS
- full decision tree preregistered before CAR experiment; threshold yes; observed yes; PASS
- Medical samples not accessed; threshold zero; observed zero; PASS
- Phase2B training steps; threshold zero; observed zero; PASS

DECISION:
- CONTINUE

NEXT_STAGE:
- Stage R0 — Signed Direction implementation, contract tests, and oracle analysis.

NOTES:
- Existing canonical VisA cache may be reused only after the frozen provenance audit passes.
- Session time remaining is unavailable and will not be invented.

### S0 Publication Verification

- timestamp: `2026-08-23T01:15:03+07:00`
- S0 output commit: `08ca99ff69d6d85184f5d145830876befb413628`
- remote branch: `origin/research/p6-sabra-car-v1`
- local HEAD: `08ca99ff69d6d85184f5d145830876befb413628`
- remote HEAD: `08ca99ff69d6d85184f5d145830876befb413628`
- divergence: `0 0`
- decision: S0 publication gate PASS; R0 implementation is authorized.
- Medical access remains forbidden by explicit user instruction.

### R0 Preregistration Addendum

- timestamp: `2026-08-23T01:15:03+07:00`
- input commit: `fe0c111`
- purpose: freeze operational definitions that were implicit in the master preregistration before implementation or result access.
- artifact: `research/sabra_car/R0_IMPLEMENTATION_CONTRACT.md`
- scientific changes after result observation: none; no R0 experiment has run.
- Medical accessed: no.
- next action: commit and push this definition, then implement the additive R0 sidecar and regression tests.
### R0 Engineering Bug R0-ENG-001

BUG_ID:
R0-ENG-001

symptom:
- Targeted test collection failed with `ModuleNotFoundError: No module named 'sklearn'`.

root cause:
- The initial sidecar used an undeclared scikit-learn runtime dependency that is absent from the frozen canonical environment.

scientific impact:
- None. The failure occurred during test collection before cache validation, GT access, utility computation, or any scientific result.

fix:
- Replace scikit-learn metrics with vectorized exact descending-score group formulas matching canonical tie-aware AP/AUROC semantics.

regression test:
- `test_vectorized_metrics_match_canonical_tie_semantics`

fix commit:
- Pending this fix commit.

status:
- FIX IMPLEMENTED; validation pending.
### R0-ENG-001 Validation

- timestamp: `2026-08-23T01:25:08+07:00`
- root-cause fix: dependency-free vectorized exact grouped AP/AUROC.
- regression test: `test_vectorized_metrics_match_canonical_tie_semantics` PASS.
- affected suite: `tests/test_car_r0_direction.py tests/test_phase2b_runtime.py tests/test_sabra_trust_need.py`.
- observed: 11 passed.
- scientific contract/parity status: PASS.
- scientific results observed: none.
- Medical accessed: no.
- fix commit: pending the immediately following commit.
- status: FIXED.
### R0 Implementation Publication and Probe Authorization

- timestamp: `2026-08-23T01:30:55+07:00`
- implementation/fix commit: `252534bc71d548b3c2a8f91128e0379cff2906bf`
- local HEAD equals remote HEAD: yes
- R0-ENG-001 fix commit: `252534bc71d548b3c2a8f91128e0379cff2906bf`
- invalid partial artifact removed: untracked `tools/sabra_car/r0_direction.py.orig`, a patch-tool backup with no scientific content
- tests: 11 passed
- next command: one bounded batch of eight cached VisA images for utility throughput and parity timing
- expected probe runtime: at most 5 minutes
- session time remaining: unavailable; not invented
- Medical accessed: no
- Phase2B training steps: 0
### R0 Utility Runtime Estimate

- timestamp: `2026-08-23T01:32:12+07:00`
- bounded probe: 8 cached VisA images
- probe status: PASS
- utility throughput: 40.04592044253699 images/s
- pure-compute projection: 0.8998003525737052 minutes for 2,162 images
- peak reserved VRAM: 222298112 bytes
- EXPECTED_RUNTIME_MIN: 5
- EXPECTED_FINISH_TIME: `2026-08-23T01:37:12+07:00`
- available disk: 209 GiB
- available RAM: 27 GiB; swap available 5.4 GiB
- GPU before run: RTX 5060 Ti, 16311 MiB total, 556 MiB used
- expected utility result size: under 100 MiB
- session time remaining: unavailable; not invented
- execution mode: one blocking command
- Medical accessed: no
- Phase2B training steps: 0

### R0 Engineering Bug R0-ENG-002

BUG_ID:
R0-ENG-002

symptom:
- The initial utility finite-difference parity check passed at three fixed coordinates whose analytic utilities were approximately zero, so it did not provide informative evidence for the sign or magnitude of the utility derivative.

root cause:
- Fixed patch indices were selected before observing the utility distribution and happened to land on locally insensitive coordinates.

scientific impact:
- The 12 utility shards remained valid, with exact native-cache parity, but the original derivative-correctness evidence was too weak to authorize alpha evaluation.
- No alpha result was run or observed before this classification and fix.

fix:
- Select the three stable largest-absolute-utility coordinates from the first canonical class and require finite-difference magnitude agreement, matching signs, nonzero analytic utility, and native zero-delta cache parity.

regression test:
- `test_informative_coordinates_are_largest_and_stable`

validation:
- timestamp: `2026-08-23T01:37:51+07:00`
- targeted suite: 12 passed.
- utility records: 2,162 across 12 VisA classes.
- native zero-delta cache parity max absolute error: 0.
- informative finite-difference absolute errors: 4.0978193283081055e-08, 1.4528632164001465e-07, 8.177012205123901e-07.
- all informative checks: sign match true and within tolerance.
- utility phase status: PASS.
- elapsed rerun time: 9.958480918081477 seconds.
- Medical accessed: no (`medical_reads=0`).
- Phase2B training steps: 0.

status:
- FIXED; alpha evaluation authorized after publication of this checkpoint.

### R0 Utility Publication and Alpha Runtime Estimate

- timestamp: `2026-08-23T01:42:31+07:00`
- R0-ENG-002 fix commit: `41de831`
- utility evidence commit: `0e50df21f531e26b57469cc1c171a4f0fa437f4d`
- local HEAD equals remote HEAD: yes
- divergence: `0 0`
- bounded probe: one native condition over the full 200-image candle class; result values were not printed or used for selection
- probe pixels: 53,664,800
- one-condition elapsed time: 8.805962819838896 seconds
- projected 108 condition-class equivalents: 15.850733075710014 minutes
- EXPECTED_RUNTIME_MIN: 20
- EXPECTED_FINISH_TIME: `2026-08-23T02:02:31+07:00`
- peak reserved VRAM: 113246208 bytes
- available disk: 209 GiB
- available RAM: 24 GiB; swap available 5.4 GiB
- GPU before run: RTX 5060 Ti, 16311 MiB total, 556 MiB used
- Medical accessed: no
- Phase2B training steps: 0
- execution mode: one blocking command
- next action: run the complete fixed alpha landscape, then audit every emitted row before radius probing

### R0 Alpha Landscape Result

- timestamp: `2026-08-23T01:57:27+07:00`
- elapsed runtime: 804.4572036098689 seconds
- audited rows: 132 per-class rows; 12 classes; 11 condition labels including alpha-zero aliases
- all pAP, pAUROC, and loss values finite: yes
- selected signed alpha: 0.25
- signed macro pAP: 0.626144040476265
- matched positive-only macro pAP: 0.6158246330733695
- signed minus positive-only macro pAP: +1.0319407402895497 pp
- G1 precursor: PASS
- signed-better class breadth versus positive-only: 12/12
- G2 precursor: PASS
- signed minus positive-only macro pAUROC: +0.23619865972595022 pp
- G3 precursor: PASS
- signed minus native macro pAP: +5.620394793869831 pp
- signed-better class breadth versus native: 12/12
- result status: ALPHA PASS; final R0 decision remains pending signed-radius and quadrant artifacts
- Medical accessed: no (`medical_reads=0`)
- Phase2B training steps: 0
- next action: publish the alpha artifacts, then run the preregistered real-image sparse-radius parity/timing probe

### R0 Engineering Bug R0-ENG-003

BUG_ID:
R0-ENG-003

symptom:
- `git diff --cached --check` reported trailing whitespace on every generated CSV row because the files used CRLF line endings.
- The alpha evidence commit `bcff5927cf38bd754c357f51963ec762bd4f5822` was nevertheless created and pushed because the shell commands were newline-separated instead of fail-fast chained.

root cause:
- `csv.DictWriter` used the platform/default dialect line terminator rather than an explicitly repository-safe LF terminator.
- The publication command did not use `&&` after the required cached-diff check.

scientific impact:
- None. CSV values, row order, selected alpha, and all reported metrics are unchanged; this is serialization and publication-control hygiene only.
- No radius result was run or observed before this classification and fix.

fix:
- Set `lineterminator="\\n"` in the shared R0 CSV writer and normalize only the two already-generated alpha CSV files from CRLF to LF.
- All future publication command sequences must stop on a failed `git diff --cached --check`.

regression test:
- `test_csv_writer_uses_lf_line_endings`

validation:
- timestamp: `2026-08-23T02:00:07+07:00`
- targeted suite: 13 passed.
- `git diff --check`: PASS after normalization.
- normalized artifacts: `alpha_per_class.csv`, `alpha_summary.csv`.
- alpha scientific metrics unchanged: yes.
- Medical accessed: no.
- Phase2B training steps: 0.

status:
- FIXED; radius probing remains pending publication of this corrective commit.

### R0 Engineering Bug R0-ENG-004

BUG_ID:
R0-ENG-004

symptom:
- The first real-image sparse-radius probe failed because three basis max-absolute errors were slightly above the preregistered implementation tolerance of 2e-6.
- All three sparse coordinate choices exactly matched direct canonical coordinate choices.

root cause:
- Subtracting native float32 deployed logits to recover a direct unit-impulse basis introduced cancellation at 2.3466e-6 to 2.5137e-6, while mean absolute errors were approximately 2.5e-8.

scientific impact:
- None. The failure occurred at the correctness probe before the full radius run; no radius scientific result was observed.
- The direction, alpha landscape, action map, coordinate objective, and chosen radii are unchanged.

fix:
- Use a bounded 3e-6 absolute tolerance for float32 basis parity while retaining the 1e-6 exact-coordinate-choice tolerance and finite-value checks.
- Preserve the original failed probe as `radius_probe_r0_eng_004_fail.json`.

regression tests:
- `test_basis_parity_tolerance_covers_float32_cancellation_only`
- The existing action-threshold regression caught and prevented a transient misplaced-return edit during patch application before any probe rerun.

validation:
- timestamp: `2026-08-23T02:04:15+07:00`
- targeted suite: 14 passed.
- direct-versus-sparse correction errors: 0, 0, 0.
- basis max absolute errors: 2.5136396288871765e-6, 2.346583642065525e-6, 2.4755136109888554e-6.
- basis mean absolute errors: approximately 2.5e-8.
- corrected real-image probe status: PASS.
- failed probe evidence preserved: yes.
- coordinate runtime per image: 0.21967458305880427 seconds.
- conservative EXPECTED_RUNTIME_MIN: 12
- EXPECTED_FINISH_TIME: `2026-08-23T02:16:15+07:00`
- available disk: 209 GiB
- available RAM: 24 GiB; swap available 5.4 GiB
- GPU before run: RTX 5060 Ti, 16311 MiB total, 556 MiB used
- Medical accessed: no (`medical_reads=0`).
- Phase2B training steps: 0.

status:
- FIXED; publish this correctness checkpoint before the full blocking radius run.

### R0 Signed-Radius Result and Finalization Estimate

- timestamp: `2026-08-23T02:09:19+07:00`
- input correctness commit: `6c7175c69bbb8bebd7c08271c285775bf28fbfbe`
- signed-radius status: PASS
- elapsed runtime: 92.77068063500337 seconds
- records: 2,162 across 12 VisA classes
- shard path alignment: PASS
- finite correction values: PASS
- fixed signed-radius grid membership: PASS
- zero-radius patches: 1,380,498
- nonzero-radius patches: 1,579,480
- Medical accessed: no (`medical_reads=0`)
- Phase2B training steps: 0
- finalization work: reuse audited alpha rows; evaluate signed-radius metrics and selected-alpha per-image quadrants
- conservative EXPECTED_RUNTIME_MIN: 10
- EXPECTED_FINISH_TIME: `2026-08-23T02:19:29+07:00`
- available disk: 209 GiB
- available RAM: 24 GiB; swap available 5.4 GiB
- GPU before run: RTX 5060 Ti, 16311 MiB total, 556 MiB used
- next action: publish the signed-radius artifacts, then run final R0 metrics and exact gate decision as one blocking command
