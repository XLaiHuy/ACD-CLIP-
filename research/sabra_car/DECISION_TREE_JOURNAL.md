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

### R0 Engineering Bug R0-ENG-005

BUG_ID:
R0-ENG-005

symptom:
- Finalization raised `NameError: name 'false' is not defined` while constructing the final summary after metric and quadrant artifacts had been written.

root cause:
- A JSON-style lowercase boolean literal was used in Python code, and the summary-construction path lacked a direct unit test.

scientific impact:
- No final R0 decision artifact was written, so this failure is not a scientific gate result.
- Partial metric/quadrant files were not inspected or used for branching after the failure.
- No downstream stage was started.

fix:
- Move summary construction into `final_summary_payload` using the Python boolean `False` and explicit zero-training contract.

regression test:
- `test_final_summary_uses_python_boolean_and_zero_training`

validation:
- timestamp: `2026-08-23T02:14:18+07:00`
- targeted suite: 15 passed.
- `git diff --check`: PASS.
- failed-run artifacts preserved with `r0_eng_005_fail` suffixes before rerun.
- Medical accessed: no.
- Phase2B training steps: 0.

status:
- FIXED; publish this engineering checkpoint before rerunning finalization.

### Stage R0 — Signed Direction

START_TIME:
2026-08-23T01:15:03+07:00

END_TIME:
2026-08-23T02:17:45+07:00

INPUT_COMMIT:
`1af541e`

OUTPUT_COMMIT:
Recorded in the next append-only publication entry after commit creation.

PURPOSE:
- Test whether exact canonical-loss gradient sign supplies pixel-AP value beyond a matched positive-only oracle, and measure coordinate-radius headroom.

HYPOTHESIS:
- Signed BOOST/SUPPRESS direction will improve macro pAP over the same-alpha positive-only intervention by at least +1.00 pp without unsafe pAUROC loss.

ALLOWED_CHANGES:
- Additive cache-only R0 tooling, deterministic tests, append-only records, and result artifacts under `results/sabra_car/r0`.

FORBIDDEN_CHANGES:
- Phase2B training or parameter updates, canonical model edits, altered alpha grid/gates, Medical access, outcome-dependent fallback, force push, or history rewrite.

DATA_USED:
- dataset: VisA source only
- records/classes: 2,162 images across the exact 12-class inventory
- split: canonical fixed source metadata
- whether GT was used: yes, for preregistered source-oracle utility, radius, and evaluation
- whether Medical was accessed: no; zero reads

IMPLEMENTATION:
- files added: `tools/sabra_car/r0_direction.py`, `tools/sabra_car/__init__.py`, `tests/test_car_r0_direction.py`, R0 result artifacts
- files modified: append-only journal and autopilot state
- important algorithmic changes: exact signed utility, matched alpha landscape, exact grouped AP/AUROC, sparse coordinate-radius oracle, per-image loss/AP quadrants
- canonical production inference and Phase2B parameters: unchanged

COMMANDS:
- `python -m tools.sabra_car.r0_direction --phase utilities --batch-size 4`
- `python -m tools.sabra_car.r0_direction --phase alpha --batch-size 4`
- `python -m tools.sabra_car.r0_direction --phase probe-radius --batch-size 4 --radius-patch-batch 64`
- `python -m tools.sabra_car.r0_direction --phase radius --batch-size 4 --radius-patch-batch 64`
- `python -m tools.sabra_car.r0_direction --phase finalize --batch-size 4`

TESTS_BEFORE_RUN:
- targeted R0/canonical suite: PASS, 15 tests after all engineering fixes
- checkpoint/cache/config/CLIP/metadata/core hashes: PASS
- native zero-delta cache parity: max absolute error 0
- informative utility finite-difference parity: PASS on three top-absolute-utility coordinates
- sparse basis/direct coordinate parity: PASS on three real-image coordinates
- finite values and exact class/path/grid invariants: PASS

RUN:
- run directory: `results/sabra_car/r0`
- utility elapsed: 16.2566 seconds initial generation; 9.9585 seconds validated cached rerun
- alpha elapsed: 804.4572 seconds
- full signed-radius elapsed: 92.7707 seconds
- final metrics elapsed: 95.2649 seconds
- execution mode: attached blocking commands; no detached jobs

RESULTS:
- selected signed alpha: 0.25
- native macro pAP / pAUROC: 0.5699400925375667 / 0.9707623684180294
- matched positive-only macro pAP / pAUROC: 0.6158246330733695 / 0.9928031760389301
- signed macro pAP / pAUROC: 0.626144040476265 / 0.9951651626361896
- signed-radius macro pAP / pAUROC: 0.5767954095171167 / 0.9954967099829061
- signed minus positive-only: +1.0319407402895497 pp pAP, +0.23619865972595022 pp pAUROC
- signed minus native: +5.620394793869831 pp pAP, +2.440279421816016 pp pAUROC
- breadth: 12/12 classes over matched positive-only and 12/12 over native
- action rates BOOST / KEEP / SUPPRESS: 0.040801033050451754 / 0.4661271892689249 / 0.4930717776806233
- sign reversal among non-KEEP: 0.9235753680833088
- loss/AP disagreement fraction: 0.05704697986577181
- quadrants loss-down/AP-up, loss-down/AP-down, loss-up/AP-up, loss-up/AP-down: 1119, 40, 28, 5
- ties: loss 4; AP 8

EXPECTED_VS_OBSERVED:
- expected: signed direction supplies at least +1.00 pp pAP over matched positive-only with breadth and pAUROC safety
- observed: +1.03194 pp, 12/12 breadth, and +0.23620 pp pAUROC; hypothesis met narrowly but exactly
- coordinate-radius oracle did not beat the selected fixed signed intervention and is retained only as negative/headroom evidence

BUGS:
- R0-ENG-001: unavailable scikit-learn dependency; replaced with exact dependency-free metrics
- R0-ENG-002: fixed parity coordinates were uninformative; replaced with stable top-absolute-utility checks
- R0-ENG-003: CRLF result serialization and non-fail-fast publication sequence; fixed and normalized without metric changes
- R0-ENG-004: float32 basis cancellation exceeded 2e-6; bounded 3e-6 parity tolerance validated
- R0-ENG-005: invalid lowercase Python boolean in final summary; factored into a directly tested helper
- all failed-run evidence was preserved before reruns

SCIENTIFIC_INTERPRETATION:
- Signed suppression provides real incremental ranking value beyond merely boosting patches selected by the same loss gradient.
- The gain over positive-only is only 0.03194 pp above the preregistered threshold, so downstream GT-free reliability must be judged strictly.
- The coordinate-radius oracle underperforms fixed signed alpha on pAP, providing no evidence to expand the scientific family at R0.

GATES:
- G0 correctness; threshold all PASS; observed PASS; PASS
- G1 signed minus matched positive-only macro pAP; threshold >= +1.00 pp; observed +1.0319407402895497 pp; PASS
- G2 signed-better breadth; threshold >= 8/12; observed 12/12; PASS
- G3 signed minus matched positive-only macro pAUROC; threshold >= -0.50 pp; observed +0.23619865972595022 pp; PASS
- Medical reads; threshold zero; observed zero; PASS
- Phase2B training steps; threshold zero; observed zero; PASS

DECISION:
- CONTINUE

NEXT_STAGE:
- Stage R1 — GT-free action predictor under fixed LOCO logistic-regression and selective-risk gates.

NOTES:
- R0B is not triggered because R0 passed.
- No Medical dataset or sample was accessed.
- All R0 scientific artifacts must be committed and local/remote equality verified before R1 implementation.

### R0 Publication Verification and R1 Contract Authorization

- timestamp: `2026-08-23T02:25:22+07:00`
- R0 output commit: `571c26b07209c3005837deebfc306f1e3d1b433b`
- remote branch: `origin/research/p6-sabra-car-v1`
- local HEAD: `571c26b07209c3005837deebfc306f1e3d1b433b`
- remote HEAD: `571c26b07209c3005837deebfc306f1e3d1b433b`
- divergence: `0 0`
- decision: R0 publication gate PASS; R1 contract preparation authorized.
- R1 contract artifact: `research/sabra_car/R1_IMPLEMENTATION_CONTRACT.md`
- source cache fields 1-9: exact canonical GT-free shards, 2,162 records, 12 classes
- stability fields 10-11: Trust-v2 `where(valid_p9,S9,0)` and `where(valid_p16,S16,0)`
- Trust-v2 hashes/path alignment/finiteness: PASS
- Trust-v2 p9/p16 validity on VisA: 100%
- solver runtime: existing Python 3.14.0, NumPy 1.26.4, scikit-learn 1.7.2 environment
- deployment runtime remains the R0 `torchhuy` environment
- scientific R1 results observed: none
- R1 labels fitted: no
- Medical accessed: no
- Phase2B training steps: 0
- next action: commit and push the R1 contract, then implement the additive split-runtime LOCO sidecar and deterministic tests.

### R1 Implementation Validation

- timestamp: `2026-08-23T02:31:47+07:00`
- input contract commit: `da8ac2b5cf137f96ac8ddda1c11739d9c0f13ad2`
- implementation files: `r1_common.py`, `r1_fit.py`, `r1_evaluate.py`
- regression file: `tests/test_car_r1_action.py`
- targeted R0/R1/canonical suite: 20 passed
- pure-NumPy feature order, supported S9/S16 masks, linear quantiles, IQR floor, stable argmax, confidence abstention, risk selection, and LF serialization: PASS
- scikit-learn constructor contract: PASS
- fitter environment: Python 3.14.0, NumPy 1.26.4, scikit-learn 1.7.2
- evaluator environment: canonical `torchhuy`
- fitter and evaluator CLI/syntax checks: PASS
- cache fields/hashes/path alignment: PASS
- scientific R1 results observed: none
- full or partial LOCO model fit performed: no
- Medical accessed: no
- MVTec accessed: no
- Phase2B training steps: 0
- next action: commit and push implementation, verify equality, then run a bounded non-decision fit timing probe.

### R1 Engineering Bug R1-ENG-001

BUG_ID:
R1-ENG-001

symptom:
- The bounded timing probe stopped in provenance validation with `canonical GT-free manifest contract failed` before loading any feature shard.

root cause:
- The validator required `mask_pixels_read is False`, while the immutable manifest correctly represents that zero-read counter as numeric `0`; the other related counters use booleans or numeric zero.

scientific impact:
- None. No feature shard was loaded into a model, no LOCO estimator was fit, no held-out probability was produced, and no R1 scientific result was observed.

fix:
- Accept only boolean `false` or numeric zero for every canonical zero-read counter, while continuing to reject missing, true, or nonzero values.

regression test:
- `test_manifest_zero_read_counters_accept_boolean_or_numeric_zero_only`

validation:
- timestamp: `2026-08-23T02:34:37+07:00`
- targeted suite: 21 passed.
- `git diff --check`: PASS.
- Medical accessed: no.
- MVTec accessed: no.
- Phase2B training steps: 0.

status:
- FIXED; publish before retrying the bounded timing-only fit.

### R1 LOCO Runtime Estimate and Fit Authorization

- timestamp: `2026-08-23T02:37:14+07:00`
- input fix commit: `389246d610c78fa5e679f11e980412349be51345`
- first timing probe: first 20,000 patches per 11 training classes; rejected as runtime-representativeness evidence because KEEP was over-sampled at 83%
- representative timing probe: 20,000 evenly spaced patches per 11 training classes
- representative probe patches: 220,000
- representative action counts SUPPRESS / KEEP / BOOST: 105,369 / 104,987 / 9,644
- load and complete hash validation: 1.6374666751362383 seconds
- representative subset fit: 33.05409368686378 seconds
- representative subset iterations: 1,000 with one convergence warning
- projected full-fold fit: 403.5571293311591 seconds
- projected 12-fold fit before overhead: approximately 81 minutes
- conservative EXPECTED_RUNTIME_MIN: 100
- EXPECTED_FINISH_TIME: `2026-08-23T04:17:14+07:00`
- CPU resources: 28 logical processors
- available RAM: 24 GiB; swap available 5.4 GiB
- representative peak RSS: 371,388 KiB
- available disk: 209 GiB
- existing SABRA-CAR result size: 88 MiB
- execution mode: one attached blocking CPU command
- fail-closed behavior: stop before writing a fold prediction if that full fold does not converge within the frozen 1,000 iterations
- scientific R1 results observed: none
- Medical accessed: no
- MVTec accessed: no
- Phase2B training steps: 0
- next action: publish this estimate, then run the complete fixed LOCO fitter.

### Stage R1 — GT-Free Action Predictor

START_TIME:
2026-08-23T02:25:22+07:00

END_TIME:
2026-08-23T02:46:12+07:00

INPUT_COMMIT:
`da8ac2b5cf137f96ac8ddda1c11739d9c0f13ad2`

OUTPUT_COMMIT:
Recorded in the next append-only publication entry after commit creation.

PURPOSE:
- Learn the preregistered GT-free three-class action predictor under strict leave-one-class-out isolation and determine whether a selective deployable action policy passes risk and efficacy gates.

HYPOTHESIS:
- The fixed multinomial predictor can converge within 1,000 LBFGS iterations and select a threshold with at least 10% coverage, at most 5% opposite-sign error, at least 25% relative risk reduction, and at least +0.50 pp source pAP.

ALLOWED_CHANGES:
- Additive R1 validation, LOCO fit/evaluation sidecars, deterministic tests, exact failure evidence, and mandatory stop/handoff artifacts.

FORBIDDEN_CHANGES:
- Alternate solver/model family, more than 1,000 iterations, feature/order changes, sampling, threshold changes, neural fallback, Phase2B updates, MVTec selection, or Medical access.

DATA_USED:
- dataset: immutable GT-free VisA feature caches plus committed R0 oracle action labels
- records/classes: 2,162 images, 2,959,778 patches, 12 classes
- split: fixed 12-fold leave-one-class-out
- whether GT was used: R0 oracle labels were consumed as source targets; no masks were opened in R1
- whether Medical was accessed: no
- whether MVTec was accessed: no

IMPLEMENTATION:
- `tools/sabra_car/r1_common.py`: provenance, exact features, fold scaling, threshold/risk logic
- `tools/sabra_car/r1_fit.py`: scikit-learn multinomial LOCO fitting and portable numeric artifacts
- `tools/sabra_car/r1_evaluate.py`: canonical held-out action deployment, never reached
- `tests/test_car_r1_action.py`: deterministic contract regressions
- split runtimes: scikit-learn fitting in `Thai`; canonical deployment reserved for `torchhuy`

COMMANDS:
- bounded first-block timing probe, 220,000 patches; rejected as representative timing evidence
- bounded evenly-spaced timing probe, 220,000 patches
- `/home/ai4/ENTER/envs/Thai/bin/python -m tools.sabra_car.r1_fit`

TESTS_BEFORE_RUN:
- targeted R0/R1/canonical suite: PASS, 21 tests
- solver constructor: PASS
- both-runtime syntax/CLI: PASS
- source/Trust-v2 hashes and identity alignment: PASS
- feature finiteness and exact 11-field order: PASS
- Medical/MVTec zero-read provenance: PASS

RUN:
- run directory: `results/sabra_car/r1`
- full fitter input commit: `58312d4`
- expected runtime: 100 minutes maximum budget
- actual behavior: first full fold failed closed after approximately 375 wall-clock seconds
- failing fold: held-out `candle`
- estimator classes: `[-1,0,1]`
- observed iterations: 1,000
- convergence warning: LBFGS total iteration limit reached
- fold prediction written: no
- threshold landscape written: no
- OOF evaluation run: no

RESULTS:
- selected threshold: NOT_RUN
- coverage: NOT_RUN
- opposite-sign rate/reduction: NOT_RUN
- macro pAP/pAUROC: NOT_RUN
- per-class breadth: NOT_RUN
- downstream stages and final benchmarks: NOT_RUN

EXPECTED_VS_OBSERVED:
- expected: every fold converges within the fixed 1,000-iteration budget before risk/efficacy evaluation
- observed: the first full fold exhausted exactly 1,000 iterations and failed the frozen correctness gate
- interpretation boundary: no valid held-out prediction exists, so no scientific efficacy or risk claim is made

BUGS:
- R1-ENG-001: canonical manifest used numeric zero for `mask_pixels_read`; validator fixed to accept only boolean false or numeric zero
- no authorized engineering fix exists for the full-fold convergence stop because solver and max_iter are frozen scientific protocol parameters

SCIENTIFIC_INTERPRETATION:
- R0 oracle direction remains a valid positive mechanistic result.
- The preregistered GT-free predictor is not numerically certified under its fixed contract.
- Adjusting solver, tolerance, iterations, scaling, sampling, or model family after this outcome would be post-hoc and is forbidden.
- Therefore the deployable CAR hypothesis is unproven and the tree stops before R2.

GATES:
- common provenance/no-Medical/no-MVTec/no-Phase2B-update gates; observed PASS
- full-fold estimator convergence before 1,000 iterations; observed FAIL at exactly 1,000
- risk-qualified threshold exists; observed NOT_RUN
- coverage >=10%; observed NOT_RUN
- opposite-sign rate <=5%; observed NOT_RUN
- relative risk reduction >=25%; observed NOT_RUN
- macro pAP delta >=+0.50 pp; observed NOT_RUN
- macro pAUROC delta >=-0.50 pp; observed NOT_RUN
- non-negative pAP breadth >=7/12; observed NOT_RUN

DECISION:
- STOP

NEXT_STAGE:
- Final decision publication only. R2, R3, R4, Freeze, MVTec development, and retrospective Medical evaluation are NOT_RUN.

NOTES:
- Failure evidence is preserved in `results/sabra_car/r1/FIT_FAILED.json`.
- Schema-valid downstream status is preserved in `results/sabra_car/final_results.csv` and `final_results.json`.
- No Medical dataset or sample was accessed.
- A future attempt requires a new preregistration and must not mutate this stopped run.

### Final Publication Verification

- timestamp: `2026-08-23T02:48:56+07:00`
- final decision commit: `44d7ab77880d548ddc5702853e5375e5610ae160`
- remote branch: `origin/research/p6-sabra-car-v1`
- local HEAD: `44d7ab77880d548ddc5702853e5375e5610ae160`
- remote HEAD: `44d7ab77880d548ddc5702853e5375e5610ae160`
- divergence: `0 0`
- worktree before closing audit: clean
- final scientific status: STOP AT R1 CONVERGENCE CORRECTNESS GATE
- downstream R1 efficacy/R2/R3/R4/Freeze: NOT_RUN
- retrospective Medical benchmark: NOT_RUN
- Medical accessed: no
- MVTec accessed in R1: no
- Phase2B training steps: 0
- force push used: no
- next action: publish this closing audit record, verify equality once more, and stop.
