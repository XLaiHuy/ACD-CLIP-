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
