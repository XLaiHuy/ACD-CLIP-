# H2 final protocol audit

`FINAL_PROTOCOL_AUDIT=PASS`

Audit performed once after all result artifacts were produced.

## Frozen training identity

- Run root: `/tmp/h2_clean_factorial_e20_20260902_ampfix`
- Freeze manifest: `audit/H2_4ARM_E15_E20_FREEZE.json`
- Freeze commit: `e0edadd8d4433de94495a443b375f6b8a28e9eaa`
- Full-run code SHA: `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`
- Shared E1 SHA256: `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`
- All eight selected checkpoints exist and match their recorded SHA256 values: `PASS`
- Same shared E1, CLIP, dataset manifest, base-H2 commit, CIR reference commit,
  and full-run code identity across selected checkpoints: `PASS`
- Correct intervention flags with no A/C cross-contamination: `PASS`
- Exact CIR V2 reference commit: `9cc0ad4cc6b34e34a8c15e74df881866516b3181`
- Active Anchor lambda: `0.0021633926715180626`; family cap `rho=0.1`: `PASS`
- Full training: `H=PASS`, `A=PASS`, `C=PASS`, `AC=PASS`
- Non-finite loss skips: `H=0`, `A=0`, `C=0`, `AC=0`
- Non-finite gradient skips: `H=1` isolated, `A=0`, `C=0`, `AC=0`

## Target-evaluation protocol

- E15 freeze before target: `PASS`
- E20 freeze before target: `PASS`
- Medical evaluator commit: `6bd932fbce0a425af5c8d3f7230dd7dc041568bd`
- Medical: all four arms at E15 and E20, fixed `current_shared`, `cls_only`,
  raw exact metrics, `pixel_stride=1`: `PASS`
- MVTec: all four E15 primary arms, `benchmark_exact`, `pixel_stride=1`,
  no checkpoint selection or tuning: `PASS`
- Medical preceded MVTec: `PASS`
- Target checkpoint selection: `NO`
- Target hyperparameter tuning: `NO`
- Intermediate target-based decisions: `NO`

## Medical primary results

| arm | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|
| H15 | 90.81533202898113 | 35.8742528791304 | 76.14876826604207 | 76.25000079472859 |
| A15 | 91.25178605739472 | 39.46836969631676 | 75.2827266852061 | 76.34336352348328 |
| C15 | 90.22026982806102 | 34.49293673731004 | 75.51521857579549 | 75.46531558036804 |
| AC15 | 90.80128589797694 | 37.0541690846828 | 76.61393483479817 | 76.69869860013326 |

Primary factorial effects (AUROC/AP):

- `ANCHOR_E15_EFFECT=+0.4364540284135927/+3.594116817186361`
- `CIR_E15_EFFECT=-0.5950622009201112/-1.3813161418203634`
- `AC_E15_EFFECT=-0.01404613100419283/+1.179916205552395`
- `INTERACTION_E15=+0.14456204150232566/-1.0328844698136024`
- `ANCHOR_SUPPORT=PASS`
- `CIR_SUPPORT=FAIL`
- `AC_SUPPORT=FAIL`
- `INTERACTION=NEUTRAL`

Secondary E20 effects (AUROC/AP):

- `ANCHOR_E20_EFFECT=-0.007880075430477973/+2.907670420452`
- `CIR_E20_EFFECT=-0.6947236241605168/-0.4301721816308941`
- `AC_E20_EFFECT=-0.3391962440528147/+2.3251167486069946`
- `INTERACTION_E20=+0.3634074555381801/-0.15238149021411118`

The preregistered minimal-winner rule gives:

`FINAL_DIRECTION=A`

Reason: A Pareto-beats H on both primary E15 pixel metrics, while C fails;
the rule therefore selects A, and E20 cannot rescue or overturn E15.

## Matched industrial transfer confirmation

MVTec E15 summary (the evaluator log presents these exact computations to six
decimal places):

| arm | pixel AUROC | pixel AP | image AUROC | image AP |
|---|---:|---:|---:|---:|
| H15 | 86.868623 | 41.612306 | 89.526111 | 95.207364 |
| A15 | 90.041289 | 45.159349 | 89.816913 | 94.779362 |
| C15 | 87.209866 | 41.040610 | 88.254977 | 94.749204 |
| AC15 | 89.756252 | 45.416002 | 89.543194 | 94.748386 |

This is a matched industrial transfer confirmation. MVTec was previously
inspected in the project, so no globally untouched-test claim is made. It was
not used to retune or change the final selection.

## Handoff

- Compact Medical results: `results/H2_4ARM_E15_MEDICAL_PER_DATASET.csv`,
  `results/H2_4ARM_E15_MEDICAL_SUMMARY.csv`,
  `results/H2_4ARM_E20_MEDICAL_PER_DATASET.csv`,
  `results/H2_4ARM_E20_MEDICAL_SUMMARY.csv`
- Compact MVTec results: `results/H2_4ARM_E15_MVTEC_PER_CLASS.csv`,
  `results/H2_4ARM_E15_MVTEC_SUMMARY.csv`
- Factorial effects: `results/H2_4ARM_FACTORIAL_EFFECTS.csv`
- Decision: `results/H2_4ARM_FINAL_DECISION.md`
- Medical provenance: `results/H2_4ARM_MEDICAL_EVAL_MANIFEST.json`
- Result payload commit SHA: `e776e65` (`Publish H2 factorial Medical and MVTec results`)
- Result payload push to `huy/research/h2-clean-repro-anchor-cir-v1`: `PASS`
- Final audit metadata update and its remote head: recorded in the final Git handoff
- Tracked worktree: verified clean after the final audit metadata update
