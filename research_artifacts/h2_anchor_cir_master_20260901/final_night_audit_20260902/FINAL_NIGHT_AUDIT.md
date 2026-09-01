# CIR-DFG-RMT-V2 final-night audit

Audit date: 2026-09-02. Scope: evaluation and forensic recheck only. No model, architecture, optimizer, loss, scheduler, RMT, deployment operator, configuration, or frozen artifact was modified. No corrected training run was started.

## Executive result

The exact same-E10 Medical evaluation is complete:

| Arm | Pixel AUROC | Pixel AP |
|---|---:|---:|
| R | 0.894299586 | 0.354425236 |
| RA | 0.878459585 | 0.371681473 |
| RCA | 0.869881268 | 0.374138190 |

RA is below R on Pixel AUROC and above R on Pixel AP. RCA is below RA on Pixel AUROC and slightly above RA on Pixel AP. This mixed result does not establish a robust Anchor or CIR/RMT gain. It is conditional on the current trajectory and, for RCA, on the pathological Anchor.

The decisive new finding is the Anchor scale defect. The current per-tensor relative-L2 Anchor gives equal normalized weight to zero-reference DFG tensors. At RA E10, the required weighted Anchor/task gradient ratio is `40068.9185`; at RA E16 it is `31138.6838`. The zero-reference DFG pre/post-norm biases dominate the Anchor. Decision-tree Case B applies: Geometry/SRTR is not authorized.

## Audit coverage

| Topic | Status | Evidence |
|---|---|---|
| H2 launch and `lambda_k` | PASS | `H2_LAUNCH_PROVENANCE_RECHECK.md/.json`; historical log args and payload metadata |
| Historical horizon | PASS: E1-E15 | `CHECKPOINT_HORIZON_AUDIT.md`; files/log prove 15 epochs |
| Same-E10 Medical | COMPLETE | `SAME_E10_MEDICAL.csv/.md` |
| R reproduction | MULTIPLE_FACTORS | `H2_TRAINING_REPRO_GAP.md` |
| Source overlap | COMPLETE: 96/96 | `SOURCE_GATE_OVERLAP.csv/.md`; status is in-distribution assessment |
| Anchor parameter scale | COMPLETE | `ANCHOR_PARAMETER_CONTRIBUTIONS.csv`; `ANCHOR_SCALE_AUDIT.md` |
| Anchor/task gradients | COMPLETE on fixed batch | `ANCHOR_GRADIENT_DECOMPOSITION.csv`; no optimizer step |
| Cross-trajectory reference | UNKNOWN | `CROSS_TRAJECTORY_REFERENCE_AUDIT.md`; new R E1 was not saved |
| Training-path parity | PARTIAL | `TRAINING_PATH_PARITY_RECHECK.md` |
| Train/deploy mismatch | SECONDARY | `TRAIN_DEPLOY_MISMATCH_RECHECK.md`; common operator, audit only |
| Old scheduler audit | INHERITED, CONFIRMED | `research_artifacts/cir_rmt_v2/forensics_20260830_pre_scheduler_fix/SCHEDULER_OPTIMIZATION_AUDIT.md`; old CIR bug remains `CIR_SCHEDULER_BUG_CONFIRMED` |
| Old alpha=0 versus alpha=.5 effect | INHERITED, CONDITIONAL | `research_artifacts/cir_rmt_v2/forensics_20260830_pre_scheduler_fix/inference_rmt_effect.csv`; no new current H2 alpha pair run |
| Peer/delta/MAD/saturation | INHERITED, NOT RERUN | current RCA telemetry and prior compact forensics; no redundant battery run |
| Stage/group attribution | INHERITED, NOT RERUN | prior compact `stage_group_attribution.csv`; Anchor audit adds family gradient attribution |
| Loss/gradient conflict | INHERITED plus Anchor probe | prior `LOSS_GRADIENT_AUDIT.csv`; new Anchor/task gradients |
| MVTec | INHERITED, NOT RERUN | committed H2 MVTec archive; no new MVTec run |

## Proven, correlational, and unknown

### Proven by artifacts

- Historical H2 used `lambda_k=.002`, `lambda_kg=.01`, Adam defaults, StepLR gamma `.9`, AMP, and a 15-epoch E1-E15 horizon.
- Current H2 master R/RA/RCA histories use the decayed StepLR and correct post-epoch scheduler state.
- The same-E10 Medical values and deltas are reproduced in the compact CSV.
- The source gate has exact 96/96 overlap with the training manifest and is not a holdout.
- The current per-tensor Anchor is dominated by zero-reference DFG tensors and has pathological weighted gradients on the fixed batch.
- Historical H2, current R, RA, and RCA use the same H2 train/deploy Gaussian deployment branch; the mismatch is common across arms.
- The old CIR-V2 scheduler defect was confirmed in the separate pre-fix forensic snapshot: the old CIR trainer constructed StepLR but did not call `scheduler.step()`.

### Correlational or conditional

- The historical H2-to-current-R Medical gap is associated with missing reproducibility state and training-path differences; it is not a clean single-cause estimate.
- Same-E10 RA-R is a conditional Anchor training contrast, not a general Anchor claim.
- Same-E10 RCA-RA is a conditional CIR/RMT training contrast on a trajectory already distorted by the Anchor.
- The old alpha=0 versus alpha=.5 table is an inference effect conditional on the buggy-trained representation; it is not a clean CIR-versus-Phase2B effect.
- The prior train/deploy map differences may affect absolute metrics, but commonality means they do not alone explain RA-versus-R.
- Existing peer/delta/MAD, stage/group, and loss-conflict measurements describe associations in the completed trajectories, not isolated mechanisms.

### Unknown or not run

- Exact stochastic reproduction of historical H2: historical seed, `PYTHONHASHSEED`, optimizer/scheduler/RNG state, and run-local launcher were not preserved.
- Same-trajectory validity of historical H2 E1 as the new R reference: new R E1 was not saved.
- A clean current-H2 alpha=0 versus alpha=.5 inference pair.
- Whether a scale-safe Anchor would recover R performance.
- Whether RMT improves a correctly trained, Anchor-free CIR arm.
- Whether geometry/SRTR would help after those confounds are removed.

## Root-cause ranking

1. **Primary: ill-conditioned Anchor formulation.** The `1e-12` per-tensor denominator/clamp makes zero-reference DFG bias tensors dominate; measured gradient ratios are tens of thousands to one.
2. **Major confound: incomplete historical-to-current reproduction contract.** Seed/RNG state and model-state checkpoint semantics differ; constructor flags also differ, and nonfinite skip epochs do not match.
3. **Major validation confound: source-gate leakage.** All 96 gate images were in the training manifest, so source results cannot support clean generalization claims.
4. **Historical protocol bug: old CIR scheduler omission.** This explains why the original pre-fix CIR benchmark could not isolate RMT from optimization; it does not explain the present H2 master contrast, whose scheduler state is correct.
5. **Secondary common operator mismatch: train versus Gaussian deployment.** It affects absolute train/deploy behavior but is shared across H2/R/RA.
6. **Unresolved mechanism: RMT peer/delta behavior.** RCA telemetry shows valid peers, nonzero transport, and substantial delta saturation, but the current conditional comparison cannot establish benefit.

## Decision and next experiment

The correct decision tonight is `STOP_AUDIT_NO_TRAINING`: no geometry/SRTR, no long E20 run, no multi-seed sweep, and no architecture redesign.

The single safe next experiment is a preregistered same-trajectory, globally normalized Anchor E1-E10 test after explicitly locking the seed/RNG contract and a clean source-validation split. It must save the shared E0 and R E1 state/hash before any Medical access. Only if that correction produces a clean, stable, source-valid baseline should a CIR/RMT causal test or geometry decision be reconsidered.

See `FINAL_NIGHT_DECISION.md` for the gate outcome and `MACHINE_HANDOFF_MANIFEST.md` for checkpoint hashes. The compact SHA256 list intentionally excludes itself.
