# CIR_DFG_RMT_V1 report

This report is a handoff scaffold for the frozen architecture. It contains no
target-derived or approximate metrics. Full experiment rows are written only by
the exact evaluator after the user launches the blocking runner.

## Identity

- architecture: CIR_DFG_RMT_V1 v1
- config: configs/cir_dfg_rmt_v1.json
- evaluator: CIR_FINAL_EXACT_V1
- source roles: VisA or MVTec only
- target roles: the opposite industrial dataset and six medical datasets
- epochs: 12, 14, 16, 18, 20 (no best-epoch replacement)

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| CIR/G0-IDENTITY | PASS (development audit with dirty tree) | tools/cir_rmt/audit_identity.py |
| CIR/G1-MATH | PASS | 11 pytest tests; reference/optimized max error <= 1e-5 |
| CIR/G2-PARITY | NOT RUN | real CLIP asset and source image were not available |
| CIR/G3-PREFLIGHT | PASS (synthetic structural audit) | tools/cir_rmt/audit_preflight.py |
| CIR/G4-PROFILE | PASS (CPU score-path profile) | tools/cir_rmt/profile_runtime.py |
| CIR/G5-SMOKE | NOT RUN | scripts/cir_rmt/smoke_train.sh requires an approved source root and CLIP asset |

A clean-tree G0 and a real source-only G2/G3/G5 run are release prerequisites.
No full training or inference was launched by this handoff.

## Performance

Bounded CPU profile (`[S=3,B=2,P=1369,D=768]`, five timed steps after one
warm-up) measured the exact score paths as follows:

| path | mean wall time | throughput |
|---|---:|---:|
| native/reference | 0.024535 s | 81.5 images/s |
| CIR/reference | 0.024950 s | 80.2 images/s |
| CIR/optimized | 0.003095 s | 646.3 images/s |

Reference-versus-optimized maximum absolute error was `1.55e-6` (threshold
`1e-5`). No GPU before/after or peak-VRAM number is claimed because no real
asset-backed profile was authorized; the alpha=0 final-map value likewise
remains pending the real G2 parity gate.

## Results

The long-form and per-source metric CSVs are intentionally header-only until
the exact runner is launched. The evaluator records architecture, source,
target, epoch, checkpoint SHA, config SHA, git SHA, and evaluator protocol on
every row. Medical rows are never used to select transport alpha.

## Resume and safety

The runner is blocking and uses set -euo pipefail. It verifies all five
candidate checkpoints before evaluation, resumes only from an existing
identity-compatible last.pth, and fails closed on identity mismatch. The G5
smoke path requires 50 optimizer steps, persists `last.pth`, validates the
smoke manifest, then performs one resumed optimizer step. Full execution
requires RELEASE_LOCK=TRUE after G0-G5 release checks.
