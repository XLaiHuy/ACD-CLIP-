# H2 new-machine three-layer recovery audit

## Scope

This audit records the state recovered on 2026-09-06 before the R1 bounded
mechanism screen. It covers repository/artifact recovery, scientific-contract
parity, and the completeness of the existing source-only evidence. It does
not authorize full E15 training or Medical/MVTec inference.

## Tier 1 — system, repository, and artifacts

`TIER1_RECOVERY=PASS`

- Current branch at audit start: `research/h2-functional-anchor-bounded-r1`.
- Current HEAD at audit start: `d28516f2dafb1eb23c5af634d920fa14e585c26c`.
- The remote branch has the same HEAD; no newer related R1 or bounded branch
  was found. The remote `research/h2-functional-anchor-bounded-v1` remains the
  superseded invalid R0 endpoint.
- Git LFS objects are present and the four required SHA256 values pass:
  CLIP, shared E1, historical H E15, and historical A E15.
- VisA resolves through the restored symlink, has 2,162 manifest rows, 12
  classes, 1,200 anomaly masks, and zero missing image or mask paths.
- Disk capacity is ample and CUDA is available on one RTX 3090 with 24 GiB.
- The only pre-audit worktree changes are untracked dataset symlinks under
  `data/`; raw datasets are intentionally outside Git.

## Tier 2 — scientific parity

`TIER2_SCIENTIFIC_PARITY=PARTIAL`

The frozen checkpoint contract verifies the historical mixed protocol:
FP32 parameters, FP16 autocast, GradScaler, BF16 off, TF32 matmul off,
native transformer AMP, DFG attention dimension 256, tau 8, true SS2D,
gamma max .2, weight-residual fusion, beta warmup010 to .1, image/text
adaptation .2, the recorded LoRA/Conv-LoRA ranks and kernels, prompt ctx4
with `a photo of a`, the E1-E3 prompt freeze, the recorded learning rates,
Adam/StepLR, batch 6, and gradient clip 1.

The current isolated environment matches the archived Torch/CUDA package
versions (`torch 2.12.1+cu130`, `torchvision 0.27.1+cu130`, NumPy 2.4.6,
pandas 3.0.3, Kornia 0.8.3, Pillow 12.2.0, pytest 9.1.1), and current
repository SS2D is dependency-free. The host cannot provide archived Python
3.11.15 or the archived RTX 5060 Ti, so bitwise machine parity is not claimed;
the current runtime is Python 3.12.3 on an RTX 3090.

## Tier 3 — evidence and research state

`TIER3_EVIDENCE_COMPLETENESS=PARTIAL`

All required compact JSON artifacts parse, all required CSV artifacts parse,
and checkpoint identities/configuration agree with their recorded SHA256
values. The final mixed diagnostic report covers 48 exact cohorts, all 2,162
source images, and the expected routing/geometry/family/gradient/prompt
families. The R1 protocol, 16-batch no-update parity preflight, and fixed
8-batch calibration are internally consistent.

The large external array root named by the historical frozen-state artifact
is not present on this machine. Compact artifacts are sufficient for the
recorded bottleneck decision, but raw-array reinspection is therefore not
claimed. The earlier invalid bounded endpoint is retained as provenance and
is not treated as evidence against R1.

## Recovery decision

`RECOVERY_STATUS=PARTIAL`

The scientific state is recoverable and the authorized next action is the
single preregistered source-only R1 screen. Full E15, target inference,
target tuning, and hyperparameter sweeps remain prohibited.

