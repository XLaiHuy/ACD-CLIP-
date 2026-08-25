# P27R1 CUDA804 Runtime-Recovery Preregistration

Status: `P27R1_RUNTIME_RECOVERY_PREREGISTERED`

P27R1 is a separately preregistered engineering-recovery lineage. It is not
P28, does not introduce a scientific hypothesis, and does not alter the frozen
P27 scientific protocol.

## Terminal parent and original consumed attempt

- Terminal parent SHA: `f319a55bf8dc70cd17e0bf42c412622bd3095940`
- Original P27 attempt UUID: `60dd4d8d-15cd-403e-b2b3-4b38f4e7da1a`
- Original status: `P27_ENGINEERING_STOP`
- Failure location: first Tier-A subprocess, before the first cache sample
- Completed folds: `0`
- Adapter, Phase2B, and CLIP optimization steps: `0`
- Immutable predictions and scored folds: `0`
- MVTec and Medical reads: `0`

The original marker and terminal evidence remain immutable. P27R1 does not
erase, relabel, continue, or silently rerun that attempt.

## Root-cause class

`HOST_RUNTIME_CUDA_PROPAGATION`

Observed failure: CUDA was available in the validated parent execution
context, but unavailable in the exact scientific Tier-A child. CUDA reported
error 804, “forward compatibility was attempted on non supported HW.” No
scientific computation had begun.

## Recovery scope

The only permitted objective is to correct environment/runtime propagation so
every exact scientific child resolves the actual host NVIDIA driver
`libcuda.so.1` before any CUDA toolkit forward-compatibility or stub library.

Allowed changes are limited to:

- shell/runtime wrappers;
- child-process environment construction and propagation;
- `LD_LIBRARY_PATH` sanitization;
- removal of CUDA `/compat` or `/stubs` precedence;
- explicit verified host NVIDIA `libcuda` precedence;
- tmux child environment behavior;
- focused runtime diagnostic and audit tests.

## Scientific freeze

The following are forbidden from change:

- P27 architecture and `RegionResidualAdapter`;
- Tier-A/Tier-B cache semantics, format, content, and FP32 dtype;
- loss, optimizer, learning rate, epochs, and batch size;
- teacher semantics, alpha, and region geometry;
- CLIP, Phase2B, P26 checkpoint, and their assets;
- VisA dataset, LOCO classes/splits/order;
- metrics, thresholds, aggregation, and the mandatory barrier requiring all 12
  immutable predictions before any scoring.

Frozen runtime versions remain PyTorch `2.5.1+cu121` and torchvision
`0.20.1+cu121`. No NVIDIA driver, CUDA, PyTorch, or scientific dependency
upgrade/downgrade is permitted.

## Engineering qualification

Before a recovery attempt marker, P27R1 must reproduce and explain error 804
without science; prove exact host `libcuda` resolution in parent, direct child,
Python subprocess, exact runner child, and relevant tmux child contexts; pass
focused runtime tests; run only a tiny exact-path frozen smoke; preserve frozen
asset/protocol hashes; and minimally revalidate cache parity, loss/gradient/
optimizer parity, LOCO firewall, held exclusion, and the scoring barrier.

## Recovery-attempt budget

After the recovery implementation is committed, pushed, remotely identical,
and all final gates pass, exactly one new P27R1 scientific recovery attempt is
authorized. Its new durable marker must link to the consumed original UUID and
identify `HOST_RUNTIME_CUDA_PROPAGATION` as its sole justification. Any failure
after that marker is terminal `P27R1_ENGINEERING_STOP`; no P27R2 is authorized
automatically.
