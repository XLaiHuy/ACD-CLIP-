# P27R1 CUDA804 Runtime Recovery

Status: `ENGINEERING_QUALIFICATION_PASS`

## Exact root cause

The tmux server was created at 14:15 UTC with only
`/usr/local/cuda/lib64` in `LD_LIBRARY_PATH`. The host-driver-first profile
sanitizer was installed at 14:45 UTC, so the already-running tmux server never
inherited it. `/usr/local/cuda/lib64` supplies no `libcuda`; the dynamic loader
therefore used the first `ldconfig` candidate,
`/usr/local/cuda-12.9/compat/libcuda.so.575.57.08`. That CUDA 12.9
forward-compatibility driver API is incompatible with the host 570 driver and
caused CUDA error 804.

The validated direct parent and all non-tmux child contexts instead resolved
the real host driver at
`/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08` and exposed the RTX 3070 Ti.

## Minimal fix

`build_p27_cuda_environment` now constructs every scientific child environment
explicitly. It removes only library-path components named `compat` or `stubs`,
deduplicates the remaining entries, prepends the verified host driver
directory, sets the already-frozen cuBLAS workspace policy, and tags the
recovery version. It does not change `CUDA_HOME` or `CUDA_PATH`, because both
were absent and were not causal.

Before any P27R1 marker, the exact runner now launches a recovered child probe
and requires PyTorch `2.5.1+cu121`, CUDA build `12.1`, host `libcuda`, and an
available GPU. The identical recovered environment is then propagated to every
cache, training, prediction, scoring, and aggregation subprocess.

## Qualification

- Error 804 reproduced in interactive tmux and tmux-created children.
- Three of three normal recovered children passed.
- Three of three children launched from the stale tmux environment passed.
- Every fixed child resolved host `libcuda.so.570.172.08`; none resolved compat
  or stubs.
- Exact-path smoke loaded immutable assets and performed one GT-free frozen
  forward with exact FP32 `[3,1,1369,768]` features and `[3,1,1369,2]` logits.
- Smoke teacher/training/prediction/scoring counts were all zero.
- Focused runtime and P27 equivalence suite: 27 passed.
- Cache tensors, loss, gradients, optimizer step, LOCO firewall, held exclusion,
  and the 12-prediction scoring barrier pass.

The validated cache architecture and performance code did not change, so the
previous 9.0475x benchmark and 49.2494 GiB projection remain authoritative.
No P27R1 recovery scientific attempt has been consumed.
