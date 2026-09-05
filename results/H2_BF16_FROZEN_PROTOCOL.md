# H2 BF16 frozen protocol

This is `H2_BF16_SCREENING_SEED1_V1`, a new scientific training protocol
rooted at clean handoff `27d869d96ffbfe9f5ff5d81553a1385af31cf226`. It uses
CUDA BF16 autocast with FP32 parameters and Adam state, no GradScaler, and
TF32 disabled. The selected implementation bypasses the retired FP16-only
local FP32 islands (`bf16_local_fp32_islands=false`).

The source run is a fresh Seed-1 VisA shared E1 followed by matched H and A
E15 continuations. It preserves the H2 architecture, batch size, losses,
optimizer, LR/scheduler, prompts, LoRA, DFG/SS2D, clipping, and Safe Anchor
settings. The exact configuration is in the adjacent JSON file and the
reproducible source runner is `scripts/run_h2_bf16_screening_seed1.sh`.

Runtime settings were selected from the benchmark in
`audit/H2_BF16_RUNTIME_BENCHMARK.csv`: activation checkpointing enabled,
six workers, pinned memory, no persistent workers, and ordinary host-to-device
copies. Disabling checkpointing exhausts RTX 3090 memory before the first
forward. The selected configuration records metrics every 25 steps and aborts
on any nonfinite loss, gradient, parameter, or optimizer-state event.

BF16 results are not a matched continuation of the historical FP16 results.
No target evaluator may run until both fresh BF16 `H/adapter_15.pth` and
`A/adapter_15.pth` have been frozen and verified.
