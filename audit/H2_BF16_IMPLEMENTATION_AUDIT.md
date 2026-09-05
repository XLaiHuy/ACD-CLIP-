# H2 BF16 implementation audit

## Decision and scope

The old FP16 run is retired after recurrent overflow in visual-transformer
backward paths. This branch implements the authorized global CUDA BF16
autocast policy with FP32 parameter storage, FP32 Adam state, TF32 disabled,
and no `GradScaler`. It is a `SCIENTIFIC_PROTOCOL_CHANGE`, so BF16 outcomes
must be trained and reported as a separate matched set.

The implementation adds an explicit `PrecisionPolicy`, stores precision and
GradScaler state in the full checkpoint contract, and rejects a BF16/FP32
resume checkpoint with a nonempty scaler state. BF16 calls `loss.backward()`
and `optimizer.step()` directly. FP16 retains a GradScaler. The selected Mode
B sets `enable_fp16_numerical_islands=false` only under BF16; FP16 behavior is
unchanged.

## Dtype and optimizer audit

The CUDA autocast context is BF16 for the model/loss calculation area. CLIP,
image adapter, text adapter, soft prompt, and all optimizer-held tensors keep
FP32 storage. A fresh 150-step BF16 source run confirmed checkpoint model
state `float32`, Adam tensor state `float32`, `precision=bf16`,
`amp_enabled=true`, `gradscaler_enabled=false`, `scaler_state={}`, and
`tf32_enabled=false`.

Focused unit tests cover BF16 policy selection, CUDA BF16 activation with FP32
weight/gradient storage, BF16 checkpoint/resume state, and rejection of stale
precision metadata. The focused suite passed: **46 passed**. Existing FP16
attention/residual island tests also passed.

## Preserved failure-path tests

All replays use real model/loss/backward/optimizer code, preserved source
states, and the original source batches. The old FP16 checkpoints are used
only as numerical counterfactual input states; they are not BF16 scientific
checkpoints.

| historical path | Mode A: islands retained | Mode B: islands bypassed | result |
|---|---:|---:|---|
| Seed1-H E2 failing batch | finite | finite | pass |
| Seed1-H E7 batch 354 from E6 | 355 replayed batches, finite | 355 replayed batches, finite | pass |
| Seed1-A E15 batch 26 | finite | finite | pass |

For E7, Mode B produced 2,519 successful optimizer steps (2,164 inherited
plus 355 replayed), zero nonfinite losses/gradients, finite parameters, finite
optimizer state, and final group LRs `text=0.0002657205`,
`image=0.000531441`, `soft=0.00005`. Mode A produced the same validity result.
Mode B took about 8m23s for the replay versus 11m12s for Mode A.

The independent fresh 150-step Seed1-H BF16 sanity run was also finite:
150/150 optimizer steps, zero nonfinite loss/gradient events, finite
parameters and Adam state. Its mean loss was 1.535992; this bounded source
diagnostic is explicitly not a reusable shared-E1 scientific checkpoint.

## Performance decision

The three viable configurations used fixed Seed1 batches and BF16 Mode B.
Post-warmup runtime is reported in `H2_BF16_RUNTIME_BENCHMARK.csv`. The
selected six-worker configuration is marginally fastest at 1.304997 seconds
per step. Disabling activation checkpointing failed before its first forward
with CUDA OOM while requesting 688 MiB with 345 MiB free.

## Unchanged scientific settings

Architecture, parameter count, optimizer, LRs, scheduler, losses and weights,
batch size, gradient clipping, Anchor value/budget, DFG/SS2D, LoRA, prompt
schedule, deterministic ordering, and source horizon are unchanged. The only
scientific change is the globally explicit compute dtype / loss-scaling policy.
No target evaluation has run.
