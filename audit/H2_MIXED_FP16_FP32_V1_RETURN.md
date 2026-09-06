# H2 historical mixed FP16+FP32 V1 restoration

## Decision

`MIXED_RETURN_READY=YES`

`HISTORICAL_MIXED_FP16_FP32_V1` is restored from BF16 decision parent `3d5a3464619bf1cc9b6e7270dc3808b66105712d` on branch `research/h2-mixed-fp16-fp32-final-v1`, using historical scientific reference `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`.

No full E15 train and no Medical or MVTec evaluation was run. The only CUDA work was a bounded, source-only 30-batch smoke.

## Scientific and precision contract

| Item | Restored value | Status |
|---|---|---|
| Architecture | ViT-L-14-336, 518, groups 3, image/text adaptation 0.2 | PASS |
| LoRA | text 16/2; Conv-LoRA 8/2, kernels 3/5 | PASS |
| Loss | main + `0.01 KG` + `0.002 K` + Safe Anchor | PASS |
| Optimizer | Adam, historical default weight decay 0 | PASS |
| LR/scheduler | image .001, text .0005, prompt .00005; StepLR 1/.9 | PASS |
| Prompt | hybrid ctx4, phrase `a photo of a`, E1-E3 frozen, alpha 0/.05/.10/.20 | PASS |
| DFG/SS2D | attn 256/tau8, SS2D, gamma .2, weight-residual warmup010/.1 | PASS |
| Anchor | lambda `.0021633926715180626`, family-safe cap `.10` | PASS |
| Training semantics | `model.eval()` / `CLIP.eval()` with adapter gradients | PASS |
| Horizon | E15 primary | PASS |
| Parameters / optimizer | FP32 / FP32 | PASS |
| Autocast / scaler | FP16 / enabled | PASS |
| TF32 / BF16 | disabled / disabled | PASS |
| Transformer | native `nn.MultiheadAttention` AMP + native AMP MLP | PASS |
| Later transformer FP32 islands | OFF | PASS |

The DFG reference code explicitly widens its residual q/k operands to FP32. The runtime trace also records that PyTorch's surrounding historical autocast emits the `einsum` score tensor as FP16. This behavior is preserved exactly; disabling autocast around that kernel would change the scientific path relative to `31167af`.

Three named modes are separately encoded by `h2_clean.precision`, logged, included in resolved scientific configs, and written to new checkpoints:

- `HISTORICAL_MIXED_FP16_FP32_V1`: FP16 autocast, GradScaler, later transformer islands off.
- `BF16_V1`: BF16 autocast, no GradScaler, later transformer islands off.
- `REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL`: FP16 autocast, GradScaler, later transformer islands on.

Older checkpoint-v3 payloads remain readable because the new top-level identity fields are additive rather than new legacy-required keys.

## Historical-reference difference classification

The minimum requested source surface was diffed directly against `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`.

| File / difference | Classification | Restoration disposition |
|---|---|---|
| `model/adapter.py`: runtime-only dtype strings | LOGGING_AUDIT | retained; no tensor is changed |
| `model/adapter_modules.py` | SCIENTIFIC_MODEL | byte-identical |
| `model/clip.py` | SCIENTIFIC_MODEL | byte-identical |
| `dataset/__init__.py` and VisA JSONL | SCIENTIFIC_MODEL / DATA | byte-identical; manifest SHA `468463...6842` |
| `model/transformer.py`: custom attention and residual/MLP FP32 branches | NUMERICAL_REPAIR | code retained, explicitly disabled for historical mode |
| `train.py`: named autocast/scaler selection and explicit TF32 disable | PRECISION_POLICY | historical mode restored |
| `train.py`: finite-state checks, telemetry, dtype hook, batch identity | LOGGING_AUDIT | retained as observational checks |
| `train.py`: save/reload/RNG/dataloader state plumbing | CHECKPOINT_INFRASTRUCTURE | retained |
| `h2_clean/contract.py`: checkpoint-v3 identity and resume validation | CHECKPOINT_INFRASTRUCTURE | retained, additive compatibility preserved |
| `h2_clean/precision.py`: explicit three-mode registry | PRECISION_POLICY | retained |
| `test.py`: explicit trusted full-checkpoint loading | EVALUATOR | retained; no evaluation run |
| BF16, replication, observability, freeze, and restoration scripts | TOOLING_ONLY | retained; restoration runner invokes source smoke only |

The dormant transformer repair is proven off both by configuration/checkpoint identity and by the CUDA trace: block 1 native attention received FP16, native MLP FC/projection produced FP16, and `later_transformer_fp32_islands_active=false`.

## Static parity gates

All required gates were frozen before CUDA execution in `audit/H2_MIXED_FP16_FP32_V1_PARITY.csv`. Architecture, Anchor, DFG, SS2D, prompt, optimizer, LR, scheduler, loss, data, augmentation, batch, horizon, FP16 autocast, GradScaler, BF16 exclusion, DFG operand widening, and historical transformer path are PASS. `LATER_FP32_TRANSFORMER_ISLANDS_ACTIVE=NO`.

## Runtime and resume evidence

The source smoke ran five E1 batches from fresh initialization and then reloaded/resumed the exact E1 checkpoint for five batches per epoch through E6: 30 attempted batches and 30 successful optimizer steps. It observed zero nonfinite losses, gradients, parameters, and optimizer states. The smoke is not expected to reproduce rare full-horizon overflows.

Trainable counts were exactly `14,095,887` for prompt-frozen E1-E3 and `14,102,031` after prompt unfreeze. Image/text StepLR values and the zero/active prompt LR were checked at every resumed epoch. All observed effective active-family Anchor ratios stayed within the 0.10 cap.

Checkpoint-v3 contains model, optimizer, scheduler, GradScaler, Python/NumPy/Torch CPU/Torch CUDA RNG, dataloader generator state, config hash, dataset SHA, CLIP SHA, Git identity, and exact scientific mode. Independent reloads reproduced saved RNG and dataloader state byte-for-byte; 25 resumed augmented-batch identities were recorded. E1 and E6 checkpoint SHA256 values are in `audit/H2_MIXED_FP16_FP32_V1_SMOKE.md`; large checkpoints remain outside Git.

The dtype trace covers input, CLIP visual block, native attention projections/returned score path, native MLP, image adapter, Conv-LoRA, DFG residual, SS2D, text adapter, soft-prompt state, losses, parameters, gradients, and optimizer state. No BF16 tensor was observed. Parameters, gradients, and floating Adam state were FP32.

## Corrected numerical provenance and quality decision

Historical mixed A was **not** zero-event numerically clean. The authoritative reconciliation is:

- E2-E15 attempted batches: `5054`
- successful optimizer steps: `5052`
- recoverable nonfinite-gradient skips: `2`
- nonfinite-loss skips: `0`
- skip rate: `2/5054 = 0.03957%`

This corrects the stale zero-event statement in the earlier BF16 decision prose. BF16 was numerically cleaner, but it was materially worse under the same Seed0 comparison:

| Domain | Mixed A Seed0 AUROC/AP | BF16 A Seed0 AUROC/AP | BF16 minus mixed |
|---|---:|---:|---:|
| Medical pixel | 91.2518 / 39.4684 | 90.2259 / 36.3341 | -1.0259 / -3.1342 |
| MVTec pixel | 90.0413 / 45.1593 | 86.1169 / 40.5415 | -3.9244 / -4.6179 |

Therefore mixed FP16+FP32 is restored as the quality-preferred research protocol with numerical status `ACCEPTABLE_WITH_RARE_RECOVERABLE_AMP_SKIPS`.

Future acceptance remains: zero nonfinite losses, parameters, and optimizer corruption; recoverable gradient skips at most 0.1%; no repeated/consecutive pattern; no escalating scaler instability.

`FULL_TRAIN_AUTHORIZED=NO`

`WAITING_FOR_USER_APPROVAL=YES`
