# PA control diff audit

Status: PASS.

The PA trainer is a separate entry point at `scripts/cir_rmt/train_pa.py`. The canonical `train.py` file is unchanged. PA imports and reuses the canonical loader, text regularizer, optimizer, epoch-state schedule, gradient clipping, checkpoint payload, and resume validation helpers.

| Contract | Evidence | Result |
|---|---|---|
| Native Phase2B training path | PA calls `model.phase2b_runtime.forward_phase2b` | PASS |
| No CIR training | No `forward_cir`, CIR runtime import, peer-validity gate, CIR segmentation output, or RMT delta fields in PA trainer | PASS |
| Same objective | Canonical `calculate_seg_loss`, cross-entropy, `lambda_kg`, and `lambda_k` are reused | PASS |
| Same optimizer | Canonical `_make_optimizer` is reused | PASS |
| Same prompt/DFG policy | Canonical `_set_epoch_state` is reused | PASS |
| Same scheduler | `StepLR(step_size=1, gamma=0.9)` and one post-epoch `scheduler.step()` before checkpoint write | PASS |
| Same clipping/update boundary | Canonical `grad_accum_window_size` and `clip_trainable_gradients` are reused | PASS |
| Same anchor | P_E14 SHA is fixed and `lambda=0.001`; anchor metadata is serialized and validated | PASS |
| New initialization | PA launcher has no resume path on a fresh run and labels the P_E14 file only as `--image-anchor-checkpoint` | PASS |
| Resume safety | Resume loads CPU RNG tensors, restores optimizer/scheduler/RNG, and validates control/anchor identity | PASS |
| Precision | Config and trainer enforce FP32, AMP false, TF32 false | PASS |

The only intended factorial variable between PA and A is CIR training intervention: PA off, A on. Inference RMT is absent from PA and is not part of the primary factorial comparison.
