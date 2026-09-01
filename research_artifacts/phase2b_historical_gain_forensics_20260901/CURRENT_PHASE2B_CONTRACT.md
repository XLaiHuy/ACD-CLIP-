# Current corrected Phase2B C2 contract

Status: recovered from the canonical config, corrected training manifest, checkpoint payloads, and current trainer.

C2 uses the same VisA source, seed, CLIP asset, model dimensions, DFG settings, batch/effective batch, Adam defaults, base image/text learning rates, gradient clip, prompt alpha/freeze schedule, and StepLR gamma=0.9. It differs from H2 in multiple scientific/protocol fields: FP32 without AMP, lambda_kg=0.001, lambda_k=0 with an explicit zero stub, soft_prompt_lr=1e-4, 20 epochs and a different candidate schedule, and the current exact evaluator with pixel stride 1.

The corrected trainer calls scheduler.step() once after every epoch and before the history row and candidate checkpoint save. The raw canonical config hash is d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb; the resolved run configuration used by the C2 checkpoint is 5ec0190ec4dc1e16e0ce646b5e470d5585b981de7221ed4b46a392b321cd27f9.
