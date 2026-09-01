# Training-path parity recheck

Overall status: `PARTIAL`.

Fixed-input parity between the H2 implementation and the extension already passed. That does not establish exact training-path equivalence. The recheck found the following:

| Detail | Result | Evidence/meaning |
|---|---|---|
| Model constructor | Partial | Historical H2 passes `use_soft_prompt=False` and then enables hybrid mode; current runner constructs with `use_soft_prompt=True` and enables hybrid mode. Fixed-input hybrid parity passed, but the constructor path is not byte-for-byte identical. |
| Hybrid flags and schedule | Matched intended behavior | Both use hybrid alpha max `.2`, freeze through E3, DFG beta `warmup010` to `.1`, and prompt LR `.00005` after unfreeze. |
| Optimizer | Matched | Adam defaults, no weight decay, betas `(0.9,.999)`, eps `1e-8`; group order text, image, soft prompt. |
| Loss placement | Matched structure | `cls + seg + .01*kg + .002*k`; current RA adds only `.001*anchor` outside the base loss. |
| K-reg precision/path | Matched | Detached W_K and per-stage K-reg path were tested and passed in the existing H2 parity audit. |
| AMP/GradScaler | Matched intended precision | AMP is enabled with the historical default float16 autocast; current checkpoints also save scaler state. |
| Gradient clipping | Matched | Image, text, and unfrozen prompt groups are clipped at norm 1.0. |
| Scheduler timing | Matched | `StepLR(step_size=1,gamma=.9)` is stepped after the epoch loop and before candidate checkpoint save. |
| Soft-prompt LR reapplication | Matched | Current runner reapplies the constant LR after scheduler step; histories show 0 through E3 and `.00005` afterward. |
| DataLoader | Effective parity | Batch 6, shuffle true, workers 6, pin memory true. Current config explicitly records `persistent_workers=false` and `prefetch_factor=2`; historical call relied on PyTorch defaults rather than recording them. |
| Nonfinite handling | Structure matched, trajectory differs | Both skip nonfinite gradients and abort only after >20 nonfinite losses; skip epochs differ because the trajectories differ. |
| Resume/RNG behavior | Not matched | Historical model-only checkpoints omit optimizer/scheduler/scaler/RNG; current E0 and run checkpoints preserve and restore these states. |
| Seed contract | Not matched | Historical seed is absent; current E0 uses seed 0. |

Thus optimizer, loss, scheduler, AMP, prompt policy, loader geometry, and RMT/CIR code paths are substantially aligned, but exact historical training reproduction is not established. The constructor and reproducibility/state differences are part of the `MULTIPLE_FACTORS` reproduction classification. No parity difference was modified in this audit.
