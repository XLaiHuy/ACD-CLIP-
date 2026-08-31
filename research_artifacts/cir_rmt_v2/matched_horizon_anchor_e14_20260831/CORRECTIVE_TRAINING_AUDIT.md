# Corrective training audit

Status: PASS.

The committed matched-horizon runner trained the frozen CIR-V2 identity on VisA through E14 only. It used FP32, AMP disabled, TF32 disabled, effective batch six, Adam betas (0.9, 0.999), eps 1e-8, weight decay zero, gradient clipping one, and StepLR(step_size=1, gamma=0.9).

Candidate checkpoint audit:

| epoch | image LR | text LR | prompt LR | scheduler last_epoch | scheduler _step_count | checkpoint SHA256 |
|---:|---:|---:|---:|---:|---:|---|
| E10 | 0.0003486784401 | 0.00017433922005 | 9e-05 | 10 | 11 | 58af2ea6e3d92232498e3cb9bcf40b251e7116cfbac9a34d1abd4b07487aeaf0 |
| E12 | 0.000282429536481 | 0.000141214768241 | 9e-05 | 12 | 13 | bbbbfc6e24ac9dd1bfa87b596e3f6fe17a1b06cee6f5d4522ef67c4147a7e2f9 |
| E14 | 0.00022876792455 | 0.000114383962275 | 9e-05 | 14 | 15 | 9b8fc5e7760037e772c9bd63d98ce56fcbbaa04f021258ca0d23aa8f2bf5ab81 |

The scheduler state is post-step and is saved before candidate checkpoint serialization. The E10 gate passed without metric-based early stopping. The E14 anchor is training-only and is not part of deployment.

Resume is guarded by the process lock, checkpoint identity, optimizer state, scheduler state, RNG state, and anchor-reference identity. Medical and MVTec evaluation were not run in this stage.
