# Why current R does not exactly reproduce historical H2

Classification: `MULTIPLE_FACTORS`.

Historical H2 E10 is `0.9092218791 / 0.4037306455` Pixel AUROC/AP under the current exact evaluator. New current R E10 is `0.8942995859 / 0.3544252360`. Existing evaluator migration checks found no material evaluator explanation for this gap. The gap is not safely reducible to one stochastic seed effect because the historical run does not record the seed or RNG states and also differs in training-state plumbing.

| Item | Historical H2 | Current R | Assessment |
|---|---|---|---|
| Explicit seed | Not recorded; historical `train.py` has no seed setup | Seed `0`, stored in E0 identity; deterministic restoration used by runner | Reproduction contract mismatch |
| Python/NumPy/torch CPU/CUDA RNG | Not recorded in historical run | E0/checkpoint RNG state is preserved/restored | Reproduction contract mismatch |
| `PYTHONHASHSEED` | Not recoverable | Not part of the historical evidence | Unknown uncontrolled factor |
| DataLoader shuffle/workers | `shuffle=True`, `num_workers=6`, pin memory | Same effective batch/worker setup; current runner uses explicit config | Structural parity, seed stream not provable |
| Augmentation/worker randomness | Depends on unrecorded historical process RNG | Current process uses restored/explicit RNG contract | Cannot isolate as mere noise |
| AMP/GradScaler | AMP enabled; scaler created but not saved in model-only checkpoints | AMP enabled; scaler/checkpoint state saved | Same intended precision, different replay state |
| Optimizer | Adam defaults, text/image/soft groups | Adam defaults, text/image/soft groups | Matched hyperparameters |
| Scheduler | StepLR gamma `.9`, post-epoch call | StepLR gamma `.9`, post-epoch call | Matched intended schedule |
| Nonfinite skip pattern | E1=2, E2=0, E3=1, E11=1 | E1=2, E2=1, E3=0, E4=0, E5=1, E6-E10=0 | Same total observed skips (4), different epochs |
| Global optimizer steps | Not recorded in historical log/checkpoints | R E10 global step `3606` | Historical value unavailable |
| Prompt/hybrid schedule | Hybrid alpha max `.2`, frozen E1-E3, prompt LR `.00005` after unfreeze | Same recorded schedule | Matched intended schedule |
| Constructor flag | Historical `use_soft_prompt=False`, then hybrid flag enabled | Current constructor passes `use_soft_prompt=True`, then hybrid flag enabled | Training-path difference; fixed-input parity does not erase it |
| K-reg/KG-reg | `lambda_k=.002`, `lambda_kg=.01` | Same config and checkpoint metadata | Matched |

The different nonfinite-gradient epochs are consistent with different random/initialization trajectories, but the missing historical seed/RNG contract and the constructor/state differences make `STOCHASTIC_TRAJECTORY_DIFFERENCE` too weak as the sole classification. No multi-seed sweep was run. No claim of exact historical replay is made.
