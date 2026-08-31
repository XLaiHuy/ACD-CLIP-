# PA control preflight

Status: PASS before PA scientific training.

## Frozen identity

- Branch: `research/cir-dfg-rmt-v2-signfix`
- Base published commit before PA implementation: `c0c0b336260745f0b8f084fae1e63ec4cce084fd`
- Source: VisA training data, seed 0
- Fixed source sample for post-training analysis: the existing 96-image deterministic sample, sample seed 9014, from `pre_full_run_root_cause_lock_20260831/SOURCE_SAMPLE_IDENTITY.json`
- CLIP: `ViT-L/14@336`, asset SHA256 `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- Image size: 518; precision: FP32; AMP: false; TF32: false
- Effective batch: 6 (`micro_batch_size=6`, `grad_accum_steps=1`)
- Optimizer: Adam, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0`
- Learning rates: image `1e-3`, text `5e-4`, prompt `1e-4` with the canonical prompt policy
- Scheduler: `StepLR(step_size=1, gamma=0.9)`, stepped once after each epoch and before checkpoint serialization
- Gradient clipping: norm 1.0 once per optimizer update
- Loss: canonical `cls + seg + 0.001*kg + 0.0*k`
- Candidate epochs: E10/E12/E14/E16/E18/E20
- Architecture freeze SHA256: `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86`
- Canonical parent config compact SHA256: `d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb`

## Factorial references

The authoritative frozen trajectories were verified byte-for-byte at every candidate epoch:

- P: `corrective_matched_retrain_20260830/parent/phase2b/checkpoints/adapter_<epoch>.pth`
- C_OLD: `corrective_matched_retrain_20260830/cir/visa/seed0/checkpoints/epoch_<epoch>.pth`
- A: `matched_horizon_anchor_e14_20260831/visa/seed0/checkpoints/epoch_<epoch>.pth`
- PA anchor reference: the exact P_E14 checkpoint, SHA256 `3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2`

PA is a new model initialization. It must start from E1 and may not resume P, A, or C_OLD. The P_E14 checkpoint is used only as a frozen image-adapter reference.

## Intended intervention

PA differs from native Phase2B only by adding the train-only image-adapter penalty with `lambda_image_anchor=0.001`. The reference is frozen, is not registered in the optimizer, and has scope `image_adapter_parameters_only`.

PA differs from A primarily by removing CIR/RMT training: no peer search, delta intervention, transport, CIR training logits, or alpha-dependent inference is invoked.

## Preflight checks

1. Branch, HEAD, remote HEAD, and tracked worktree were verified.
2. P/C_OLD/A candidate checkpoint files and hashes were verified for E10/E12/E14/E16/E18/E20.
3. CLIP, config, architecture-freeze, source, and anchor identities were verified.
4. Static PA audit confirms the native `forward_phase2b` call and no CIR runtime import or CIR training output use.
5. Focused parity tests pass.
6. Real-asset bounded smoke and CPU-RNG resume smoke pass.

Medical and MVTec were not accessed by this preflight.
