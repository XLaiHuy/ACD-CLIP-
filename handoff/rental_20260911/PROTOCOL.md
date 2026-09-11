# Fresh source-transfer protocol: 2026-09-11 rental

This directory defines the clean VisA-source and MVTec-all-source runs. The
historical launchers under `handoff/lab_20260910/` remain unchanged for
provenance and are not used here.

Both runs keep the frozen architecture and scientific settings from
`NEXT_EXPERIMENT_SPEC.md`: ViT-L-14-336 at 518 pixels, three groups, the
frozen Dual-Branch DFG and Conv-LoRA/text-LoRA/HPA settings, `lambda_kg=0.01`,
`lambda_k=0.002`, image/text learning rates `0.001/0.0005`, StepLR gamma
`0.9`, batch size `6`, seed `0`, AMP, gradient checkpointing, deterministic
algorithms, gradient clipping `1.0`, Family-Safe lambda
`0.0021633926715180626`, Family-Safe budget `0.10`, CIR disabled,
hard-background ranking disabled, 20 epochs, and checkpoints E1 through E20.
Only `num_workers` is operational and must be selected by the one-versus-two
worker preflight on this two-core rental.

## E1 reference and continuation state

Each source run is two-stage. E1 is trained fresh with the task configuration
and without a Family-Safe term because the fresh reference does not exist until
E1 has been produced. The exact E1 file is then the fixed Family-Safe Anchor
reference for E2 through E20, while the second invocation resumes E1 with
`--resume`.

The Anchor reference and training resume state are separate objects. The
reference is read from `adapter_1.pth` and remains immutable. Resume restores
model adapters, optimizer moments and parameter groups, scheduler state, AMP
scaler state, Python/NumPy/Torch RNG state, and the dataloader generator state.
The checkpoint contract allows the Anchor branch to change only at the E1 fork
and rejects other scientific identity changes. `smoke_e1_e2_continuation.sh`
checks this before a full run.

The old shared E1 Anchor under
`runs/h2_clean_factorial_e20_20260902_ampfix/` is intentionally absent from
both v2 launchers.

## Selection and final evaluation

Training has no TTA. All E1 through E20 checkpoints are evaluated on Brain,
Liver, Retina, Colon ClinicDB, ColonDB, and Kvasir with identity/no TTA,
`benchmark_exact`, and `pixel_stride=1`. The ranking rule is highest Medical
macro Pixel AP, then higher Medical macro Pixel AUROC, then earlier epoch.
`select_medical_epoch_v2.py` applies this rule to all six no-TTA `test.log` files
and writes both `medical_epoch_selection.csv` and `medical_epoch_selection.json`.
TTA is never used to select an epoch. These are **TARGET-SELECTED
EXPLORATORY TRANSFER EXPERIMENTS** because Medical targets select the checkpoint.

`eval_selected_checkpoint_v2.sh` evaluates the selected checkpoint separately
with Single-View and the locked Four-View policy
`identity,hflip,vflip,hvflip`. The Four-View implementation keeps an auditable
sequential baseline. A candidate must match baseline logits, inverse-mapped
segmentation maps, image scores, exact pixel/image metrics, runtime, and peak
CUDA memory on identical samples before it can be enabled. The wrapper writes
`DELTA_TTA.json`, including per-class and macro Four-View minus Single-View
metrics; image-level deltas are reported for Brain, Liver, and Retina.

Run V's industrial cross-domain target is MVTec AD. Run M's cross-domain
target is VisA. MVTec is the source for Run M and must not be described as an
untouched target.

## Execution order

Run `prepare_data_links.sh`, complete the v2 preflight and TTA parity gate,
commit and push this branch, then execute Run V through selection and final
evaluation. Only after Run V is complete may Run M start. Launch training with
`launch_training_tmux_v2.sh V` and then `launch_training_tmux_v2.sh M`; the
wrapper refuses an active training process or existing session, while each
launcher records output with `tee`. No launcher reuses an existing run root.
