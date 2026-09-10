# R3 E15 Source Confirmation

Status: `PASS`

This is the full S2 confirmation of the source-selected locked candidate. The
selection was made on the preregistered VisA category-held-out source slice;
target labels and target metrics were not used for selection or tuning.

## Locked protocol

- Source dataset: VisA full training split, with `candle`, `macaroni1`,
  `pcb3`, and `pipe_fryum` used as the held-out evaluation slice.
- Parent: canonical shared E1, SHA256
  `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`.
- Hybrid alpha: `0.20`, selected by the source alpha gate.
- Anchor family budget rho: `0.10`, selected by the source rho gate.
- Anchor lambda: `0.0021633926715180626`.
- CIR: disabled.
- TTA-F: identity, horizontal flip, vertical flip, and horizontal-plus-vertical
  flip, inverse-mapped and averaged.
- Full uncapped E15 training, batch size 6, seed 0, deterministic algorithms.

## Source replay

| Policy | Pixel AP | Pixel AUROC | Image AP | Image AUROC |
| --- | ---: | ---: | ---: | ---: |
| TTA-I | 51.88353392 | 97.15568791 | 96.11829966 | 95.75396329 |
| TTA-H | 52.52750213 | 97.26577750 | 96.25955522 | 95.84242254 |
| TTA-F | 54.58400149 | 97.34997273 | 96.66284025 | 96.22074366 |

The locked TTA-F source Pixel AP is `54.584001494597956`. The exact raw replay
is preserved in [E15_SOURCE_TTA.json](E15_SOURCE_TTA.json).

## Checkpoint and audit

- Checkpoint: [adapter_15.pth](e15_confirmation/adapter_15.pth)
- Checkpoint SHA256:
  `c11a73a8c3b20257c0a0f7cfe1328939e7a72f988a3fe3d6644062e167ee9f77`
- Training code SHA: `4126a5cbdc7f10acde64a9b432eb41f358e8e59c`.
- Final anchor family partition: complete.
- Final global effective ratio: `1.4887726792835684e-05`.
- Final maximum active-family effective ratio: `0.08280756825020887`.
- Final epoch non-finite loss and gradient skips: zero.

The initial attempt encountered a CUDA allocator OOM at epoch 4 batch 81.
The run was resumed from the valid epoch 3 checkpoint with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the recovery setting is
recorded in commit `fdedb73d10ae35a143deaf22ac921d43972b77f5`.

## Decision boundary

This phase passes its source-only confirmation gate. The next permitted step
is a single locked target audit of this checkpoint. No target-guided tuning is
authorized by this artifact.
