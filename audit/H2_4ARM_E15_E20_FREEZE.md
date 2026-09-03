# H2 four-arm E15/E20 freeze

Status: `READY_FOR_TARGET_EVALUATION`

This manifest freezes the completed H/A/C/AC factorial in
`/tmp/h2_clean_factorial_e20_20260902_ampfix` before any target evaluation.
E15 is the primary horizon; E20 is the fixed secondary extension. The full
training checkpoints were produced from code SHA
`31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`, with shared E1 SHA
`7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`.

## Frozen checkpoint table

| Arm | Intervention | E15 primary | E20 secondary |
|---|---|---|---|
| H | none | `6830137b52fc16321192909fe7b3ca7565664afe61a8592fe7b7a78d2dff4771` | `a3857d018014b9ab77a06568e3c04e7fd8a4c0a7c24ccec3796e706c009dd087` |
| A | calibrated Anchor, lambda `0.0021633926715180626` | `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe` | `de3d160efc4cdc24454594274ca01c4d36724ecfa1ec75d2a660d4b6c037ff8c` |
| C | CIR, alpha `0.5`, peers `8`, radius `3` | `3530d7cb9f912c37de07fb984beca927460237c222f4f3a09a5c1cad2e035b71` | `80be0b3c3e43f1afd2003cc1a7fb4193f24eb5a980da32d338523062d882451f` |
| AC | calibrated Anchor + CIR | `2364984ad94faf3ab70833f6d0ad1e544daeebaf6e5ab0349a348b11322d608d` | `b0b1b20e937577d006ad1910f0b93b099d7aff894d19936787fe634ff10dd934` |

The machine-readable file contains the complete paths, global steps, arm
configuration digests, and the shared E1/checkpoint SHA256 values.

## Identity and integrity

- Common dataset/model contract: VisA, ViT-L-14-336, image size 518,
  three groups, batch size 6, seed 0, AMP, deterministic algorithms, and
  training horizon 20.
- Common code, CLIP, dataset-manifest, base-H2, and CIR-reference provenance
  matches across all selected checkpoints.
- The arm-specific configuration digests are intentionally different only for
  the preregistered intervention flags: H none, A Anchor, C CIR, AC Anchor+CIR.
- All selected model and optimizer tensors are finite. Scheduler, scaler,
  CPU/CUDA/Python/NumPy/dataloader RNG state fields are present and finite.
- E15/E20 log skip counts are recorded in the JSON manifest. H-E15 has one
  isolated `non_finite_grad` skip; all loss skip counts are zero and there is
  no repeated skip pattern.

## Recovery provenance

The first AC attempt hit CUDA OOM during epoch-4 backward after H, A, and C
had completed. That attempt is not used as a selected checkpoint source. AC
was resumed from the preserved `AC/adapter_3.pth` in the same run root and
completed E20 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This
was a runtime allocator mitigation only; the scientific configuration and
selected AC identity digest remained unchanged.

## Evaluation firewall

No Medical or MVTec evaluation is authorized until this manifest is committed
and pushed and local/remote Git state is reverified. The locked evaluation
contract is `benchmark_exact`, `pixel_stride=1`, raw exact macro aggregation,
E15 primary and E20 secondary, no checkpoint selection, no tuning, and no
intermediate target-based decisions. Medical evaluation must precede any
protocol-matched MVTec confirmation.
