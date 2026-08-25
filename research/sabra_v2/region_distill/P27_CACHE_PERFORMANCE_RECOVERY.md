# P27 Cache Performance Recovery

Status: `READY`

The recovery caches exactly four invariant FP32 tensors for each of the 11
source classes in a LOCO fold: `seg_features`, `native_logits`, the source-only
teacher region target, and the unchanged source localization mask. Files are
fixed-shape raw tensors accessed through memory mapping. A completion manifest
records the held class, ordered source inventory, P26/CLIP/config/protocol
hashes, dataset root, tensor shapes/dtype/byte counts, and SHA-256 for every
tensor file. Missing, incomplete, corrupt, wrong-fold, wrong-asset, and
wrong-inventory caches are rejected.

The focused TDD suite covers exact cached inputs, student forward, all losses,
gradients, an optimizer step, multiple steps, checkpoint reload, source-only
construction, zero held GT/mask reads, and all required rejection cases. On
the actual RTX 3070 frozen path, inputs, student outputs, and losses were
bit-exact. The maximum CUDA gradient difference was `7.450580596923828e-09`
and maximum post-AdamW parameter difference was `1.1641532182693481e-10`, both
inside the independently fixed `1e-7` tolerance. A tolerance is necessary
because Torch 2.5.1 reports that the frozen `adaptive_average_pool2d` CUDA
backward has no deterministic implementation; the scientific operator was not
replaced.

Measured engineering benchmark (three real VisA source samples, same GPU):

- cache build: `0.3766816799 s/sample`
- exact cache size: `13,723,180 bytes/sample`
- estimated fold cache: `26.91–27.61 GB`
- uncached frozen forward: `0.2322476360 s/sample`
- uncached full step: `0.3037630981 s/sample`
- cold memory-mapped read/copy: `0.0038820941 s/sample`
- cached training step: `0.0113147900 s/sample`
- measured full-step speedup: `26.8466x`
- projected cache-build plus 20-epoch training for all 12 folds: `14,340.01 s`
  (`3.98 h`), excluding prediction, scoring, integrity scans, and orchestration

Measured and projected values are recorded separately in
`P27_CACHE_BENCHMARK.json`. Caches are built one fold at a time and removed
only after that fold's prediction is frozen and scored, keeping peak storage
practical. No frozen scientific setting or protocol file changed.
