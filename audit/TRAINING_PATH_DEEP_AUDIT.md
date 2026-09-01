# Training path deep audit

## Baseline contract

The recovered H2 run is the model-only checkpoint
`ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb`, from
source commit `e03966997d4cecfd985943a4053a93e1e40197ec`. The clean path keeps
the recovered model geometry, optimizer family, learning rates, AMP, DFG
settings, hybrid prompt schedule, losses, data transforms, and medical image
score blend unchanged.

The historical entry path leaves the model in evaluation mode while updating
trainable adapters. This matters because the adapter stack contains BatchNorm
buffers. The clean trainer explicitly calls `model.eval()` at every epoch and
does not introduce a `model.train()` transition.

## Changes audited

1. `h2_clean.contract` centralizes hashes, environment/RNG capture, explicit
   worker/DataLoader seeding, full-state checkpoint construction/restoration,
   the safe image anchor, and the detached CIR contract.
2. The trainer preserves the historical loss path and adds optional A/C terms
   behind explicit flags. Global step increments only after a successful
   scaler update; skipped non-finite gradients are not counted as updates.
3. Class traversal is sorted, removing process/hash-seed-dependent dictionary
   insertion order without changing the set of classes.
4. Every clean checkpoint retains historical `image_adapter`, `text_adapter`,
   `epoch`, and DFG/prompt metadata aliases, and adds optimizer, scheduler,
   scaler, RNG, DataLoader generator, config/hash, environment, and anchor
   reference state. The resumable contract stores a parameter-only image
   reference separately from the state-dict alias, so BatchNorm buffers cannot
   enter the anchor identity set.
5. Safe anchor loss is one global numerator over image-adapter parameters and
   one global reference denominator plus epsilon. References are detached CPU
   clones. Zero-reference tensors cannot dominate through per-tensor
   normalization.
6. CIR is train-only and inserted after native segmentation logits are formed.
   Peer selection is GT-free, deterministic, shared across groups, detached,
   robustified by midpoint median/MAD and `tanh`, and invalid sparse-peer
   queries receive zero shift. Alpha zero returns the native tensor itself.
7. The evaluator has explicit `legacy_h2_replay` and `benchmark_exact` modes.
   The default legacy mode preserves stride and per-class rounding; benchmark
   mode forces stride 1, no threshold binning, and raw metric output.

## Evidence

- Contract, anchor, CIR, K-reg, evaluator, model-path alpha-zero, external
  exact-metric, and resume tests pass (`15 passed`).
- Historical H2 legacy replay reproduces the published per-class E10 values
  and macro 90.98/40.35 after historical rounding.
- The exact benchmark replay and bounded candidate Brain replay are recorded
  in `H2_ORACLE_EVALUATOR_PARITY.csv`; they are not used to select an arm.
- The bounded shared-E1 and H/A/C/AC smoke completed with finite losses,
  finite safe-anchor ratios, and detached CIR deltas; see
  `H2_CLEAN_SMOKE_RESULTS.md`.
- The source manifest audit has zero missing files or duplicate paths. The
  96-image historical source gate is explicitly marked non-generalizing.

## Boundaries

The historical model-only H2 E10 checkpoint is replay-only. It cannot support
exact optimizer/RNG continuation. The clean factorial therefore starts one
shared seeded E1 run and branches all arms from its full-state E1 checkpoint.
No E15 arm checkpoint or medical result is claimed until the bounded smoke
gates pass and a user-authorized full run is launched.
