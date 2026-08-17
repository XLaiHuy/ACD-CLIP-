# P5-E0R Frozen HRIP Evaluation Recovery — Design Review

## Scope

P5-E0R is an evaluator-only recovery of the already-frozen P5-E0 GT-free
evidence. It authorizes zero model forwards, zero training, no medical work,
and no HRIP recomputation. The original P5-E0 result remains historically
`P5E0_HRIP_AUDIT_INVALID` and its files and tools are read-only.

The only authorized semantic correction is to apply the authoritative
`tools/audit_phase5_hsir.py::shifted_map` independently to each frozen
518x518 HRIP image map before concatenating images within a class. The old
evaluator incorrectly applied the same helper to a class-concatenated array.

## Adversarial review and resolutions

- Frozen-cache corruption or manifest mismatch: validate all 2,162 record
  SHA256 values, record metadata, schema, aggregate manifest hash, and
  canonical-order hash before implementation and again at evaluation.
- Accidental evidence recomputation or model inference: the new evaluator
  imports no model loader and consumes only frozen records plus post-freeze GT;
  source/runtime guards reject model-forward markers and the test suite uses
  only synthetic arrays before GT evaluation.
- Class-boundary contamination: the shift helper receives exactly one
  flattened 518x518 map per call; class concatenation occurs only afterward.
- Image, label, and pixel-ID order drift: frozen manifest order is the sole
  record order; GT is loaded by the same class-local canonical order; pixel IDs
  are derived from the frozen canonical image index and local pixel index.
- Wrong spatial domain: the shift is on the finalized 518x518 evidence map,
  never on the 37x37 patch grid.
- GT or label mutation: only HRIP_SHIFT is transformed. HRIP, E_nonlocal,
  score, final margin, D_rank, labels, pixel IDs, matching, and risk masks are
  preserved byte/numerically from their frozen construction.
- Shifted rematching: deterministic matches are computed once from score,
  D_rank, labels, pixel IDs, and the frozen top-20% risk population, then
  reused for all three evidence signals.
- Risk/triage drift: risk is frozen at ceil(20%) by D_rank and triage at
  ceil(10%) by evidence, using the authoritative stable selectors.
- Bootstrap drift: class bootstrap uses 2,000 repetitions and the frozen
  role-specific seeds 5101 through 5107; paired deltas resample the same class
  indices for both members.
- Historical-source modification: no file under the original E0 output
  directory and no original E0 tool is modified; E0R has a new namespace and
  new evaluator/test files.
- Result-driven changes: no alternate shift, candidate, threshold, seed,
  metric, risk fraction, triage fraction, or rescue path is available.
- GT timing: no GT performance metric is read or generated before the E0R
  protocol and implementation commits are remotely available.

## Recovery invariants

`MODEL_FORWARDS=0`, `TRAINING_STEPS=0`, `MEDICAL=false`,
`EVIDENCE_RECOMPUTED=false`, and `CANDIDATE=NONE` are immutable. P5-E0R can
recover only the preregistered evaluation of frozen evidence; it cannot relabel
or rewrite the historical P5-E0 result.
