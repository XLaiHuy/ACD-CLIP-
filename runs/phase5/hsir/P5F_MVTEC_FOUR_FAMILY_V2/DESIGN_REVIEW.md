# P5-F v2 adversarial design review

This study is an isolated, industrial-only, evidence-only development study
on MVTec AD. Historical E0, E0R, E0RC, B1, and HRIP conclusions remain
immutable. The four families are pure deterministic transforms of common
GT-free B1 geometry; they do not train, correct, fuse, or alter Phase2B.

## Hard barriers

- Setup must be `DATASET_READY=PASS` before implementation or inference.
- The official loader receives only canonical index, class name, and image path.
  It opens RGB images only. It never instantiates the repository dataset
  classes, reads `label`, opens `mask_path`, or traverses `ground_truth` before
  the GT-free manifest is finalized.
- One common model forward is authorized per canonical MVTec test identity
  (1,725 identities); no duplicate, preliminary, or result-driven pass exists.
- All 26 configurations are constructed from the same frozen common cache
  before GT release. GT is evaluation-only and cannot affect geometry,
  evidence, selection, or retry.

## Threats challenged and resolutions

GT leakage, target/occupancy leakage, label access, mask access, ground-truth
path access, class-count reconstruction from labels, selector drift,
deterministic-order drift, patch-index drift, preprocessing drift, checkpoint
or config substitution, text-feature drift, stage mismatch, normalization
mismatch, percentile mismatch, D_rank mismatch, per-stage tuning, hidden
temperature/radius/K/threshold tuning, B1 centroid mismatch, invalid-reference
reinterpretation, shifted-control reselection, post-hoc primary replacement,
hidden candidate/threshold search, duplicate forwards, giant feature caches,
unsafe resume, sibling-family contamination, and protected-source edits are
all explicit G0 subchecks. A failure is terminal; there is no rescue path.

The common cache persists only compact native outputs, B1 peer indices,
validity, query-peer cosine geometry, peer Gram upper triangles, B1 centroid
evidence, and D_rank. It never persists 768-D stage features, images, masks,
labels, or target occupancy.

Family modules are pure: they accept only `c`, `G`, `valid_reference`, and a
frozen config, return finite arrays/diagnostics, and have no filesystem,
dataset, model, subprocess, network, or sibling-family access. Search spaces,
complexity ranks, folds, seeds, fixed budgets, gates, and ranking rules are
frozen before any GT metric is read.

## Deliberate limitations

MVTec is development-selection evidence, not external generalization proof.
Research-value scoring is separate from empirical scientific ranking and can
never rescue a failed family. The final selected MVTec configuration, if any,
is frozen for a future untouched industrial dataset and is not retested on
MVTec as unbiased validation.
