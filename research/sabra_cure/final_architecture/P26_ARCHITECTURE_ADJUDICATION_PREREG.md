# P26 Architecture Adjudication Preregistration

Status: `FROZEN_BEFORE_ADJUDICATION`

Parent terminal: `c8f505aa69b581afffead83db9b146df53179ce4`

P26 is a read-only evidence adjudication and deployable-architecture freeze. It
creates no scientific attempt, fits no model, generates no prediction, and
does not access MVTec or Medical. It performs no CLIP forward and no Phase2B
optimization step.

## Evidence hierarchy

Claims are accepted in this order: valid preregistered source science,
post-hoc diagnostic evidence, arithmetic derivation, then engineering
evidence. Oracle and post-hoc findings can motivate a component but cannot
make it deployable. Engineering-stop studies cannot support scientific claims.

The adjudication will verify the terminal artifacts and commit identities for
the valid SABRA/CURE lineage, including R0, R1, R2, R2-v2, P13, P20, P23,
P24, and P25R3. P25R3 is the terminal parent and has priority over the earlier
P24 recommendation wherever P25R3 directly answers the subsequently tested
patch-benefit-identifiability question.

## Component decision rule

Each component is classified `RETAIN_DEPLOYABLE`, `RETAIN_DISABLED_REFERENCE`,
`DROP`, or `UNKNOWN_RESEARCH_ONLY`.

A learned or corrective component is `RETAIN_DEPLOYABLE` only if persisted,
valid, leakage-safe source evidence establishes its complete GT-free inference
path and its frozen scientific gate. Broad improvement, oracle headroom, or
diagnostic association alone is insufficient. Any component that lacks such
evidence is excluded from the deployable path. When no correction controller
meets this rule, the deterministic final policy is the native frozen detector
with `KEEP` as the only reachable action; this is the preregistered conservative
fallback, not an outcome-selected hyperparameter.

The base detector remains eligible only if its exact checkpoint, backbone,
configuration, preprocessing, aggregation, and postprocessing identities can
be verified. Missing runtime artifacts affect handoff readiness separately and
do not silently substitute an alternative architecture.

## Frozen adjudication questions

1. Does signed direction have valid source evidence, and does it have a valid
   deployable benefit-selection path?
2. Does the harm-risk head establish safety alone or a complete beneficial
   intervention policy?
3. Do image-level value, coarse budget, or patch-level benefit heads meet their
   frozen gates?
4. Is any corrective action justified as the final external-validation path?
5. If not, is native-only frozen Phase2B fully specified and portable?

## Required architecture completeness

The final source of truth must specify without open scientific choices: base
checkpoint and CLIP asset identities; backbone and stages; image preprocessing;
dtype; deterministic seed; logits, blur, interpolation, stage aggregation, and
softmax order; retained and dropped heads; action and fallback semantics;
alpha and thresholds even when a disabled historical actuator is recorded;
coverage and abstention; forbidden adaptation; and the external-validation
lock. Every runtime parameter must also appear in machine-readable JSON.

## Reproducibility and portability rules

All required artifacts will be checked for existence, size, SHA256, Git/LFS
status, origin, and reconstructability. Datasets are never artifacts in the
portable bundle. Missing required external artifacts produce
`P26_HANDOFF_INCOMPLETE`; they do not change the scientific freeze. The final
entry point must load the canonical config, expose `--check-only`, `--dry-run`,
and reserved `--run`, and reject any external run unless separately authorized.

The restore script must verify repository/branch/revision, clean state, config
hash, manifests, required artifact hashes, and `--check-only`. It may not
download data or substitute checkpoints.

## Publication and firewall

P26 uses exactly three commits: this adjudication freeze, the architecture
freeze, and the portable handoff. Explicit paths only are staged. No force
push, history rewrite, or scientific attempt marker is allowed.

Frozen counters: MVTec reads `0`; Medical reads `0`; additional CLIP forwards
`0`; Phase2B training steps `0`; new scientific fits `0`; P26 scientific
attempt markers `0`. External validation remains `AUTHORIZED=FALSE`.
