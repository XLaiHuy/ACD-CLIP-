# Corrected CIR-V2 go/no-go decision

## Decision

PHASE2B_REPRESENTATION_PRESERVATION

This is a diagnosis decision, not authorization to implement a new mechanism in this commit.

## Why

1. The scheduler mismatch is resolved and no longer confounds the P/C0/C05 comparison.
2. Corrected C0 is not close to P on Medical Pixel AUROC/AP, despite competitive or stronger source AP and stronger image AUROC.
3. C05−C0 is effectively zero, so the current inference transport does not supply a reliable recovery or gain.
4. The residual pattern is target transfer and Pixel AP, not a demonstrated direct cls/seg gradient conflict.
5. Peer invariants pass, but near-zero MAD and heavy tanh saturation make the current transport reliability questionable.
6. The train/deploy operator mismatch is real and must remain a gated diagnostic variable, not an unverified fix.

## What this does not mean

The conclusion is not “RMT failed.” The pre-fix benchmark had a confirmed CIR scheduler bug, and the corrected run shows that inference C05 is neutral while CIR training changes target transfer. The clean statement is: current corrected CIR training does not preserve the Phase2B target-transfer behavior, and the current RMT inference transport has no measurable benefit in this matrix.

## Action gate

No additional training or architecture change is launched here. The previously required matched Phase2B-vs-CIR corrective retrain is complete. If the research continues, the next separately authorized experiment is one source-only representation-preservation variable with native Phase2B deployment; operator consistency and any RMT redesign remain separate experiments.

## Stop conditions

- Do not select alpha or thresholds from Medical results.
- Do not use post-hoc target GT contamination or stage labels as hyperparameter rules.
- Do not combine optimizer, loss, deployment, stage-fusion, and RMT changes.
- Do not claim formal equivalence from the descriptive matrix without paired uncertainty analysis.
