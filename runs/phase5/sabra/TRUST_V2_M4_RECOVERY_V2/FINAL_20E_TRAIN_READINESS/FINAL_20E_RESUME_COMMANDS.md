# SABRA Recovery-v2 resume record

This is a persistence and review record. It is not permission to start
training.

## Verify the persisted boundary

```bash
cd /workspace/ACD-CLIP-
git switch research/p5-sabra-g
git fetch origin research/p5-sabra-g
git rev-parse HEAD
git rev-parse origin/research/p5-sabra-g
git rev-list --left-right --count HEAD...origin/research/p5-sabra-g
git lfs pull origin research/p5-sabra-g
git lfs fsck
sha256sum -c runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2/FINAL_20E_TRAIN_READINESS/FINAL_20E_SHA256SUMS.txt
```

The final handoff directory contains the authoritative study resume notes;
the package manifest contains the exact source, cache, checkpoint, CLIP,
protocol, result, and invalidation paths.

## Authorization guard

```text
FULL_20E_TRAIN_AUTHORIZED=false
EXPLORATORY_20E=false
medical_access=false
MVTec_external_status=MVTEC_EXTERNAL_UNAVAILABLE
```

Do not start a 20-epoch run, access medical data, or claim MVTec validation
from this package. A future exploratory run must first be explicitly labelled
`EXPLORATORY_20E=true` and must not be reported as an authorized continuation.

## Scientific resume boundary

The frozen Recovery-v2 candidate is `M1_E_Credibility`; its M4 diagnostic is
the exact selected non-PCRR feature order followed by `D_rel`. PCRR is
diagnostic-only and was dropped. The VisA candidate and freeze provenance are
already committed. MVTec remains unavailable because the required image root
was absent; no external metric is present to authorize the next full-training
study.
