# Next real CIR machine run

This repository is release-gated for the canonical CIR_DFG_RMT_V1 experiment.

CURRENT GATE STATE

- G0 Identity: PASS.
- G1 Math/unit: PASS; reference-vs-optimized maximum error is 1.43e-6.
- G2 real alpha=0 parity: BLOCKED / NOT RUN because the real CLIP asset,
  input, and checkpoint are unavailable.
- G3 source preflight: PARTIAL; synthetic-only PASS, real VisA preflight NOT RUN.
- G4 performance: PARTIAL; CPU micro-profile only; GPU latency/VRAM NOT RUN.
- G5 train smoke: BLOCKED / NOT RUN because the real source dataset and CLIP
  asset are unavailable.
- ALPHA_STATUS: PROVISIONAL.
- RELEASE_LOCK: FALSE.

FULL TRAIN AUTHORIZED: NO.

When the real assets are available, run the gates in this exact order:

1. Run real G2 alpha=0 parity.
2. Run real VisA G3 source preflight.
3. Run GPU G4 latency/VRAM profile.
4. Rerun G1, G2, and G3 if the optimized implementation changed.
5. Run real G5 train smoke.
6. Generate `runs/cir_rmt/CIR_DFG_RMT_V1/release_lock.json` only after G0,
   G1, G2_REAL, G3_REAL, G4_GPU, and G5_REAL are all PASS with matching
   identity/evidence. The alpha must be FROZEN. No environment variable can
   bypass the lock.
7. Run the full runner.

The exact full-run commands are:

```bash
bash scripts/cir_rmt/run_full_cir_v1.sh --source visa --device 0 --seed 0
bash scripts/cir_rmt/run_full_cir_v1.sh --source mvtec --device 0 --seed 0
bash scripts/cir_rmt/run_full_cir_v1.sh --source both --device 0 --seed 0
```

Each source pipeline trains exactly 20 epochs, verifies checkpoints 12, 14,
16, 18, and 20, evaluates every checkpoint on the opposite industrial source
and ColonDB, ClinicDB, Kvasir, BrainMRI, Liver CT, and Retina OCT, and reports
every epoch-by-dataset result. There is no best-epoch replacement.

Do not reuse old Phase/SABRA result directories, checkpoints, or metrics.
This document does not authorize architecture research or a full run before
the real release gates pass.
