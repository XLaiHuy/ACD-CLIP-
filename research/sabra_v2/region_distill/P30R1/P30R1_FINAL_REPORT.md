# P30R1 Scientific Stage 2 Final Report

Primary decision: `STAGE2_SCIENTIFIC_STOP`.

Exactly one preregistered candle attempt was executed. Stage 3 and the full 12-class run were not started.

## Attempt and frozen execution

- attempt UUID: `788c969e-3df9-418d-b659-550641d0cb69`
- UTC start: `2026-08-26T17:36:11.194972+00:00`
- branch: `research/p29r1-fast-objective-forensic-v1`
- scientific execution commit: `6f636368a2fdc4a72ccab62366dd05a74724bdb8`
- engineering qualification commit: `b59fb225a1a794ea83687078f9d0826ad28416f1`
- preregistration SHA-256: `ceff5944e3602f8b640e7ace02a5b99962244f71eeefe6cb4882c4cf3af92a43`
- class: `candle`; fit `1962`; held `200`; expected steps `39240`
- objective, optimizer, cache, architecture, and inference path remained frozen; no tuning or rerun occurred.

## P30R1 vs P30 vs P29

| metric | P29 | P30 | P30R1 | frozen gate |
|---|---:|---:|---:|---|
| pAP | 0.490503231199 | 0.144618064179 | 0.511513734224 | >= 0.4641403049313743 |
| pAUROC | 0.970040297208 | 0.972904419344 | 0.980534708954 | >= 0.9306671435137679 |
| directional cosine | 0.708549173789 | 0.736923573997 | -0.070148224730 | >= 0.6985491737886378 |
| sign agreement | 0.565493827160 | 0.569567901235 | 0.119691358025 | >= 0.5554938271604938 |
| Pearson alignment | 0.769957203646 | 0.378208786002 | 0.349901146198 | descriptive |
| Spearman alignment | 0.717418156345 | 0.714233750011 | 0.054565093964 | descriptive |
| mean absolute residual | 2.036682758542 | 1.542490325771 | 0.178743865245 | descriptive |
| residual absolute q99 | 4.321676936150 | 25.929798946381 | 4.528306531906 | <= 8.643353872299194 |
| normal-score q99 shift | 0.000001158785 | 0.998690783978 | 0.000007328280 | <= 0.0010011587851122385 |

## Mechanism result

- directional behavior retained under the preregistered threshold: `False` (cosine `-0.070148224730`, sign `0.119691358025`).
- radial q99 and normal-score saturation controlled under the preregistered thresholds: `True` (residual q99 `4.528306531906`, normal q99 shift `0.000007328280`).
- pAP gate recovered: `True`; pAUROC gate safe: `True`.
These are results of this one candle test only; they do not authorize a new objective, tuning, or Stage 3.

## Training and data audit

- optimizer steps: `39240`; finite loss/gradients: `True`/`True`; nonfinite counts: `0`/`0`.
- student parameter delta L2: `47.326017160399104`; teacher parameter delta: exactly `0.0`; teacher scale detached: `True`.
- new CLIP forwards: `0`; new Phase2B forwards: `0`; cache rebuild: `False`.
- held GT/mask reads before prediction freeze: `0`/`0`; post-freeze mask reads for scoring: `100`.
- prediction freeze: `True`; prediction SHA-256: `30c250b52ff980e7b16fa0e97ffaebb19e13fd5521722e5397cc86ed2c4e1218`.

## Runtime audit

- training wall time: `1215.836871` seconds (parent process `1219.987683`); median step `0.004262911797` seconds; mean `0.004616965130`; p90 `0.004973567963`.
- frozen limits: median <= `0.007889637351036071` seconds and wall <= `1447.5493008233607` seconds.
- inference overhead: `0%`; deployment and scoring path were unchanged.

## Gate and final audit

- Stage 2 gate: `STAGE2_SCIENTIFIC_STOP`; failed checks: `['directional_cosine', 'sign_agreement']`.
- post-run audit: `PASS`; attempt count: `1`; Stage 3 started: `False`; full run started: `False`.
- all scoring occurred after the immutable GT-free prediction freeze; no held result was inspected during training.

`STAGE2_SCIENTIFIC_STOP`
