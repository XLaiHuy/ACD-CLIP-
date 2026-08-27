# P33 Scientific Stage 2 Final Report

Final status: P33_STAGE2_SCIENTIFIC_STOP.

Exactly one preregistered candle attempt was executed. No rerun, tuning, Stage 3, subset expansion, or full run occurred.

## Frozen execution

- attempt UUID: 1a29ea4a-8952-4eeb-b75b-c574f6769261
- branch: research/p29r1-fast-objective-forensic-v1
- scientific execution commit: e4f73206cd2bb7ca10201fb944163e40d55b4a44
- engineering qualification commit: eef6877c692fcb0102c7fcab686d68c4a28f39f4
- preregistration SHA-256: d2460555be14af7d23316e43ad16c8585faeecbedf1698ee71f29dce765aed6c
- class/split: candle; fit 1962; held 200; optimizer steps 39240
- objective: P33_CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER_V1; objective count 1; seed 0; FP32 AdamW schedule remained frozen.

## Locked endpoint comparison

| metric | P31/native | P30R1 | P32 | P33 | P33 minus native | P33 minus P32 |
|---|---:|---:|---:|---:|---:|---:|
| pAP | 0.514140304931 | 0.511513734224 | 0.510351502947 | 0.519395095936 | +0.005254791005 | +0.009043592989 |
| pAUROC | 0.980667143514 | 0.980534708954 | 0.971460700418 | 0.978184288830 | -0.002482854684 | +0.006723588412 |

- P33 pAP gate: True; pAUROC gate: False.
- P31/native, P30R1, and P32 values are frozen historical comparators; no historical result was used to alter P33 gates.

## Selectivity and actionability result

- inherited-threshold residual support: P30R1 0.111358024691; P33 0.999074074074; P32 reference was 0.871481481481.
- P33/P30R1 support Jaccard: 0.110432098765; P33 containment of P30R1: 0.9916851441241685.
- P33 residual effective support fraction: 0.962760408648; Gini: 0.069176234345; top-10% mass: 0.112053243823.
- source-only actionability weights: mean 0.5151192263753267; median 0.5015398561954498; q90 1.0; q95 1.0; q99 1.0; exact-zero fraction 0.20813423814114543; range [0.0, 1.0].
- candidate score-effect q99 abs: 0.000012143285; normal-score q99 shift: 0.000006601931.
- The mechanism is selective transfer through a bounded training-only weight; raw teacher-vector fidelity was not an objective or gate.

## Mechanism answers

- Did P33 reduce intervention density relative to P32? no under the inherited residual-support diagnostic.
- Did it retain P30R1 support? yes descriptively.
- Did pAP improve relative to P32? yes.
- Did pAUROC recover toward or exceed native? yes toward native; did not exceed native.
- Did radial/tail and normal-score behavior remain safe? yes under the preregistered gates.
- Did actionability help without raw direction fidelity? This is interpreted only through the locked detection endpoints and descriptive selectivity diagnostics; no direction metric was optimized.

## Runtime and audit

- training wall time: 1101.238828 seconds; parent process time: 1106.479829 seconds; median step 0.005722 seconds; prediction time 6.308113 seconds; scoring time 32.456340 seconds.
- peak GPU allocated/reserved during training: 46520320 / 75497472 bytes.
- finite loss/gradient: True/True; nonfinite counts: 0/0.
- student parameter delta L2: 51.91565682149541; teacher/frozen parameter delta: 0.0.
- prediction frozen before scoring: True; prediction SHA-256: 91d6400823afbad810f880fc22fa37a74ac42dd2191e3f9a51e4b57e2403cd8a.
- held GT/mask reads before freeze: 0/0; after freeze: 200/100.
- new CLIP/Phase2B/teacher forwards: 0/0/0; cache rebuilds: False; reruns: False.

## Terminal audit

- scientific gate: P33_STAGE2_SCIENTIFIC_STOP; failed checks: ['pAUROC', 'automatic_rerun'].
- post-run audit: PASS; attempt count: 1; Stage 3 started: False; full run started: False.
- authoritative P33 preregistration was not edited after attempt identity creation.

P33_STAGE2_SCIENTIFIC_STOP
