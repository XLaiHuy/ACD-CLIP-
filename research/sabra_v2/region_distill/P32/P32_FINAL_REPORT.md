# P32 Scientific Stage 2 Final Report

Final status: P32_STAGE2_SCIENTIFIC_STOP.

Exactly one preregistered candle attempt was executed. No Stage 3 or full 12-class run was started, and no rerun or tuning occurred.

## Frozen execution

- attempt UUID: 1818b1ef-9afc-41ad-972c-ee5288ee1286
- branch: research/p29r1-fast-objective-forensic-v1
- scientific execution commit: 383665d104891caeeacf9a131b6d795e3d22eec8
- preregistration SHA-256: 5141722b2c3e3d3aac721390a8943d54356dd17bdfdad8aaa6bd7302766a5cc2
- class: candle; fit 1962; held 200; optimizer steps 39240
- objective: P32_FUNCTIONAL_MARGIN_EFFECT_SMOOTHL1_V1; objective count 1; seed 0; FP32 AdamW schedule remained frozen.

## Locked endpoint comparison

| metric | P31 native / zero adapter | P30R1 | P32 | P32 minus P31 |
|---|---:|---:|---:|---:|
| pAP | 0.514140304931 | 0.511513734224 | 0.510351502947 | -0.003788801984 |
| pAUROC | 0.980667143514 | 0.980534708954 | 0.971460700418 | -0.009206443096 |

- pAP gate: False; pAUROC gate: False.
- P32 minus P30R1: pAP -0.001162231277, pAUROC -0.009074008536.

## Mechanism and safety diagnostics

- residual absolute q99: 4.963814287186; frozen maximum 8.643353872299; pass True.
- normal score q99 shift: 0.000005382090; frozen maximum 0.001001158785; pass True.
- residual exact nonzero fraction: 1.000000000000; native-to-P32 score-effect q99 absolute: 0.000009602303.
- raw direction metrics were descriptive only and were not used as gates; no held-derived tuning or new teacher forward occurred.

## Runtime and data audit

- training wall time: 1206.031317 seconds; parent process 1209.396516 seconds; median measured step 0.005654 seconds.
- peak GPU allocated/reserved: 46519808 / 75497472 bytes; inference overhead: 0%.
- finite loss/gradient: True/True; nonfinite counts: 0/0.
- pre-freeze held GT/mask reads: 0/0; post-freeze held GT/mask reads: 200/100.
- prediction freeze: True; prediction SHA-256: e5573828a3205dd18955e3e0551a8297a75b7a07d24d2b35d49d165002f596f0.
- new CLIP/Phase2B/teacher forwards: 0/0/0; cache rebuilds: False.

## Terminal audit

- gate: P32_STAGE2_SCIENTIFIC_STOP; failed checks: ['pAP', 'pAUROC'].
- post-run audit: PASS; attempt count: 1; Stage 3 started: False; full run started: False.
- the authoritative P32 preregistration was not edited after the attempt identity was created.

P32_STAGE2_SCIENTIFIC_STOP
