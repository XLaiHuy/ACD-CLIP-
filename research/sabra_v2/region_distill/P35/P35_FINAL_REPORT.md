# P35 Scientific Stage 2 Final Report

Final status: `P35_STAGE2_SCIENTIFIC_STOP`.

Exactly one preregistered candle Stage 2 attempt was executed. No rerun, tuning, Stage 3, subset expansion, or full run occurred.

## Attempt

- UUID: `5e1526d7-fa89-40f9-9976-4e5bb196ee79`
- execution commit: `ea11d937c4394ef754c2b8f31c3acf7e601af0ac`
- engineering qualification commit: `ea11d937c4394ef754c2b8f31c3acf7e601af0ac`
- preregistration SHA-256: `d92a8144e071412608292b4c48f5fe69381f82c3b205f6990266f2383336e3d8`
- class/split: candle LOCO; fit `1962`; held `200`
- optimizer: AdamW; seed `0`; FP32; steps `39240/39240`

## Detection

| Method | pAP | pAUROC |
|---|---:|---:|
| P31/native | 0.514140304931 | 0.980667143514 |
| P30R1 | 0.511513734224 | 0.980534708954 |
| P32 | 0.510351502947 | 0.971460700418 |
| P33 | 0.519395095936 | 0.978184288830 |
| P35 | 0.520252695765 | 0.977882517205 |

- P35 minus native: pAP `+0.006112390834`, pAUROC `-0.002784626309`.
- P35 minus P33: pAP `+0.000857599829`, pAUROC `-0.000301771625`.

## Safety and mechanism diagnostics

- residual absolute q99: `5.270559787750`; normal-score q99 shift: `0.000006801919`.
- nonfinite loss/gradient counts: `0/0`.
- inherited-threshold descriptive residual support: P30R1 `0.111358024691`, P32 `0.871481481481`, P33 `0.999074074074`, P35 `0.999074074074`; this is not a gate.
- residual concentration: effective support `0.960498448626`, Gini `0.075259043785`, top-10% mass `0.111815686963`; descriptive only.
- tanh actionability source weights: mean `0.402035740809`, q10/q25/q50/q75/q90/q95/q99 `0.000000000000/0.000170154901/0.463847421092/0.761594277078/0.761594317452/0.761594317452/0.761594357826`, exact-zero `0.208215406053`, >.25/.5/.75/.9 `0.547329259633/0.493234391095/0.445333413341/0.000000000000`, exact-one `0.000000000000`.
- full teacher target confirmed: `True`; P34 target shaping absent: `True`; tanh map confirmed: `stop_gradient(tanh(abs(E_t)/C))`.

## Locked gates

- pAP: value `0.5202526957653664`, requirement `0.5141403049313743`, pass `True`.
- pAUROC: value `0.9778825172046846`, requirement `0.9806671435137679`, pass `False`.
- global_residual_abs_q99: value `5.270559787750244`, requirement `8.643353872299194`, pass `True`.
- normal_score_effect_q99_shift: value `6.801918634664436e-06`, requirement `0.0010011587851122385`, pass `True`.
- nonfinite_loss_count: value `0`, requirement `0`, pass `True`.
- nonfinite_gradient_count: value `0`, requirement `0`, pass `True`.

## Audit

- prediction frozen before scoring: `True`; prediction SHA-256: `5c195012317c2724adfbc2a662ef731fbdeaccd433a277a42550d7999f055413`.
- held GT/mask reads before freeze: `0/0`; after freeze: `200/100`.
- new CLIP/Phase2B/teacher forwards: `0/0/0`; cache rebuilds: `False`.
- student parameter delta L2: `52.860010314673`; frozen/teacher delta: `0.0`.
- report-wrapper status: `PASS`; post-run audit: `PASS`.
- reruns: `0`; held tuning: `0`; Stage 3: `False`; full run: `False`.

## Runtime

- training wall time: `1056.541495` s; median/p90/mean step: `0.005721/0.006720/0.006002` s; prediction: `7.609042` s; scoring: `33.087171` s.
- peak GPU allocated/reserved: `46520320/75497472` bytes; peak RSS: `11343532` KiB.

## Counts

- P35 scientific Stage 2 attempts = 1
- Stage 3 = 0
- full runs = 0
- held tuning iterations = 0
- new CLIP forwards = 0
- new Phase2B forwards = 0
- teacher forwards = 0
- cache rebuilds = 0

P35_STAGE2_SCIENTIFIC_STOP
