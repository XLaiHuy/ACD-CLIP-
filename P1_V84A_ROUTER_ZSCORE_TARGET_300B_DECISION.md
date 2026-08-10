# P1-v8.4-A Router z-score target: one 300-batch decision

Source state at launch: `8abd135be16a4cf8576924c4782fd5fb0f7e9ab9`.

## Frozen target contract

The only new Router target was `patch_zscore_softmax`:

`q = softmax((gain - mean(gain)) / clamp(std(gain), 1e-12))`.

The one frozen audit reconstructed the fingerprinted baseline exactly: input
manifest SHA256 `20575581d04b90b1221130d9870e5a6b50466d942c88eeb8bf18c39c474cc25b`,
support 1,071,261 overall / 1,066,751 normal / 4,510 anomaly, and anomaly
winners F2=2,215, F3=1,895, F4=400. q was finite, preserved raw-gain argmax,
and had median normalized entropy 0.60235 overall and 0.79537 anomaly.

Fresh no-step calibration selected `lambda_router=0.00044262806523447237`.
Its router weighted gradient ratios were median/p95/max
0.43464/0.50000/0.53574; model hash was unchanged and optimizer/scheduler
steps were zero. The ACT and factor controls remained fixed.

## One fresh 300-batch result

The fresh OpenAI-CLIP seed-0 VisA/train run completed exactly 300
microbatches and 50 optimizer steps in 546.80 seconds. It used FP32 with
AMP/TF32 off, accumulation 6, and fixed non-trainable rho=0.05. No NaN, Inf,
OOM, residual/reconstruction, surgery, or MAIN exact-change invariant failed.

Router supervision remained material: cumulative informative support was
0.89666 overall, 0.89649 normal, and 0.92554 anomaly. The cumulative target
median normalized entropy was 0.70045 (0.69558 normal mean; 0.77487 anomaly
mean). Student dense-router entropy moved from 0.99903 at batch 50 to 0.99877
at batch 300, while q/raw-winner top-1 agreement rose from 0.31364 to
0.47138. The final Router gradient was nonzero (`2.8729e-04`); the final
lambda-weighted Router-to-main shared-gradient ratio was 0.11539. Router usage
remained distributed: [0.25808, 0.24330, 0.24689, 0.25174].

Final cumulative loss diagnostics (overall / normal / anomaly) were:

| Metric | Overall | Normal | Anomaly |
| --- | ---: | ---: | ---: |
| Base | 0.07679654 | 0.05828443 | 3.22461820 |
| FullSoftRouted ACT=1 | 0.07688454 | 0.05839553 | 3.22077918 |
| ActualGated | 0.07682879 | 0.05832373 | 3.22345281 |
| ResidualBestSingle | 0.07676001 | 0.05822862 | 3.21505284 |
| ResidualOracleMulti | 0.07655331 | 0.05810275 | 3.21390867 |

On normal patches, ActualGated suppressed `0.00007180` of the
`0.00011110` FullSoftRouted ACT=1 damage (64.6%). On anomaly patches, it
retained `0.00116539` of the `0.00383902` FullSoftRouted ACT=1 benefit
(30.4%). These are training diagnostics only, not medical evaluation.

Decision: `ROUTER_300B_EVIDENCE_READY_FOR_REVIEW`.

No second attempt, target variation, Router eligibility change, threshold
change, loss reweighting, capacity change, or medical evaluation is authorized
by this result. Next action is discussion only.
