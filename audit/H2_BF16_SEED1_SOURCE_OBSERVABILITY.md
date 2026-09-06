# H2 BF16 Seed1 source-only observability audit

## Scope and frozen state

VisA test split only; no training, target evaluation, or configuration changes were performed.

`SOURCE_ONLY_AUDIT=PASS`. Frozen-state checks cover the H/A/shared-E1 checkpoint SHA256 values, BF16 autocast metadata, disabled GradScaler, disabled TF32, CLIP SHA, scientific-config hashes, and the VisA JSONL SHA. The evaluator processed all 2,162 source images in its documented category-major order.

- H pixel AUROC/AP: 0.98448344 / 0.78360881
- A pixel AUROC/AP: 0.98437823 / 0.80594487
- H image AUROC/AP: 0.98022872 / 0.98467803
- A image AUROC/AP: 0.98026335 / 0.98524296

Raw float32 pre-smoothing and final maps are retained under the external artifact root recorded in JSON and are intentionally excluded from Git. Quantile estimates use the documented fixed stride-257 source-only sample; means/std and AUROC/AP are full-resolution.

## Interpretation

| Mechanism | H | A | A-H | Interpretation | Confidence |
|---|---:|---:|---:|---|---|
| Source pixel AP | 0.783609 | 0.805945 | +0.022336 | Anchor improves source pixel ranking precision. | High |
| Source pixel AUROC | 0.984483 | 0.984378 | -0.000105 | Global rank ordering is effectively unchanged. | High |
| Source image AP | 0.984678 | 0.985243 | +0.000565 | Image discrimination is already saturated. | High |
| Small/medium/large pixel AP | .02716/.17057/.82856 | .02983/.18120/.85314 | +.00267/+.01062/+.02458 | Improvement is present across fixed source-mask strata, largest for large anomalies. | High |
| Boundary/interior mean score | .26263/.67400 | .22944/.69887 | -.03319/+.02487 | Anchor suppresses boundary-adjacent false positives while increasing interior anomaly response. | High |
| Near-boundary background mean | .13729 | .11324 | -.02405 | Better boundary localization is the main source-side mechanism. | High |
| DFG mean entropy (normal/abnormal) | .85917/.85547 | .92324/.94151 | +.06407/+.08604 | Neither arm is collapsed; A is less concentrated and has greater routing separation. | High |
| H/A feature distance (stages 1/2/3 cosine) | .04461/.06136/.10151 | — | — | A remains closer than H to shared E1 at every stage; its distinction grows in later stages. | High |
| Anchor cap telemetry | 2,520 family observations | — | — | Active mainly in SS2D and DFG query/key families; Conv-LoRA family was not capped. | High |

`BF16_SOURCE_RANKING=HEALTHY`; `BF16_FEATURE_SEPARATION=HEALTHY`; `BF16_DFG_ROUTING=SUSPICIOUS` (highly non-uniform SS2D component but no collapse); `BF16_PROTOTYPE_BEHAVIOR=UNKNOWN` (prototype drift from a historical comparator is unavailable); `BF16_ADAPTER_UTILIZATION=HEALTHY`; `ANCHOR_MECHANISM=HELPFUL`; `PIXEL_LOCALIZATION=BOTTLENECK`; `GENERALIZATION_ONLY_HYPOTHESIS=POSSIBLE`.

This audit establishes source behavior of the two BF16 arms. It cannot establish BF16 degradation relative to a matched valid non-BF16 training protocol, and it cannot establish target generalization because target data were prohibited.
