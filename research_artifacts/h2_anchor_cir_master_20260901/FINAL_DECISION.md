# Final research decision

## Decision

**H2 + selective image-parameter anchor, native inference.** The final frozen candidate is **RA E16**. CIR is not carried into the final recipe, and inference-time RMT remains disabled (`alpha=0`).

This follows the preregistered source-only selection: RA E16 had the highest deterministic-source Pixel AUROC. Medical results were confirmatory and did not revise that freeze. The source-side CIR increment was mixed and did not provide robust matched evidence to select RCA over RA.

## Evidence

- Historical H2 replay: PASS; current exact evaluator `0.90922188 / 0.40373065` Pixel AUROC/AP.
- R E10 Medical: `0.89429959 / 0.35442524`.
- RA E16 Medical: `0.87890350 / 0.37162464`.
- RCA E12 Medical: `0.86840354 / 0.37354750`.
- Final RA E16 MVTec: `0.88081744 / 0.39372035` Pixel AUROC/AP; `0.90525987 / 0.95842129` Image AUROC/AP.

At the fixed source E10 gate, RA improves Pixel AUROC/AP over R by `+0.03304292 / +0.08580855`. RCA changes those metrics relative to RA by `-0.04585601 / +0.00252156`, so the CIR signal is mixed. On Medical selected candidates, RA improves Pixel AP but lowers Pixel AUROC; RCA again lowers Pixel AUROC and only slightly raises Pixel AP. Supported image metrics rise for RCA, but this is insufficient to claim robust primary-pixel benefit.

## Interpretation boundary

The result does not support the statement “RMT failed” as a universal claim. It supports the narrower decision that this H2 master study did not establish a robust benefit from CIR training, and no inference-time RMT effect was tested here. The Medical RA-R and RCA-RA rows compare different source-selected epochs, so they are not clean same-epoch causal estimates.

The new R E10 Medical result is below the historical H2 replay. H2 identity, fixed-input parity, and protocol checks pass, but the cause of that target-transfer/reproduction gap remains unknown and should not be silently attributed to Anchor or CIR.

No new architecture search, target tuning, MVTec variant comparison, or post-MVTec tuning is authorized by this snapshot.
