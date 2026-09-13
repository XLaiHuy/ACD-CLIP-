# Checkpoint guide

The fresh source-transfer runs save full-state checkpoints. Each checkpoint
contains model adapters plus optimizer, scheduler, AMP scaler, RNG, data-loader
state, scientific configuration, and provenance metadata.

## Required checkpoints

| Use | Run V: VisA source | Run M: MVTec AD source |
|---|---|---|
| Best Medical inference/anomaly map | runs/fresh_visa_source_20260911/adapter_7.pth | runs/fresh_mvtec_all_source_20260911/adapter_1.pth |
| Exact fresh-run anchor/resume | runs/fresh_visa_source_20260911/adapter_1.pth | runs/fresh_mvtec_all_source_20260911/adapter_1.pth |
| Continue from final saved state | runs/fresh_visa_source_20260911/adapter_20.pth | runs/fresh_mvtec_all_source_20260911/adapter_20.pth |

adapter_7.pth is the selected Medical checkpoint for Run V. adapter_1.pth
is the selected Medical checkpoint for Run M. The selection criterion is the
highest no-TTA macro Medical pixel AP, then pixel AUROC, then earlier epoch.

The full E1--E20 collections are intentionally not required for inference or
anomaly-map generation. Keeping all forty would add about 10.4 GiB. The Git
commit retains the five checkpoints above, all reports/logs, and the compact
result artifacts.

## Anomaly map use

Use the selected checkpoint for the source run whose model you want to apply:

    VisA-trained model:  runs/fresh_visa_source_20260911/adapter_7.pth
    MVTec-trained model: runs/fresh_mvtec_all_source_20260911/adapter_1.pth

The locked evaluator computes the pixel anomaly map internally and supports
single-view or four-view inference. test.py and the locked TTA evaluator
currently write metrics; a separate visualization/export step is needed to
save heatmap overlays or boxed images as PNG files.

The frozen backbone is model/ViT-L-14-336px.pt and is already tracked with
