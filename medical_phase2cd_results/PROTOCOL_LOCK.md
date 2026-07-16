# Locked medical protocol

{
  "anomaly_channel": 1,
  "batch_size": 8,
  "checkpoint_count": 9,
  "evaluator": "phase2b_anchor_diagnosis.py via phase2cd_medical_eval.py",
  "final_map_resize": {
    "align_corners": true,
    "mode": "bilinear"
  },
  "gaussian_smoothing": {
    "kernel": 9,
    "sigma": 1.5
  },
  "image_score": "cls_only",
  "image_size": 518,
  "metric_thresholds": null,
  "mode": "model.eval + torch.no_grad",
  "model": "OpenAI CLIP ViT-L/14-336",
  "n_groups": 3,
  "num_workers": 6,
  "pixel_metrics": [
    "BinaryAUROC",
    "BinaryAveragePrecision"
  ],
  "pixel_stride": 4,
  "tta": false,
  "units": "percentage"
}
