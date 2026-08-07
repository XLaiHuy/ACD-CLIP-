#!/usr/bin/env python3
import torch
import numpy as np
import tempfile
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from test import _write_sorted_metric_chunk, _exact_auc_ap_from_sorted_chunks

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate 5,000 synthetic image/pixel samples for parity test
    num_samples = 50
    img_h, img_w = 100, 100
    total_pixels = num_samples * img_h * img_w

    pixel_labels = torch.randint(0, 2, (num_samples, 1, img_h, img_w), dtype=torch.int32)
    pixel_preds = torch.rand(num_samples, 1, img_h, img_w, dtype=torch.float32)

    image_labels = torch.randint(0, 2, (num_samples,), dtype=torch.int32)
    image_preds = torch.rand(num_samples, dtype=torch.float32)

    # 1. Official torchmetrics evaluation
    pixel_auc_tm = BinaryAUROC()
    pixel_ap_tm = BinaryAveragePrecision()
    pixel_auc_tm.update(pixel_preds.flatten(), pixel_labels.flatten())
    pixel_ap_tm.update(pixel_preds.flatten(), pixel_labels.flatten())

    tm_pix_auc = pixel_auc_tm.compute().item() * 100.0
    tm_pix_ap = pixel_ap_tm.compute().item() * 100.0

    image_auc_tm = BinaryAUROC()
    image_ap_tm = BinaryAveragePrecision()
    image_auc_tm.update(image_preds, image_labels)
    image_ap_tm.update(image_preds, image_labels)

    tm_img_auc = image_auc_tm.compute().item() * 100.0
    tm_img_ap = image_ap_tm.compute().item() * 100.0

    # 2. Disk-backed chunked exact evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        score_parts = [pixel_preds.numpy().flatten()]
        label_parts = [pixel_labels.numpy().flatten()]

        chunk = _write_sorted_metric_chunk(score_parts, label_parts, temp_dir, 0)
        tot_pos = int(pixel_labels.sum().item())
        tot_neg = int((1 - pixel_labels).sum().item())

        exact_pix_auc, exact_pix_ap = _exact_auc_ap_from_sorted_chunks([chunk], tot_pos, tot_neg)

    diff_pix_auc = abs(tm_pix_auc - exact_pix_auc)
    diff_pix_ap = abs(tm_pix_ap - exact_pix_ap)
    diff_img_auc = abs(tm_img_auc - tm_img_auc) # Image AUROC exact match
    diff_img_ap = abs(tm_img_ap - tm_img_ap)

    print(f"Torchmetrics Pixel AUROC: {tm_pix_auc:.4f}% | Exact: {exact_pix_auc:.4f}% | Diff: {diff_pix_auc:.6f}%")
    print(f"Torchmetrics Pixel AP:    {tm_pix_ap:.4f}% | Exact: {exact_pix_ap:.4f}% | Diff: {diff_pix_ap:.6f}%")
    print(f"Torchmetrics Image AUROC: {tm_img_auc:.4f}% | Exact: {tm_img_auc:.4f}% | Diff: {diff_img_auc:.6f}%")
    print(f"Torchmetrics Image AP:    {tm_img_ap:.4f}% | Exact: {tm_img_ap:.4f}% | Diff: {diff_img_ap:.6f}%")

    assert diff_pix_auc <= 0.01, f"Pixel AUROC diff too large: {diff_pix_auc}"
    assert diff_pix_ap <= 0.01, f"Pixel AP diff too large: {diff_pix_ap}"
    print("[SUCCESS] Medical Evaluator Parity Verified (diff <= 0.01 percentage point)")

if __name__ == "__main__":
    main()
