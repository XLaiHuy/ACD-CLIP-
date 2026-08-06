import torch
import numpy as np
import tempfile
import sys
import os
import json
import heapq

from torchmetrics.functional import auroc, average_precision

def _write_sorted_metric_chunk(score_parts, label_parts, temp_dir: str, chunk_index: int):
    scores = np.concatenate(score_parts).astype(np.float32, copy=False)
    labels = np.concatenate(label_parts).astype(np.uint8, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    score_path = os.path.join(temp_dir, f"scores_{chunk_index:05d}.npy")
    label_path = os.path.join(temp_dir, f"labels_{chunk_index:05d}.npy")
    np.save(score_path, scores[order])
    np.save(label_path, labels[order])
    return score_path, label_path

def _exact_auc_ap_from_sorted_chunks(chunks, total_pos: int, total_neg: int):
    if total_pos == 0 or total_neg == 0:
        return 0.0, 0.0
    arrays = [
        (np.load(score_path, mmap_mode="r"), np.load(label_path, mmap_mode="r"))
        for score_path, label_path in chunks
    ]
    heap = []
    for chunk_id, (scores, labels) in enumerate(arrays):
        if len(scores) > 0:
            heapq.heappush(heap, (-float(scores[0]), chunk_id, 0))

    pos_seen = 0
    neg_seen = 0
    auc_pair_sum = 0.0
    ap = 0.0
    while heap:
        score = heap[0][0]
        group_pos = 0
        group_neg = 0
        while heap and heap[0][0] == score:
            _, chunk_id, index = heapq.heappop(heap)
            labels = arrays[chunk_id][1]
            if int(labels[index]) == 1:
                group_pos += 1
            else:
                group_neg += 1
            next_index = index + 1
            scores = arrays[chunk_id][0]
            if next_index < len(scores):
                heapq.heappush(heap, (-float(scores[next_index]), chunk_id, next_index))

        auc_pair_sum += group_neg * (pos_seen + 0.5 * group_pos)
        pos_seen += group_pos
        neg_seen += group_neg
        if group_pos > 0:
            precision = pos_seen / max(pos_seen + neg_seen, 1)
            ap += precision * (group_pos / total_pos)

    return (auc_pair_sum / (total_pos * total_neg)) * 100.0, ap * 100.0

def run_parity_check():
    torch.manual_seed(42)
    np.random.seed(42)

    n_pixels = 2_000_000
    pixel_preds = torch.randn(n_pixels, device="cpu")
    pixel_labels_dist = torch.rand(n_pixels, device="cpu")
    pixel_labels = (pixel_labels_dist > 0.95).int()

    print("Computing torchmetrics...")
    tm_auc = auroc(pixel_preds, pixel_labels, task="binary").item()
    tm_ap = average_precision(pixel_preds, pixel_labels, task="binary").item()

    print("Computing disk-backed metrics...")
    scores = pixel_preds.numpy()
    labels = pixel_labels.numpy().astype(np.uint8)
    
    total_pos = int(labels.sum())
    total_neg = int(labels.size - labels.sum())

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_file = _write_sorted_metric_chunk([scores], [labels], temp_dir, 0)
        chunks = [chunk_file]
        disk_auc, disk_ap = _exact_auc_ap_from_sorted_chunks(chunks, total_pos, total_neg)
    
    disk_auc /= 100.0  # it returns percentages
    disk_ap /= 100.0
    
    auc_diff = abs(tm_auc * 100 - disk_auc * 100)
    ap_diff = abs(tm_ap * 100 - disk_ap * 100)
    
    result = {
        "torchmetrics": {
            "pixel_AUROC": tm_auc * 100,
            "pixel_AP": tm_ap * 100
        },
        "disk_backed": {
            "pixel_AUROC": disk_auc * 100,
            "pixel_AP": disk_ap * 100
        },
        "differences": {
            "AUROC_diff": auc_diff,
            "AP_diff": ap_diff
        },
        "parity_passed": auc_diff <= 0.01 and ap_diff <= 0.01
    }
    
    out_path = "runs/phase4/p1_fast_audit/metric_parity.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_parity_check()
