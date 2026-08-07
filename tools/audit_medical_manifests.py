#!/usr/bin/env python3
import json
import hashlib
import os
from pathlib import Path

datasets = ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]
manifest_root = Path("runs/phase4/progress1_v7_full_seed0_ready3/train/protocol/medical_manifests")

print(f"{'Dataset':<16} | {'Manifest Path':<55} | {'SHA-256':<64} | {'Total':<6} | {'Norm':<6} | {'Anom':<6} | {'Masks':<6}")
print("-" * 170)

manifest_info = {}

for ds in datasets:
    if ds in ["Brain", "Liver", "Retina"]:
        mpath = Path(f"dataset/hub/{ds}.jsonl")
    else:
        mpath = manifest_root / f"{ds}_test.jsonl"
    
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
    
    tot = len(rows)
    norm = sum(1 for r in rows if r["label"] == 0)
    anom = sum(1 for r in rows if r["label"] == 1)
    masks = sum(1 for r in rows if r.get("mask_path") is not None)
    
    print(f"{ds:<16} | {str(mpath):<55} | {sha} | {tot:<6} | {norm:<6} | {anom:<6} | {masks:<6}")
    manifest_info[ds] = {
        "manifest_path": str(mpath),
        "sha256": sha,
        "total_samples": tot,
        "normal_samples": norm,
        "anomaly_samples": anom,
        "mask_samples": masks,
        "split": "test",
    }

os.makedirs("runs/phase4/p1_v8_2_medical_e15_e20/protocol", exist_ok=True)
with open("runs/phase4/p1_v8_2_medical_e15_e20/protocol/manifest_fingerprints.json", "w") as f:
    json.dump(manifest_info, f, indent=2)
