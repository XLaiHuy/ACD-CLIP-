#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
checks = {
    "VisA": root / "data/VisA_20220922",
    "MVTec_all_supervised": root / "data/mvtec_ad",
    "Brain": root / "data/MedAD/Brain_AD/test",
    "Liver": root / "data/MedAD/Liver_AD/test",
    "Retina": root / "data/MedAD/Retina_RESC_AD/test",
    "Colon_clinicDB": root / "data/Colon/CVC-ClinicDB",
    "Colon_colonDB": root / "data/Colon/CVC-ColonDB",
    "Colon_Kvasir": root / "data/Colon/Kvasir",
}
expected = {
    "VisA": "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842",
    "MVTec_all_supervised": "1b850c1b52de3a08db0ad4e14adb51933f716bb69e66776a9ed5cde71f6d7129",
    "Brain": "89092dd5f3e36d2e611b115b2a97e4e9ee83af183ebec298abac983a7a323e4e",
    "Liver": "1483b5a43f011a3ef02211d5fa81c5b09031423bd0ca5c0ef6cbf0375fee4fc8",
    "Retina": "d0de975045262b321851ac3770eb7b5e68d4d7fb3bdba833b1cbbbe32f212e24",
    "Colon_clinicDB": "1f057657a64221672a5123c3e87b926d226b9eb6a3276768385ca3a7554cdb5c",
    "Colon_colonDB": "e3be9a5e158bef9a2c7f481827339798152c78e92590df9631c4281a6b6c31c3",
    "Colon_Kvasir": "ac948309511f02e8ec66b9c3b5dbc4a4be5e85d22dbd17742c74212ceee2ee94",
}
for name, data_root in checks.items():
    manifest = root / "dataset/hub" / f"{name}.jsonl"
    if not data_root.is_dir() or not any(data_root.rglob("*")):
        raise SystemExit(f"missing or empty data root: {data_root}")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected[name]:
        raise SystemExit(f"manifest hash mismatch: {name}")
    for row in rows:
        image = data_root / row["image_path"]
        if not image.is_file():
            raise SystemExit(f"missing image: {name} {row['image_path']}")
        if int(row["label"]):
            mask = data_root / row["mask_path"]
            if not mask.is_file():
                raise SystemExit(f"missing mask: {name} {row['mask_path']}")
    print(f"{name}_RECORDS={len(rows)}")
mvtec = [
    json.loads(line)
    for line in (root / "dataset/hub/MVTec_all_supervised.jsonl").read_text().splitlines()
    if line.strip()
]
normal = sum(int(row["label"]) == 0 for row in mvtec)
anomaly = sum(int(row["label"]) == 1 for row in mvtec)
if len(mvtec) != 5354 or normal != 4096 or anomaly != 1258:
    raise SystemExit("MVTec all record counts mismatch")
print(f"MVTEC_ALL_RECORDS={len(mvtec)}")
print(f"MVTEC_NORMAL={normal}")
print(f"MVTEC_ANOMALY={anomaly}")
print("MEDICAL_6_PIXEL_READY=YES")
print("MEDICAL_3_IMAGE_READY=YES")
print("DATA_VALIDATION_PASS")
