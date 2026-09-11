#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dataset import get_text_and_image_dataset

dataset = get_text_and_image_dataset("VisA", 518, "train")
for workers in (1, 2):
    loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=workers)
    start = time.perf_counter()
    for index, _ in enumerate(loader):
        if index >= 7:
            break
    print(f"workers={workers} seconds={time.perf_counter() - start:.6f}")
