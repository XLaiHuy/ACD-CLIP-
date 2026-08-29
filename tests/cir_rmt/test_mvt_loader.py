from pathlib import Path
from types import SimpleNamespace
import json

import torch

import scripts.cir_rmt.train_full as train_full


class _RecordingDataset:
    def __init__(self, data_path: str, meta_path: str, img_size: int):
        self.data_path = data_path
        self.meta_path = meta_path
        self.img_size = img_size
        self.meta = [json.loads(line) for line in Path(meta_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.meta[int(index)]


def _args():
    return SimpleNamespace(micro_batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2)


def test_mvtec_loader_uses_canonical_supervised_manifest(monkeypatch, tmp_path):
    captured = {}

    def factory(data_path, meta_path, img_size):
        dataset = _RecordingDataset(data_path, meta_path, img_size)
        captured["dataset"] = dataset
        return dataset

    monkeypatch.setattr(train_full, "TextAndImageDataset", factory)
    dataset, loader = train_full.build_loader("mvtec", tmp_path, {"img_size": 518}, _args(), torch.Generator().manual_seed(0))
    assert dataset is captured["dataset"]
    assert Path(dataset.meta_path).name == "MVTec.jsonl"
    repo_root = Path(__file__).resolve().parents[2]
    assert Path(dataset.meta_path).resolve() == (repo_root / "dataset" / "hub" / "MVTec.jsonl").resolve()
    assert Path(dataset.data_path) == tmp_path
    assert dataset.img_size == 518
    labels = {int(row["label"]) for row in dataset.meta}
    assert labels == {0, 1}
    assert any(int(row["label"]) == 1 and row.get("mask_path") for row in dataset.meta)
    assert any(int(row["label"]) == 0 and "mask_path" not in row for row in dataset.meta)
    assert len(loader.dataset) == len(dataset)
    assert not hasattr(train_full, "MVTecTrainDataset")
