import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

evaluator = importlib.import_module("test")
import utils


class TinyMedicalDataset(Dataset):
    def __init__(self):
        self.samples = []
        for index, score in enumerate((0.1, 0.2, 0.8, 0.9)):
            image = torch.full((1, 4, 4), score, dtype=torch.float32)
            mask = torch.zeros((1, 4, 4), dtype=torch.int32)
            label = int(index >= 2)
            if label:
                mask[:, 1:3, 1:3] = 1
            self.samples.append(
                {
                    "image": image,
                    "mask": mask,
                    "label": torch.tensor(label, dtype=torch.int32),
                    "class_name": "toy",
                    "file_name": f"toy-{index}.png",
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FixedSegmentationModel:
    def __init__(self):
        self.current_image = None

    def __call__(self, image):
        self.current_image = image
        batch = image.shape[0]
        return [torch.zeros(batch, 16, 768)], [torch.zeros(batch, 768)]

    def vision_text_fusion_gate_seg(self, seg_features, text_features, test_mode=False, domain=None):
        assert test_mode is True
        assert domain == "Medical"
        return self.current_image[:, 0]


def _run_stream(batch_size):
    model = FixedSegmentationModel()
    loader = DataLoader(TinyMedicalDataset(), batch_size=batch_size, shuffle=False)
    text = torch.zeros(1, 768, 2)
    return evaluator.get_streaming_metrics(
        model=model,
        class_text_embeddings=text,
        test_loader=loader,
        device=torch.device("cpu"),
        class_name="toy",
        dataset="Brain",
        thresholds=None,
        pixel_stride=1,
        round_result=False,
    )


def test_exact_streaming_matches_naive_and_is_batch_invariant():
    dataset = TinyMedicalDataset()
    text = torch.zeros(1, 768, 2)
    naive_model = FixedSegmentationModel()
    masks, labels, preds, image_preds, _ = evaluator.get_predictions(
        model=naive_model,
        class_text_embeddings=text,
        test_loader=DataLoader(dataset, batch_size=4, shuffle=False),
        device=torch.device("cpu"),
        dataset="Brain",
    )
    naive = utils.metrics_eval_gpu(
        masks,
        labels,
        preds,
        image_preds,
        "toy",
        domain="Medical",
        round_result=False,
    )
    stream_batch1 = _run_stream(1)
    stream_batch3 = _run_stream(3)
    for key in ("pixel AUC", "pixel AP", "image AUC", "image AP"):
        assert np.isclose(stream_batch1[key], naive[key], atol=1e-6)
        assert np.isclose(stream_batch3[key], naive[key], atol=1e-6)


def test_legacy_metric_rounding_is_explicit():
    model = FixedSegmentationModel()
    text = torch.zeros(1, 768, 2)
    masks, labels, preds, image_preds, _ = evaluator.get_predictions(
        model=model,
        class_text_embeddings=text,
        test_loader=DataLoader(TinyMedicalDataset(), batch_size=4, shuffle=False),
        device=torch.device("cpu"),
        dataset="Brain",
    )
    raw = utils.metrics_eval_gpu(
        masks, labels, preds, image_preds, "toy", domain="Medical", round_result=False
    )
    legacy = utils.metrics_eval_gpu(
        masks, labels, preds, image_preds, "toy", domain="Medical", round_result=True
    )
    for key in ("pixel AUC", "pixel AP", "image AUC", "image AP"):
        assert legacy[key] == round(raw[key] / 100.0, 4) * 100


def test_benchmark_exact_disk_spool_matches_streaming_reference(tmp_path):
    reference = _run_stream(2)
    result = evaluator.get_streaming_metrics(
        model=FixedSegmentationModel(),
        class_text_embeddings=torch.zeros(1, 768, 2),
        test_loader=DataLoader(TinyMedicalDataset(), batch_size=2, shuffle=False),
        device=torch.device("cpu"),
        class_name="toy",
        dataset="Brain",
        thresholds=None,
        pixel_stride=1,
        round_result=False,
        spool_root=tmp_path / "spool",
    )
    for key in ("pixel AUC", "pixel AP", "image AUC", "image AP"):
        assert np.isclose(result[key], reference[key], atol=1e-6)
