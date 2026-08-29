#!/usr/bin/env python3
"""CIR/EVAL-{target}-SOURCE: exact evaluator for one frozen CIR checkpoint."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_records, image_score
from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32
from tools.cir_rmt.identity import config_sha256, git_identity, load_cir_config, sha256_file, validate_checkpoint_identity
from tools.cir_rmt.runtime import forward_cir


MEDICAL_TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
IMAGE_SIZE = 518
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)


class ManifestDataset(Dataset):
    def __init__(self, root: Path, metadata: Path, image_size: int = IMAGE_SIZE):
        self.root = root.expanduser().resolve()
        self.rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
        self.mask = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST), transforms.ToTensor()])
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        with Image.open((self.root / row["image_path"]).resolve()) as handle:
            image = self.image(handle.convert("RGB")).contiguous()
        if int(row.get("label", 0)):
            with Image.open((self.root / row["mask_path"]).resolve()) as handle:
                mask = self.mask(handle.convert("L")).gt(0).to(torch.float32)
        else:
            mask = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        return {"image": image, "mask": mask.contiguous(), "label": torch.tensor(int(row.get("label", 0)), dtype=torch.int64), "class_name": str(row["class_name"]), "image_path": str(row["image_path"])}


class MappingDataset(Dataset):
    def __init__(self, mappings: dict[str, Dataset]):
        self.entries = [(name, index) for name, dataset in mappings.items() for index in range(len(dataset))]
        self.mappings = mappings
    def __len__(self) -> int:
        return len(self.entries)
    def __getitem__(self, index: int) -> dict[str, Any]:
        name, item = self.entries[int(index)]
        row = self.mappings[name][item]
        return {"image": row["image"], "mask": row["mask"], "label": torch.as_tensor(row["label"], dtype=torch.int64), "class_name": name, "image_path": str(row.get("file_name", row.get("image_path", item)))}


def _target_dataset(target: str, root: Path | None) -> Dataset:
    target_lower = target.lower()
    if target_lower == "mvtec":
        if root is None:
            root = resolve_mvtec_root(None)
        if root is None:
            raise ValueError("MVTec target requires --target-root or ACDCLIP_MVTEC_ROOT")
        return MVTecDatasetAdapter(root)
    if target_lower == "visa":
        if root is None:
            raise ValueError("VisA target requires --target-root")
        return ManifestDataset(root, Path(__file__).resolve().parents[2] / "dataset/hub/VisA.jsonl")
    canonical = next((name for name in MEDICAL_TARGETS if name.lower() == target_lower), None)
    if canonical is None:
        raise ValueError(f"unsupported target: {target}")
    if root is None:
        raise ValueError(f"{canonical} target requires --target-root")
    os.environ["ACDCLIP_DATA_ROOT"] = str(root.expanduser().resolve())
    os.environ["MEDICAL_ROOT"] = str(root.expanduser().resolve())
    from dataset import get_text_and_image_dataset
    mappings = get_text_and_image_dataset(canonical, IMAGE_SIZE, stage="test")
    return MappingDataset(mappings)


def _domain(target: str) -> str:
    return "Medical" if any(target.lower() == value.lower() for value in MEDICAL_TARGETS) else "Industrial"


def _write_result(output_dir: Path, result: dict[str, Any], row: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cir_config = load_cir_config(args.config)
    target_lower = str(args.target).lower()
    if (args.source == "visa" and target_lower == "visa") or (args.source == "mvtec" and target_lower == "mvtec"):
        raise ValueError("source and target datasets must be different")
    stage_name = "CIR/EVAL-" + str(args.source).upper() + "-SOURCE"
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(checkpoint, cir_config, source_dataset=args.source, expected_git_sha=git_identity()["head"])
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    parent_config = dict(checkpoint.get("parent_config") or json.loads(parent_path.read_text(encoding="utf-8")))
    configure_canonical_fp32()
    device = torch.device(args.device)
    model = build_phase2b_frozen(parent_config, checkpoint, args.clip_asset, device)
    dataset = _target_dataset(args.target, args.target_root)
    loader_kwargs: dict[str, Any] = {"batch_size": int(args.batch_size), "shuffle": False, "num_workers": int(args.num_workers), "pin_memory": bool(device.type == "cuda")}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": int(args.prefetch_factor)})
    loader = DataLoader(dataset, **loader_kwargs)
    domain = _domain(args.target)
    records = []
    for batch in tqdm(loader, desc=stage_name, unit="img"):
        image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
        names = [str(x) for x in batch["class_name"]]
        output = forward_cir(model, image, names, device, cir_config, domain=domain, require_grad=False, dataset_name=args.target)
        pixels = output.cir_segmentation_probability.detach().cpu().numpy()
        classifications = output.classification_probability.detach().cpu().numpy()
        masks = batch["mask"].detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy()
        paths = [str(x) for x in batch["image_path"]]
        for index, name in enumerate(names):
            pixel = pixels[index].reshape(-1)
            records.append({"class_name": name, "pixel_scores": pixel, "pixel_labels": masks[index].reshape(-1), "image_scores": [image_score(float(classifications[index]), float(pixel.max()), domain)], "image_labels": [int(labels[index])], "image_path": paths[index]})
    evaluated = evaluate_records(records, method="phase2b", allow_undefined_image_metrics=(domain == "Medical"))
    checkpoint_sha = sha256_file(checkpoint_path)
    evaluator_hash = sha256_file(Path(__file__).resolve())
    macro = evaluated["macro"]
    identity = {
        "arch_id": cir_config["arch_id"],
        "architecture_version": cir_config["architecture_version"],
        "source": args.source,
        "target": args.target,
        "epoch": checkpoint.get("epoch"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha256(cir_config),
        "git_sha": checkpoint.get("git_sha"),
        "evaluator_protocol": cir_config["evaluator_protocol"],
        "evaluator_hash": evaluator_hash,
    }
    row = {**identity, **{f"macro_{key}": value for key, value in macro.items()}}
    result = {
        "stage": stage_name,
        "status": "PASS",
        **identity,
        "per_class": evaluated["per_class"],
        "macro": macro,
    }
    _write_result(args.output_dir, result, row)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["visa", "mvtec"], required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args(argv)
    evaluate(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
