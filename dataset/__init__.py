import csv
import hashlib
import json
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import AddGaussianNoise
from .info import CLASS_NAMES, DATA_PATH, DOMAINS


def deterministic_sample_id(meta):
    """Return a stable ID without depending on JSONL order or absolute paths."""
    image_path = str(meta["image_path"]).replace("\\", "/").lstrip("./")
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:20]


def _manifest_sample_ids(manifest_path):
    if manifest_path is None:
        return None
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise ValueError(f"Manifest {manifest_path} must contain a sample_id column")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Manifest {manifest_path} contains duplicate sample IDs")
    return set(sample_ids)


class TextAndImageDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
            manifest_path: str | None = None,
            augment: bool = True,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.augment = augment
        allowed_ids = _manifest_sample_ids(manifest_path)
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                record = json.loads(line)
                record["sample_id"] = deterministic_sample_id(record)
                if allowed_ids is None or record["sample_id"] in allowed_ids:
                    self.meta.append(record)
        if allowed_ids is not None:
            found = {record["sample_id"] for record in self.meta}
            missing = sorted(allowed_ids - found)
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} contains {len(missing)} IDs absent from {meta_path}: {missing[:3]}"
                )

        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = []
        if augment:
            transform_x = [
                AddGaussianNoise(std=1, p=0.7),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7),
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7),
                transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
            ]
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        if self.augment:
            random_transform = transforms.Compose(self.transforms_list)
            transform_tensor = torch.cat([img, mask], dim=0)
            assert transform_tensor.shape[0] == 4
            transform_tensor = random_transform(transform_tensor)
            img = transform_tensor[0:3, :, :]
            mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "sample_id": meta["sample_id"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
            class_name: str
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                m["sample_id"] = deterministic_sample_id(m)
                if m["class_name"] == class_name:
                    self.meta.append(m)

        # Define transforms
        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        if meta["label"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "sample_id": meta["sample_id"],
        }
        return inputs


def get_text_and_image_dataset(
        dataset_name: str,
        img_size: int,
        stage: str = "train",
        manifest_path: str | None = None,
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )
    if stage == "train":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + ".jsonl"
        )
        data_path = DATA_PATH[dataset_name.split("-")[0]]
        dataset = TextAndImageDataset(
            data_path, meta_path, img_size, manifest_path=manifest_path, augment=True
        )
        return dataset
    elif stage == "val":
        meta_path = os.path.join("./dataset/hub", dataset_name + ".jsonl")
        data_path = DATA_PATH[dataset_name.split("-")[0]]
        return TextAndImageDataset(
            data_path, meta_path, img_size, manifest_path=manifest_path, augment=False
        )
    elif stage == "test":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + ".jsonl"
        )
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, val, test")
