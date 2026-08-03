import json
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode

from .transforms import AddGaussianNoise
from .info import CLASS_NAMES, DATA_PATH, DOMAINS, MEDICAL_EVAL_PATHS


class TextAndImageDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))

        self.transforms_list = [
            tv_transforms.RandomApply(
                [tv_transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            tv_transforms.RandomApply(
                [tv_transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            tv_transforms.RandomHorizontalFlip(p=0.5),
            tv_transforms.RandomVerticalFlip(p=0.5),
        ]

        transform_x = [
            AddGaussianNoise(std=1, p=0.7),
            tv_transforms.RandomApply([tv_transforms.ColorJitter(brightness=0.5)], p=0.7),
            tv_transforms.RandomApply([tv_transforms.ColorJitter(contrast=0.5)], p=0.7),
            tv_transforms.RandomApply([tv_transforms.ColorJitter(saturation=0.5)], p=0.7)
        ]
        self.transform_x = tv_transforms.Compose(
            transform_x
            + [
                tv_transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = tv_transforms.Compose(
            [
                tv_transforms.Resize((img_size, img_size), InterpolationMode.NEAREST),
                tv_transforms.ToTensor(),
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

        random_transform = tv_transforms.Compose(self.transforms_list)
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
                if m["class_name"] == class_name:
                    self.meta.append(m)

        # Define transforms
        self.transform_x = tv_transforms.Compose(
            [
                tv_transforms.Resize((img_size, img_size), Image.BICUBIC),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = tv_transforms.Compose(
            [
                tv_transforms.Resize((img_size, img_size), Image.NEAREST),
                tv_transforms.ToTensor(),
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
        }
        return inputs


def get_text_and_image_dataset(
        dataset_name: str,
        img_size: int,
        stage: str = "train",
        medical_manifest_root: str | None = None,
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
        dataset = TextAndImageDataset(data_path, meta_path, img_size)
        return dataset
    elif stage in {"test", "val"}:
        if stage == "val":
            if dataset_name not in MEDICAL_EVAL_PATHS:
                raise ValueError(f"validation split is only configured for Phase4 medical datasets, got {dataset_name}")
            if medical_manifest_root is None:
                raise ValueError("medical_manifest_root is required for the Phase4 medical validation split")
            meta_path = os.path.join(medical_manifest_root, f"{dataset_name}_val.jsonl")
            data_path = MEDICAL_EVAL_PATHS[dataset_name]["val"]
        elif dataset_name in MEDICAL_EVAL_PATHS and medical_manifest_root is not None:
            candidate = os.path.join(medical_manifest_root, f"{dataset_name}_test.jsonl")
            # Brain/Liver/Retina keep their official test manifests in dataset/hub.
            # Colon test manifests are generated alongside their validation split.
            if os.path.exists(candidate):
                meta_path = candidate
            else:
                meta_path = os.path.join("./dataset/hub", dataset_name + ".jsonl")
            data_path = MEDICAL_EVAL_PATHS[dataset_name]["test"]
        else:
            meta_path = os.path.join("./dataset/hub", dataset_name + ".jsonl")
            data_path = DATA_PATH[dataset_name]
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=data_path,
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, val, test")
