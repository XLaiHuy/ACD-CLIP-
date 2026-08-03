"""Dataset-local transforms kept independent from metric and prompt utilities."""

import random

import torch
from torchvision import transforms


class AddGaussianNoise:
    def __init__(self, std=1.0, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"p={self.p}, std={self.std}"
