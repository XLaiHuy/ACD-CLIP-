import torch
from torchvision import transforms as tv_transforms
import math

def probe_augmentation():
    img_size = 32
    # Mock image and mask
    img = torch.ones(3, img_size, img_size) * 0.5  # middle gray
    mask = torch.ones(1, img_size, img_size)
    
    transform_tensor = torch.cat([img, mask], dim=0)
    
    # Apply a known rotation (e.g., 45 degrees) without randomness
    rot_transform = tv_transforms.RandomRotation(degrees=(45, 45))
    
    out = rot_transform(transform_tensor)
    
    print(f"Output shape: {out.shape}")
    print(f"Fill value for image (expected near 0 if default 0, but wait, 0 is used for tensor): {out[0, 0, 0].item()}")
    print(f"Fill value for mask: {out[3, 0, 0].item()}")
    
    # Check interpolation mode effect (e.g., are there non-binary values in the mask?)
    mask_vals = torch.unique(out[3])
    print(f"Unique mask values after rotation: {mask_vals.tolist()}")

if __name__ == "__main__":
    probe_augmentation()
