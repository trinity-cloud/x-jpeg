"""Training data: random 256x256 crops from a folder of images."""

import random
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from .stats import dct_band_stats

ImageFile.LOAD_TRUNCATED_IMAGES = True

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageCrops(Dataset):
    def __init__(self, root, size: int = 256, scale=(0.3, 1.0)):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in _EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images under {root}")
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=scale, ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> torch.Tensor:
        for attempt in range(5):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)
            except (OSError, SyntaxError):
                idx = random.randrange(len(self.paths))
        raise RuntimeError("Too many unreadable images in a row")


class PairedThumbCrops(Dataset):
    """Deployment-faithful training pairs.

    Returns (thumbnail, crop): the 256x256 bicubic squash of the WHOLE image
    (what the table-prediction encoder sees at inference) and a random
    native-resolution 256x256 crop of the same image (what the JPEG encoder
    actually has to compress, block statistics intact).
    """

    def __init__(
        self, root, size: int = 256, num_tables: int = 3,
        compute_stats: bool = False,
    ):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in _EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images under {root}")
        self.size = size
        self.num_tables = num_tables
        self.compute_stats = compute_stats
        self.crop = transforms.RandomCrop(size, pad_if_needed=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        for attempt in range(5):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                thumb = img.resize((self.size, self.size), Image.BICUBIC)
                crop = self.crop(img)
                if self.compute_stats:
                    full = self.to_tensor(img)
                    stats = dct_band_stats(full.unsqueeze(0), self.num_tables)[0]
                else:
                    stats = torch.empty(0)
                return self.to_tensor(thumb), self.to_tensor(crop), stats
            except (OSError, SyntaxError):
                idx = random.randrange(len(self.paths))
        raise RuntimeError("Too many unreadable images in a row")


def kodak_images(root):
    """Full-resolution PIL images of the Kodak evaluation set."""
    paths = sorted(Path(root).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Kodak .png files under {root}")
    return [(p.name, Image.open(p).convert("RGB")) for p in paths]
