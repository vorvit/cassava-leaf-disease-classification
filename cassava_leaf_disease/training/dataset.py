"""Datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CassavaSample:
    image_id: str
    label: int


class CassavaCsvDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Kaggle cassava dataset backed by `train.csv` and `train_images/` directory."""

    def __init__(self, samples: list[CassavaSample], images_dir: Path, transform: Any) -> None:
        self._samples = samples
        self._images_dir = images_dir
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        image_path = self._images_dir / sample.image_id
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_np = np.array(img)

        transformed = self._transform(image=image_np)
        image_tensor: torch.Tensor = transformed["image"]
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        return image_tensor, label_tensor


class SyntheticCassavaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic synthetic dataset that is learnable (loss decreases quickly).

    Each class corresponds to a different dominant color channel intensity pattern.
    """

    def __init__(self, size: int, num_classes: int, image_size: int, seed: int) -> None:
        self._size = int(size)
        self._num_classes = int(num_classes)
        self._image_size = int(image_size)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = int(idx % self._num_classes)

        base = np.zeros((self._image_size, self._image_size, 3), dtype=np.float32)
        channel = label % 3
        intensity = (label + 1) / (self._num_classes + 1)
        base[..., channel] = intensity

        noise = self._rng.normal(loc=0.0, scale=0.05, size=base.shape).astype(np.float32)
        image = np.clip(base + noise, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor
