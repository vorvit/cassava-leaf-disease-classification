"""Tests for cassava_leaf_disease.training.dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


def test_datasets(tmp_path) -> None:
    from cassava_leaf_disease.training.dataset import (
        CassavaCsvDataset,
        CassavaSample,
        SyntheticCassavaDataset,
    )

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "a.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    def identity(image):
        return {"image": torch.zeros((3, 8, 8))}

    ds = CassavaCsvDataset(
        [CassavaSample(image_id="a.jpg", label=1)], images_dir=img_dir, transform=identity
    )
    x, y = ds[0]
    assert x.shape == (3, 8, 8)
    assert int(y.item()) == 1

    with pytest.raises(FileNotFoundError):
        CassavaCsvDataset([CassavaSample(image_id="missing.jpg", label=0)], img_dir, identity)[0]

    syn = SyntheticCassavaDataset(size=5, num_classes=5, image_size=8, seed=1)
    x2, y2 = syn[0]
    assert x2.shape == (3, 8, 8)
    assert 0 <= int(y2.item()) < 5


def test_synthetic_dataset_deterministic() -> None:
    from cassava_leaf_disease.training.dataset import SyntheticCassavaDataset

    syn1 = SyntheticCassavaDataset(size=10, num_classes=5, image_size=8, seed=42)
    syn2 = SyntheticCassavaDataset(size=10, num_classes=5, image_size=8, seed=42)
    x1, y1 = syn1[0]
    x2, y2 = syn2[0]
    assert torch.allclose(x1, x2)
    assert y1 == y2


def test_synthetic_dataset_len() -> None:
    from cassava_leaf_disease.training.dataset import SyntheticCassavaDataset

    syn = SyntheticCassavaDataset(size=100, num_classes=5, image_size=8, seed=1)
    assert len(syn) == 100
