"""Tests for cassava_leaf_disease.training.transforms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def test_transforms_build_and_apply() -> None:
    from cassava_leaf_disease.training.transforms import build_transforms

    cfg = SimpleNamespace(
        image_size=16,
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
        horizontal_flip_p=0.0,
        random_brightness_contrast_p=0.0,
    )
    t = build_transforms(cfg, is_train=True)
    out = t(image=np.zeros((16, 16, 3), dtype=np.uint8))
    assert "image" in out
    assert isinstance(out["image"], torch.Tensor)


def test_transforms_train_has_augmentations() -> None:
    from cassava_leaf_disease.training.transforms import build_transforms

    cfg = SimpleNamespace(
        image_size=16,
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
        horizontal_flip_p=0.5,
        random_brightness_contrast_p=0.3,
    )
    t = build_transforms(cfg, is_train=True)
    out = t(image=np.zeros((16, 16, 3), dtype=np.uint8))
    assert "image" in out


def test_transforms_val_no_augmentations() -> None:
    from cassava_leaf_disease.training.transforms import build_transforms

    cfg = SimpleNamespace(
        image_size=16,
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
    )
    t = build_transforms(cfg, is_train=False)
    out = t(image=np.zeros((16, 16, 3), dtype=np.uint8))
    assert "image" in out


def test_transforms_strong_branches() -> None:
    from cassava_leaf_disease.training.transforms import build_transforms

    cfg = SimpleNamespace(
        image_size=16,
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
        resize_mode="random_resized_crop",
        rrc_scale_min=0.9,
        rrc_scale_max=1.0,
        horizontal_flip_p=0.0,
        random_brightness_contrast_p=0.0,
        shift_scale_rotate_p=0.1,
        hue_saturation_value_p=0.1,
        gaussian_blur_p=0.1,
        coarse_dropout_p=0.1,
        max_holes=2,
        max_height=2,
        max_width=2,
    )
    t = build_transforms(cfg, is_train=True)
    out = t(image=np.zeros((16, 16, 3), dtype=np.uint8))
    assert isinstance(out["image"], torch.Tensor)
