"""Image transforms."""

from __future__ import annotations

from typing import Any


def build_transforms(augment_cfg: Any, is_train: bool) -> Any:
    """Build Albumentations transforms from config."""
    import albumentations as alb
    from albumentations.pytorch import ToTensorV2

    image_size = int(augment_cfg.image_size)
    mean = list(augment_cfg.mean)
    std = list(augment_cfg.std)

    transforms: list[Any] = [alb.Resize(image_size, image_size)]
    if is_train:
        horizontal_flip_p = float(getattr(augment_cfg, "horizontal_flip_p", 0.5))
        random_brightness_contrast_p = float(
            getattr(augment_cfg, "random_brightness_contrast_p", 0.2)
        )
        transforms.extend(
            [
                alb.HorizontalFlip(p=horizontal_flip_p),
                alb.RandomBrightnessContrast(p=random_brightness_contrast_p),
            ]
        )
    transforms.extend(
        [
            alb.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return alb.Compose(transforms)
