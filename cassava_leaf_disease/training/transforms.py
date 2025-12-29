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

    transforms: list[Any] = []
    resize_mode = str(getattr(augment_cfg, "resize_mode", "resize")).lower()
    if is_train and resize_mode == "random_resized_crop":
        scale_min = float(getattr(augment_cfg, "rrc_scale_min", 0.8))
        scale_max = float(getattr(augment_cfg, "rrc_scale_max", 1.0))
        transforms.append(
            alb.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(scale_min, scale_max),
                ratio=(0.9, 1.1),
                p=1.0,
            )
        )
    else:
        transforms.append(alb.Resize(image_size, image_size))

    if is_train:
        horizontal_flip_p = float(getattr(augment_cfg, "horizontal_flip_p", 0.5))
        random_brightness_contrast_p = float(
            getattr(augment_cfg, "random_brightness_contrast_p", 0.2)
        )
        shift_scale_rotate_p = float(getattr(augment_cfg, "shift_scale_rotate_p", 0.0))
        hue_saturation_value_p = float(getattr(augment_cfg, "hue_saturation_value_p", 0.0))
        coarse_dropout_p = float(getattr(augment_cfg, "coarse_dropout_p", 0.0))
        gaussian_blur_p = float(getattr(augment_cfg, "gaussian_blur_p", 0.0))
        transforms.extend(
            [
                alb.HorizontalFlip(p=horizontal_flip_p),
                alb.RandomBrightnessContrast(p=random_brightness_contrast_p),
            ]
        )
        if shift_scale_rotate_p > 0:
            transforms.append(
                alb.ShiftScaleRotate(
                    shift_limit=float(getattr(augment_cfg, "shift_limit", 0.0625)),
                    scale_limit=float(getattr(augment_cfg, "scale_limit", 0.1)),
                    rotate_limit=int(getattr(augment_cfg, "rotate_limit", 15)),
                    border_mode=0,
                    p=shift_scale_rotate_p,
                )
            )
        if hue_saturation_value_p > 0:
            transforms.append(
                alb.HueSaturationValue(
                    hue_shift_limit=int(getattr(augment_cfg, "hue_shift_limit", 10)),
                    sat_shift_limit=int(getattr(augment_cfg, "sat_shift_limit", 15)),
                    val_shift_limit=int(getattr(augment_cfg, "val_shift_limit", 10)),
                    p=hue_saturation_value_p,
                )
            )
        if gaussian_blur_p > 0:
            transforms.append(
                alb.GaussianBlur(
                    blur_limit=int(getattr(augment_cfg, "gaussian_blur_limit", 3)),
                    p=gaussian_blur_p,
                )
            )
        if coarse_dropout_p > 0:
            transforms.append(
                alb.CoarseDropout(
                    max_holes=int(getattr(augment_cfg, "max_holes", 8)),
                    max_height=int(getattr(augment_cfg, "max_height", image_size // 10)),
                    max_width=int(getattr(augment_cfg, "max_width", image_size // 10)),
                    min_holes=int(getattr(augment_cfg, "min_holes", 1)),
                    fill_value=0,
                    p=coarse_dropout_p,
                )
            )
    transforms.extend(
        [
            alb.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return alb.Compose(transforms)
