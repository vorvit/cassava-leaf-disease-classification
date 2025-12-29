"""Lightweight config helpers.

Serving code (FastAPI) is intentionally kept minimal and should not require Hydra.
However, we still want to avoid duplicating constants (e.g. class names).
"""

from __future__ import annotations

from pathlib import Path


def get_default_class_names() -> list[str]:
    """Return class names from `configs/data/cassava.yaml` if available.

    Falls back to a hard-coded list if configs are not available
    (e.g. installed package without repo).
    """
    default = [
        "Cassava Bacterial Blight",
        "Cassava Brown Streak Disease",
        "Cassava Green Mottle",
        "Cassava Mosaic Disease",
        "Healthy",
    ]

    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "data" / "cassava.yaml"
    ).resolve()
    if not config_path.exists():
        return default

    try:
        from omegaconf import OmegaConf
    except Exception:
        return default

    try:
        cfg = OmegaConf.load(str(config_path))
        class_names = list(cfg.dataset.class_names)
    except Exception:
        return default

    if not class_names:
        return default
    return [str(x) for x in class_names]
