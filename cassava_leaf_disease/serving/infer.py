"""CLI inference.

This is intentionally lightweight: it loads a checkpoint (weights-only supported),
runs a forward pass on a single image, and prints JSON to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cassava_leaf_disease.data import dvc_pull
from cassava_leaf_disease.training.lightning_module import CassavaClassifier
from cassava_leaf_disease.training.transforms import build_transforms


def _resolve_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(cfg: Any, ckpt_path: Path, device: torch.device) -> CassavaClassifier:
    model = CassavaClassifier(cfg)
    # NOTE: We explicitly set weights_only=False for compatibility with recent PyTorch
    # defaults (weights_only=True) and because our Lightning checkpoints can contain
    # metadata objects (e.g. OmegaConf DictConfig). Only load checkpoints you trust.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Support both Lightning full checkpoints and "weights-only" checkpoints.
    state_dict = ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None)
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format (expected dict with 'state_dict').")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]} (and more)")

    # Missing keys can happen when strict=False; keep it strict-ish for a clean UX.
    if missing:
        raise ValueError(f"Missing keys in checkpoint: {missing[:5]} (and more)")

    model.eval()
    model.to(device)
    return model


def infer(cfg: Any) -> dict[str, Any]:
    """Run inference for a single image path."""
    data_dir = str(cfg.paths.data_dir)
    pull_result = dvc_pull(targets=[data_dir])
    if not pull_result.success:
        print(f"[dvc] pull failed (continuing): {pull_result.message}")

    image_path_raw = getattr(cfg.infer, "image_path", None)
    if image_path_raw in (None, "null"):
        raise SystemExit(
            "infer.image_path is required (e.g. infer.image_path=data/cassava/train_images/xxx.jpg)"
        )

    image_path = Path(str(image_path_raw))
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    ckpt_path = Path(str(cfg.infer.checkpoint_path))
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = _resolve_device(str(cfg.infer.device))
    model = _load_model(cfg, ckpt_path=ckpt_path, device=device)

    class_names = list(cfg.data.dataset.class_names)
    transform = build_transforms(cfg.augment, is_train=False)

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    aug = transform(image=arr)
    inputs = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    top_k = int(getattr(cfg.infer, "top_k", len(class_names)))
    top_k = max(1, min(top_k, len(class_names)))
    top_idx = np.argsort(-probs)[:top_k].tolist()

    pred_id = int(np.argmax(probs))
    result: dict[str, Any] = {
        "predicted_class_id": pred_id,
        "class_name": class_names[pred_id],
        "confidence": float(probs[pred_id]),
        "top_k": [
            {"class_id": int(i), "class_name": class_names[int(i)], "prob": float(probs[int(i)])}
            for i in top_idx
        ],
        "probabilities": {name: float(p) for name, p in zip(class_names, probs, strict=True)},
    }

    print(json.dumps(result, ensure_ascii=False))
    return result
