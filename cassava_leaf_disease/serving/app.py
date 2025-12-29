"""FastAPI app for image classification inference.

This is an optional component (not required for Task2). It is kept minimal on purpose.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from cassava_leaf_disease.serving.infer import _load_model, _resolve_device
from cassava_leaf_disease.training.transforms import build_transforms
from cassava_leaf_disease.utils.config import get_default_class_names


def create_app() -> FastAPI:
    app = FastAPI(title="Cassava leaf disease classification")

    class_names = get_default_class_names()
    ckpt_path_env = os.getenv("CASSAVA_CHECKPOINT_PATH")
    device_env = os.getenv("CASSAVA_DEVICE", "auto")

    _cached: dict[str, Any] = {"model": None, "transform": None, "device": None}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    upload_file: Any = File(...)

    @app.post("/predict")
    async def predict(file: UploadFile = upload_file) -> dict[str, Any]:
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=415, detail="Only JPEG/PNG are supported.")

        raw = await file.read()
        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        if not ckpt_path_env:
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint is not configured. Set CASSAVA_CHECKPOINT_PATH.",
            )

        if _cached["model"] is None:
            from omegaconf import OmegaConf

            # Load default configs from repo, to reuse the same preprocessing/model settings.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cfg = OmegaConf.load(os.path.join(repo_root, "configs", "infer.yaml"))
            # ensure dataset class_names are aligned with serving config
            cfg.data.dataset.class_names = class_names
            device = _resolve_device(device_env)
            _cached["model"] = _load_model(cfg, ckpt_path=Path(ckpt_path_env), device=device)
            _cached["transform"] = build_transforms(cfg.augment, is_train=False)
            _cached["device"] = device

        model = _cached["model"]
        transform = _cached["transform"]
        device = _cached["device"]

        arr = np.asarray(image)
        aug = transform(image=arr)
        x = aug["image"].unsqueeze(0).to(device)

        import torch

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        return {
            "predicted_class_id": pred_id,
            "class_name": class_names[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {name: float(p) for name, p in zip(class_names, probs, strict=True)},
        }

    return app


app = create_app()
