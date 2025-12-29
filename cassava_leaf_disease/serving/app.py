"""FastAPI app for image classification inference.

This is an optional component (not required for Task2). It is kept minimal on purpose.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


def create_app() -> FastAPI:
    app = FastAPI(title="Cassava leaf disease classification")

    class_names = [
        "Cassava Bacterial Blight",
        "Cassava Brown Streak Disease",
        "Cassava Green Mottle",
        "Cassava Mosaic Disease",
        "Healthy",
    ]

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

        # Minimal baseline inference: random stub until a real exported model is added.
        # We keep it deterministic to be testable.
        arr = np.asarray(image)
        seed = int(arr.mean())
        rng = np.random.default_rng(seed)
        probs = rng.random(len(class_names))
        probs = probs / probs.sum()

        pred_id = int(np.argmax(probs))
        return {
            "predicted_class_id": pred_id,
            "class_name": class_names[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {name: float(p) for name, p in zip(class_names, probs, strict=True)},
        }

    return app


app = create_app()
