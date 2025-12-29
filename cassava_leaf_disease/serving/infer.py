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
from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri
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

    # Lightning checkpoints include non-model module states (loss, metrics, etc.).
    # For inference we only need the backbone/classifier weights.
    inner = getattr(model, "model", None)
    if not isinstance(inner, torch.nn.Module):
        raise ValueError("Unsupported model wrapper (expected `CassavaClassifier.model`).")

    if any(str(k).startswith("model.") for k in state_dict):
        filtered = {
            str(k)[len("model.") :]: v for k, v in state_dict.items() if str(k).startswith("model.")
        }
        missing, unexpected = inner.load_state_dict(filtered, strict=False)
    else:
        missing, unexpected = inner.load_state_dict(state_dict, strict=False)

    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]} (and more)")
    if missing:
        raise ValueError(f"Missing keys in checkpoint: {missing[:5]} (and more)")

    model.eval()
    model.to(device)
    return model


def infer(cfg: Any) -> dict[str, Any]:
    """Run inference for a single image path."""
    # Pull data via DVC (same as training).
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

    # Pull model checkpoint via DVC (same pattern as data).
    ckpt_path_raw = getattr(cfg.infer, "checkpoint_path", None)
    ckpt_path: Path | None = None
    temp_ckpt_path: Path | None = None  # Track temp files for cleanup

    if ckpt_path_raw not in (None, "null"):
        ckpt_path_str = str(ckpt_path_raw)
        # If checkpoint path is DVC-tracked, pull it via DVC.
        ckpt_pull_result = dvc_pull(targets=[ckpt_path_str])
        if not ckpt_pull_result.success:
            print(f"[dvc] checkpoint pull failed (continuing): {ckpt_pull_result.message}")
        ckpt_path = Path(ckpt_path_str)

    # Fallback: if checkpoint not found locally, try downloading from S3 URI (lazy download).
    if ckpt_path is None or not ckpt_path.exists():
        s3_uri = getattr(cfg.infer, "checkpoint_s3_uri", None)
        if s3_uri not in (None, "null"):
            import os
            import tempfile

            endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
            # Download to a temporary file (will be cleaned up after model loading).
            temp_dir = Path(tempfile.gettempdir()) / "cassava_infer"
            temp_dir.mkdir(parents=True, exist_ok=True)
            dl_result = download_checkpoint_from_s3_uri(
                s3_uri=str(s3_uri),
                dst_dir=temp_dir,
                overwrite=True,  # Always re-download for fresh model
                endpoint_url=endpoint_url,
            )
            if not dl_result.success or dl_result.path is None:
                raise SystemExit(
                    f"Checkpoint not found locally and S3 download failed: {dl_result.message}. "
                    "Set infer.checkpoint_path to a DVC-tracked path or ensure "
                    "infer.checkpoint_s3_uri is accessible."
                )
            ckpt_path = dl_result.path
            temp_ckpt_path = ckpt_path  # Mark for cleanup
            print(f"[infer] Downloaded checkpoint from S3 to temporary file: {ckpt_path}")
        else:
            raise SystemExit(
                "Checkpoint not found. Set infer.checkpoint_path (DVC-tracked) or "
                "infer.checkpoint_s3_uri (S3 URI)."
            )

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

    # Cleanup temporary checkpoint file if it was downloaded from S3.
    if temp_ckpt_path is not None and temp_ckpt_path.exists():
        try:
            temp_ckpt_path.unlink()
            print(f"[infer] Cleaned up temporary checkpoint: {temp_ckpt_path}")
        except Exception as exc:
            print(f"[infer] Warning: failed to cleanup temp file {temp_ckpt_path}: {exc}")

    print(json.dumps(result, ensure_ascii=False))
    return result
