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


def _find_latest_checkpoint(outputs_dir: Path) -> Path | None:
    """Find the most recently modified checkpoint in outputs/ directory.

    Searches for 'best.ckpt' files in outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/ pattern.
    Returns the most recently modified checkpoint, or None if none found.
    """
    if not outputs_dir.exists():
        return None

    checkpoints: list[tuple[float, Path]] = []

    # Search for best.ckpt files in outputs/ subdirectories
    for ckpt_file in outputs_dir.rglob("best.ckpt"):
        if ckpt_file.is_file():
            # Use modification time as key for sorting
            mtime = ckpt_file.stat().st_mtime
            checkpoints.append((mtime, ckpt_file))

    if not checkpoints:
        return None

    # Sort by modification time (most recent first) and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


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
    # Determine project root (parent of cassava_leaf_disease package directory)
    repo_root = Path(__file__).resolve().parents[2]

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

    # Resolve image path relative to project root (or use as-is if absolute)
    image_path_str = str(image_path_raw)
    image_path_raw_path = Path(image_path_str)
    if image_path_raw_path.is_absolute():
        image_path = image_path_raw_path.resolve()
    else:
        image_path = (repo_root / image_path_str).resolve()
    if not image_path.exists():
        raise SystemExit(
            f"Image not found: {image_path} "
            f"(resolved from {image_path_str} relative to {repo_root})"
        )

    # Resolve checkpoint path with fallback chain:
    # 1. Explicit checkpoint_path from config (if set)
    # 2. Latest checkpoint from outputs/ directory (auto-discovery)
    # 3. S3 URI download (if checkpoint_s3_uri is set)
    ckpt_path_raw = getattr(cfg.infer, "checkpoint_path", None)
    ckpt_path: Path | None = None
    temp_ckpt_path: Path | None = None  # Track temp files for cleanup

    # Normalize checkpoint_path: handle None, "null", and empty strings
    if ckpt_path_raw is None:
        ckpt_path_raw = None
    elif isinstance(ckpt_path_raw, str):
        ckpt_path_raw = ckpt_path_raw.strip()
        if ckpt_path_raw.lower() in ("null", "none", ""):
            ckpt_path_raw = None

    # Step 1: Try explicit checkpoint_path from config
    if ckpt_path_raw is not None:
        ckpt_path_str = str(ckpt_path_raw)
        # If checkpoint path is DVC-tracked, pull it via DVC.
        ckpt_pull_result = dvc_pull(targets=[ckpt_path_str])
        if not ckpt_pull_result.success:
            print(f"[dvc] checkpoint pull failed (continuing): {ckpt_pull_result.message}")
        # Resolve checkpoint path relative to project root (or use as-is if absolute)
        ckpt_path_raw_path = Path(ckpt_path_str)
        if ckpt_path_raw_path.is_absolute():
            ckpt_path = ckpt_path_raw_path.resolve()
        else:
            ckpt_path = (repo_root / ckpt_path_str).resolve()
        if ckpt_path.exists():
            print(f"[infer] Using checkpoint from config: {ckpt_path}")

    # Step 2: If not found, try auto-discovery of latest checkpoint in outputs/
    if ckpt_path is None or not ckpt_path.exists():
        # Resolve outputs_dir relative to project root (or use as-is if absolute)
        outputs_dir_str = str(cfg.paths.outputs_dir)
        outputs_dir_raw_path = Path(outputs_dir_str)
        if outputs_dir_raw_path.is_absolute():
            outputs_dir = outputs_dir_raw_path.resolve()
        else:
            outputs_dir = (repo_root / outputs_dir_str).resolve()
        latest_ckpt = _find_latest_checkpoint(outputs_dir)
        if latest_ckpt is not None:
            ckpt_path = latest_ckpt
            print(f"[infer] Auto-discovered latest checkpoint: {ckpt_path}")

    # Step 3: Fallback to S3 URI download if still not found
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
                "Checkpoint not found. Options:\n"
                "  1. Set infer.checkpoint_path to a DVC-tracked path (e.g. artifacts/best.ckpt)\n"
                "  2. Run training first to create a checkpoint in outputs/\n"
                "  3. Set infer.checkpoint_s3_uri to an S3 URI (e.g. s3://bucket/key/best.ckpt)"
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
