"""Download model checkpoint from S3 and add it to DVC tracking.

This is a helper command to:
1. Download a checkpoint from S3 (Yandex Object Storage) into artifacts/
2. Add it to DVC tracking (dvc add)
3. Optionally push to DVC remote
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri


@dataclass(frozen=True)
class DownloadModelToDvcResult:
    """Result of downloading model and adding to DVC."""

    success: bool
    message: str
    checkpoint_path: Path | None


def download_model_to_dvc(cfg: Any) -> DownloadModelToDvcResult:
    """Download checkpoint from S3 and add it to DVC tracking.

    Steps:
    1. Download from S3 URI into dst_dir (default: artifacts/)
    2. Run `dvc add <checkpoint_path>` to track it
    3. Optionally push to DVC remote if push=True
    """
    s3_uri = getattr(cfg.download_model, "s3_uri", None)
    if s3_uri in (None, "null"):
        return DownloadModelToDvcResult(
            success=False,
            message="download_model.s3_uri is required (e.g. s3://bucket/key/best.ckpt)",
            checkpoint_path=None,
        )

    dst_dir = Path(str(getattr(cfg.download_model, "dst_dir", "artifacts")))
    overwrite = bool(getattr(cfg.download_model, "overwrite", False))
    push_to_remote = bool(getattr(cfg.download_model, "push", False))
    remote_name = str(getattr(cfg.download_model, "remote", "yandex_s3"))

    # Step 1: Download from S3
    endpoint_url = None
    import os

    endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    dl_result = download_checkpoint_from_s3_uri(
        s3_uri=str(s3_uri),
        dst_dir=dst_dir,
        overwrite=overwrite,
        endpoint_url=endpoint_url,
    )

    if not dl_result.success or dl_result.path is None:
        return DownloadModelToDvcResult(
            success=False,
            message=f"Download failed: {dl_result.message}",
            checkpoint_path=None,
        )

    ckpt_path = dl_result.path

    # Step 2: Add to DVC tracking
    try:
        from dvc.repo import Repo as DvcRepo
    except Exception as exc:
        return DownloadModelToDvcResult(
            success=False,
            message=f"Failed to import DVC (install dvc): {exc}",
            checkpoint_path=None,
        )

    try:
        with DvcRepo(".") as repo:
            # dvc add creates a .dvc file and stages it
            repo.add(str(ckpt_path), force=overwrite)
    except Exception as exc:
        return DownloadModelToDvcResult(
            success=False,
            message=f"Failed to add checkpoint to DVC: {exc}",
            checkpoint_path=None,
        )

    # Step 3: Optionally push to remote
    if push_to_remote:
        try:
            with DvcRepo(".") as repo:
                repo.push(targets=[str(ckpt_path)], remote=remote_name)
        except Exception as exc:
            return DownloadModelToDvcResult(
                success=True,
                message=f"Downloaded and added to DVC, but push failed: {exc}",
                checkpoint_path=ckpt_path,
            )

    return DownloadModelToDvcResult(
        success=True,
        message="Downloaded from S3 and added to DVC tracking",
        checkpoint_path=ckpt_path,
    )
