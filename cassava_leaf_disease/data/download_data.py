"""Download and extract a public dataset archive (Yandex Disk public link).

This is a convenience helper for reviewers / local development when DVC pull is not available.
It does NOT commit any data into git.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DownloadResult:
    """Result of a download-data attempt."""

    success: bool
    message: str


def download_data(cfg: Any) -> DownloadResult:
    """Download and extract dataset according to Hydra config."""
    public_url = str(cfg.download_data.public_url)
    data_dir = Path(str(cfg.paths.data_dir)).resolve()
    force = bool(getattr(cfg.download_data, "force", False))

    expected_csv = data_dir / "train.csv"
    expected_images = data_dir / "train_images"

    if not force and expected_csv.exists() and expected_images.exists():
        return DownloadResult(success=True, message=f"Dataset already exists at: {data_dir}")

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        _download_and_extract_yadisk_public(public_url=public_url, data_dir=data_dir, force=force)
    except Exception as exc:
        return DownloadResult(success=False, message=str(exc))

    return DownloadResult(success=True, message=f"Downloaded and extracted to: {data_dir}")


def _download_and_extract_yadisk_public(public_url: str, data_dir: Path, force: bool) -> None:
    href = _get_yadisk_download_href(public_url)
    downloads_dir = (data_dir / "_downloads").resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    archive_path = downloads_dir / _infer_filename_from_url(href)
    if archive_path.exists() and not force:
        _extract_archive(archive_path, data_dir=data_dir, force=force)
        return

    _download_file(href, archive_path)
    _extract_archive(archive_path, data_dir=data_dir, force=force)


def _get_yadisk_download_href(public_url: str) -> str:
    """Resolve a Yandex Disk public link into a direct download URL (href)."""
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    url = f"{api}?public_key={quote(public_url, safe='')}"
    req = Request(url, headers={"User-Agent": "cassava-leaf-disease/0.1"})
    with urlopen(req, timeout=60) as response:
        payload = response.read().decode("utf-8", errors="replace")

    data = json.loads(payload)
    href = data.get("href")
    if not isinstance(href, str) or not href:
        raise ValueError("Failed to resolve Yandex Disk public link: missing href")
    return href


def _infer_filename_from_url(url: str) -> str:
    # Yandex Disk href is usually a URL with a filename in its path; keep a safe fallback.
    from urllib.parse import urlparse

    name = Path(urlparse(url).path).name
    if not name:
        return "dataset.zip"
    return name


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "cassava-leaf-disease/0.1"})
    with urlopen(req, timeout=120) as response, dst.open("wb") as f:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _extract_archive(archive_path: Path, data_dir: Path, force: bool) -> None:
    tmp_dir = (data_dir / "_tmp_extract").resolve()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(tmp_dir)
        else:
            raise ValueError(
                f"Unsupported archive format: {archive_path.name}. Expected zip or tar.*"
            )

        extracted_root = _pick_extracted_root(tmp_dir)
        _move_tree_contents(extracted_root, dst=data_dir, force=force)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _pick_extracted_root(tmp_dir: Path) -> Path:
    entries = [p for p in tmp_dir.iterdir() if p.name not in {"__MACOSX"}]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return tmp_dir


def _move_tree_contents(src: Path, dst: Path, force: bool) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if not force:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))
