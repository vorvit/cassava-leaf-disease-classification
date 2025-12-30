"""Helpers to download model checkpoints for inference.

We keep this module dependency-light:
- Prefer boto3 when credentials are available.
- Fall back to plain HTTP download (public bucket/object) via stdlib.

Downloaded files must go into a git-ignored directory (e.g. `artifacts/`).
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class DownloadModelResult:
    """Result of a model download attempt."""

    success: bool
    path: Path | None
    message: str


_DEFAULT_ENDPOINT_URL: Final[str] = "https://storage.yandexcloud.net"
_DEFAULT_DOTENV_NAME: Final[str] = ".env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_file(path: Path) -> dict[str, str]:
    """Load a simple KEY=VALUE .env file.

    This intentionally does not implement full dotenv semantics; it's enough for local dev.
    """
    data: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return data

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key:
            data[key] = value
    return data


def _ensure_s3_env_from_dotenv(repo_root: Path) -> None:
    """Populate S3 env vars from `.env` if present and env is missing."""
    import os

    dotenv_path = (repo_root / _DEFAULT_DOTENV_NAME).resolve()
    if not dotenv_path.exists():
        return

    parsed = _load_dotenv_file(dotenv_path)
    for key, value in parsed.items():
        if key not in os.environ:
            os.environ[key] = value


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key -> (bucket, key)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected s3:// URI.")
    rest = s3_uri[len("s3://") :]
    if "/" not in rest:
        raise ValueError("Expected s3://bucket/key format.")
    bucket, key = rest.split("/", 1)
    bucket = bucket.strip()
    key = key.strip().lstrip("/")
    if not bucket or not key:
        raise ValueError("Expected non-empty bucket and key in s3:// URI.")
    return bucket, key


def _http_url_from_s3(bucket: str, key: str, endpoint_url: str) -> str:
    """Convert S3 bucket/key to HTTP URL for public access.

    For Yandex Object Storage, the format is:
    https://storage.yandexcloud.net/<bucket>/<key>

    Note: The bucket must be configured for public read access for this to work.
    """
    endpoint_url = endpoint_url.rstrip("/")
    return f"{endpoint_url}/{bucket}/{key}"


def _download_http(url: str, dst_path: Path) -> None:
    """Download a file via HTTP into dst_path using stdlib."""
    import shutil
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "cassava-leaf-disease/1.0"})
    with urllib.request.urlopen(req) as resp, dst_path.open("wb") as f:
        shutil.copyfileobj(resp, f)


def download_checkpoint_from_s3_uri(
    *,
    s3_uri: str,
    dst_dir: Path,
    overwrite: bool = False,
    endpoint_url: str | None = None,
) -> DownloadModelResult:
    """Download a checkpoint from S3 URI into dst_dir and return its local path.

    The filename is inferred from the S3 key basename.
    """
    try:
        bucket, key = _parse_s3_uri(str(s3_uri))
    except Exception as exc:
        return DownloadModelResult(False, None, f"invalid s3 uri: {exc}")

    endpoint = str(endpoint_url or _DEFAULT_ENDPOINT_URL)
    filename = Path(key).name or "model.ckpt"
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = (dst_dir / filename).resolve()

    if dst_path.exists() and not overwrite:
        return DownloadModelResult(True, dst_path, "already exists")

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        with suppress(Exception):
            tmp_path.unlink()

    # Try boto3 when credentials exist; fallback to HTTP.
    import os

    # If creds are not exported (common on Windows), try repo `.env` before giving up.
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("YC_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("YC_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        repo_root = Path(__file__).resolve().parents[2]
        _ensure_s3_env_from_dotenv(repo_root=repo_root)

    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("YC_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("YC_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint,
            )
            s3.download_file(bucket, key, str(tmp_path))
            tmp_path.replace(dst_path)
            if dst_path.stat().st_size <= 0:
                return DownloadModelResult(False, None, "downloaded file is empty")
            return DownloadModelResult(True, dst_path, "downloaded via boto3")
        except Exception:
            # Fall back to HTTP below.
            pass

    try:
        url = _http_url_from_s3(bucket=bucket, key=key, endpoint_url=endpoint)
        _download_http(url, tmp_path)
        tmp_path.replace(dst_path)
        if dst_path.stat().st_size <= 0:
            return DownloadModelResult(False, None, "downloaded file is empty")
        return DownloadModelResult(True, dst_path, "downloaded via http")
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return DownloadModelResult(False, None, f"download failed: {exc}")
