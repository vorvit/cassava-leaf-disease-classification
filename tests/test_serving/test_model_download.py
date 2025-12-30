"""Tests for cassava_leaf_disease.serving.model_download."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_s3_uri_branches() -> None:
    from cassava_leaf_disease.serving import model_download as md

    assert md._parse_s3_uri("s3://b/k") == ("b", "k")
    assert md._parse_s3_uri("s3://bucket/a/b/c.ckpt") == ("bucket", "a/b/c.ckpt")
    with pytest.raises(ValueError):
        md._parse_s3_uri("http://x")
    with pytest.raises(ValueError):
        md._parse_s3_uri("s3://bucket-only")
    with pytest.raises(ValueError):
        md._parse_s3_uri("s3:///k")


def test_download_checkpoint_http_success(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri

    def fake_download_http(url: str, dst_path: Path) -> None:
        assert "storage.yandexcloud.net" in url
        dst_path.write_bytes(b"ok")

    import cassava_leaf_disease.serving.model_download as md

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(md, "_download_http", fake_download_http)

    res = download_checkpoint_from_s3_uri(
        s3_uri="s3://mlops-cassava-project/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=False,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is True
    assert res.path is not None
    assert res.path.exists()
    assert res.path.read_bytes() == b"ok"


def test_download_checkpoint_invalid_uri(tmp_path) -> None:
    from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri

    res = download_checkpoint_from_s3_uri(s3_uri="nope", dst_dir=tmp_path)
    assert res.success is False
    assert res.path is None


def test_load_dotenv_and_strip_quotes(tmp_path) -> None:
    from cassava_leaf_disease.serving import model_download as md

    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# comment",
                "AWS_ACCESS_KEY_ID='a'",
                'AWS_SECRET_ACCESS_KEY="b"',
                "AWS_ENDPOINT_URL=https://storage.yandexcloud.net",
                "INVALID_LINE",
                "",
            ]
        ),
        encoding="utf-8",
    )
    parsed = md._load_dotenv_file(env)
    assert parsed["AWS_ACCESS_KEY_ID"] == "a"
    assert parsed["AWS_SECRET_ACCESS_KEY"] == "b"
    assert "INVALID_LINE" not in parsed


def test_download_checkpoint_already_exists(tmp_path) -> None:
    from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri

    existing = tmp_path / "best.ckpt"
    existing.write_bytes(b"ok")
    res = download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=False,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is True
    assert res.path == existing.resolve()


def test_download_checkpoint_http_failure_cleans_tmp(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cassava_leaf_disease.serving.model_download as md
    from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri

    def boom(_url: str, _dst: Path) -> None:
        raise RuntimeError("fail")

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(md, "_download_http", boom)

    res = download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=True,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is False
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_download_checkpoint_uses_boto3_when_creds_present(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    import cassava_leaf_disease.serving.model_download as md

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")

    class DummyClient:
        def download_file(self, _bucket, _key, filename):
            Path(filename).write_bytes(b"ok")

    class DummyBoto3:
        @staticmethod
        def client(*_a, **_k):
            return DummyClient()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=DummyBoto3.client))

    def no_http(*_a, **_k):
        raise RuntimeError("no http")

    monkeypatch.setattr(md, "_download_http", no_http)

    res = md.download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=True,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is True
    assert res.path is not None
    assert res.path.read_bytes() == b"ok"


def test_load_dotenv_file_exception(tmp_path) -> None:
    """Test _load_dotenv_file when file read fails."""
    from cassava_leaf_disease.serving import model_download as md

    # Create a directory instead of a file to trigger exception
    bad_path = tmp_path / ".env"
    bad_path.mkdir()
    result = md._load_dotenv_file(bad_path)
    assert result == {}


def test_ensure_s3_env_from_dotenv_no_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _ensure_s3_env_from_dotenv when .env doesn't exist."""
    import os

    from cassava_leaf_disease.serving import model_download as md

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    # repo_root without .env file
    md._ensure_s3_env_from_dotenv(repo_root=tmp_path)
    assert "AWS_ACCESS_KEY_ID" not in os.environ


def test_ensure_s3_env_from_dotenv_with_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _ensure_s3_env_from_dotenv when .env exists."""
    import os

    from cassava_leaf_disease.serving import model_download as md

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AWS_ACCESS_KEY_ID=test_key\n", encoding="utf-8")
    md._ensure_s3_env_from_dotenv(repo_root=tmp_path)
    assert os.getenv("AWS_ACCESS_KEY_ID") == "test_key"


def test_download_checkpoint_boto3_exception_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that boto3 exception falls back to HTTP."""
    import sys
    from types import SimpleNamespace

    import cassava_leaf_disease.serving.model_download as md

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")

    class FailingClient:
        def download_file(self, _bucket, _key, filename):
            raise RuntimeError("boto3 failed")

    class DummyBoto3:
        @staticmethod
        def client(*_a, **_k):
            return FailingClient()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=DummyBoto3.client))

    def fake_http(url: str, dst_path: Path) -> None:
        dst_path.write_bytes(b"http_ok")

    monkeypatch.setattr(md, "_download_http", fake_http)

    res = md.download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=True,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is True
    assert res.path is not None
    assert res.path.read_bytes() == b"http_ok"


def test_download_checkpoint_empty_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of empty downloaded file."""
    import cassava_leaf_disease.serving.model_download as md
    from cassava_leaf_disease.serving.model_download import download_checkpoint_from_s3_uri

    def fake_download_http(url: str, dst_path: Path) -> None:
        dst_path.write_bytes(b"")  # Empty file

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(md, "_download_http", fake_download_http)

    res = download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=True,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is False
    assert "empty" in res.message.lower()


def test_download_checkpoint_boto3_empty_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test boto3 download with empty file."""
    import sys
    from types import SimpleNamespace

    import cassava_leaf_disease.serving.model_download as md

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "b")

    class DummyClient:
        def download_file(self, _bucket, _key, filename):
            Path(filename).write_bytes(b"")  # Empty file

    class DummyBoto3:
        @staticmethod
        def client(*_a, **_k):
            return DummyClient()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=DummyBoto3.client))

    def no_http(*_a, **_k):
        raise RuntimeError("no http")

    monkeypatch.setattr(md, "_download_http", no_http)

    res = md.download_checkpoint_from_s3_uri(
        s3_uri="s3://b/x/best.ckpt",
        dst_dir=tmp_path,
        overwrite=True,
        endpoint_url="https://storage.yandexcloud.net",
    )
    assert res.success is False
    assert "empty" in res.message.lower()


def test_http_url_from_s3() -> None:
    """Test _http_url_from_s3 function."""
    from cassava_leaf_disease.serving import model_download as md

    url = md._http_url_from_s3(
        bucket="test-bucket", key="models/best.ckpt", endpoint_url="https://storage.yandexcloud.net"
    )
    assert url == "https://storage.yandexcloud.net/test-bucket/models/best.ckpt"
    # Test with trailing slash
    url2 = md._http_url_from_s3(
        bucket="test-bucket",
        key="models/best.ckpt",
        endpoint_url="https://storage.yandexcloud.net/",
    )
    assert url2 == "https://storage.yandexcloud.net/test-bucket/models/best.ckpt"
