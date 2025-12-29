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
