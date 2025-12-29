"""Tests for cassava_leaf_disease.data.download_model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_download_model_to_dvc_requires_uri() -> None:
    from cassava_leaf_disease.data.download_model import download_model_to_dvc

    cfg = SimpleNamespace(download_model=SimpleNamespace(s3_uri=None))
    res = download_model_to_dvc(cfg)
    assert res.success is False
    assert "s3_uri is required" in res.message


def test_download_model_to_dvc_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.data import download_model as dm

    monkeypatch.setattr(
        dm,
        "download_checkpoint_from_s3_uri",
        lambda **_k: SimpleNamespace(success=False, path=None, message="nope"),
    )

    cfg = SimpleNamespace(download_model=SimpleNamespace(s3_uri="s3://b/k", dst_dir="artifacts"))
    res = dm.download_model_to_dvc(cfg)
    assert res.success is False
    assert "Download failed" in res.message


def test_download_model_to_dvc_add_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.data import download_model as dm

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake")

    monkeypatch.setattr(
        dm,
        "download_checkpoint_from_s3_uri",
        lambda **_k: SimpleNamespace(success=True, path=ckpt, message="ok"),
    )

    import dvc.repo

    class DummyRepo:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def add(self, *_a, **_k):
            raise RuntimeError("add failed")

    monkeypatch.setattr(dvc.repo, "Repo", lambda *_a, **_k: DummyRepo())

    cfg = SimpleNamespace(
        download_model=SimpleNamespace(
            s3_uri="s3://b/k/best.ckpt",
            dst_dir=str(tmp_path),
            overwrite=False,
            push=False,
            remote="yandex_s3",
        )
    )
    res = dm.download_model_to_dvc(cfg)
    assert res.success is False
    assert "Failed to add checkpoint to DVC" in res.message


def test_download_model_to_dvc_success_add_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.data import download_model as dm

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake")

    monkeypatch.setattr(
        dm,
        "download_checkpoint_from_s3_uri",
        lambda **_k: SimpleNamespace(success=True, path=ckpt, message="ok"),
    )

    class DummyRepo:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def add(self, *_a, **_k):
            return None

    import dvc.repo

    monkeypatch.setattr(dvc.repo, "Repo", lambda *_a, **_k: DummyRepo())

    cfg = SimpleNamespace(
        download_model=SimpleNamespace(
            s3_uri="s3://b/k/best.ckpt",
            dst_dir=str(tmp_path),
            overwrite=False,
            push=False,
            remote="yandex_s3",
        )
    )
    res = dm.download_model_to_dvc(cfg)
    assert res.success is True
    assert res.checkpoint_path == ckpt


def test_download_model_to_dvc_push_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from cassava_leaf_disease.data import download_model as dm

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake")

    monkeypatch.setattr(
        dm,
        "download_checkpoint_from_s3_uri",
        lambda **_k: SimpleNamespace(success=True, path=ckpt, message="ok"),
    )

    class DummyRepo:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def add(self, *_a, **_k):
            return None

        def push(self, *_a, **_k):
            raise RuntimeError("push failed")

    import dvc.repo

    monkeypatch.setattr(dvc.repo, "Repo", lambda *_a, **_k: DummyRepo())

    cfg = SimpleNamespace(
        download_model=SimpleNamespace(
            s3_uri="s3://b/k/best.ckpt",
            dst_dir=str(tmp_path),
            overwrite=False,
            push=True,
            remote="yandex_s3",
        )
    )
    res = dm.download_model_to_dvc(cfg)
    assert res.success is True
    assert "push failed" in res.message
