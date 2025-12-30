"""Tests for cassava_leaf_disease.serving.app."""

from __future__ import annotations

import asyncio
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from starlette.datastructures import UploadFile


def test_fastapi_app_predict_direct_call(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.serving.app as app_mod
    from cassava_leaf_disease.serving.app import create_app

    img_bytes = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_bytes, format="PNG")
    img_bytes.seek(0)

    monkeypatch.setenv("CASSAVA_CHECKPOINT_PATH", str(tmp_path / "m.ckpt"))
    (tmp_path / "m.ckpt").write_bytes(b"fake")
    monkeypatch.setenv("CASSAVA_DEVICE", "cpu")

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    import torch

    monkeypatch.setattr(app_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        app_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    # FastAPI app reads `configs/infer.yaml` via OmegaConf.load; stub a minimal cfg structure.
    from omegaconf import OmegaConf

    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda *_a, **_k: OmegaConf.create(
            {"data": {"dataset": {"class_names": []}}, "augment": {}}
        ),
    )

    app = create_app()
    endpoint = None
    for route in app.routes:
        if getattr(route, "path", None) == "/predict":
            endpoint = route.endpoint
            break
    assert endpoint is not None

    upload = UploadFile(
        filename="x.png",
        file=BytesIO(img_bytes.getvalue()),
        headers={"content-type": "image/png"},
    )
    result = asyncio.run(endpoint(upload))
    assert "predicted_class_id" in result


def test_fastapi_app_predict_s3_uri_download(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.serving.app as app_mod
    from cassava_leaf_disease.serving.app import create_app

    monkeypatch.delenv("CASSAVA_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("CASSAVA_CHECKPOINT_S3_URI", "s3://b/x/best.ckpt")
    monkeypatch.setenv("CASSAVA_DEVICE", "cpu")

    # Stub downloader to "download" into tmp_path.
    downloaded = tmp_path / "best.ckpt"

    def fake_download_checkpoint_from_s3_uri(**_k):
        downloaded.write_bytes(b"fake")
        return type("R", (), {"success": True, "path": downloaded, "message": "ok"})()

    monkeypatch.setattr(
        app_mod,
        "download_checkpoint_from_s3_uri",
        fake_download_checkpoint_from_s3_uri,
    )

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    import torch

    monkeypatch.setattr(app_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        app_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    from omegaconf import OmegaConf

    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda *_a, **_k: OmegaConf.create(
            {"data": {"dataset": {"class_names": []}}, "augment": {}}
        ),
    )

    app = create_app()
    predict = None
    for route in app.routes:
        if getattr(route, "path", None) == "/predict":
            predict = route.endpoint
            break
    assert predict is not None

    from io import BytesIO

    import numpy as np
    from PIL import Image
    from starlette.datastructures import UploadFile

    buf = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    upload = UploadFile(
        filename="x.png",
        file=BytesIO(buf.getvalue()),
        headers={"content-type": "image/png"},
    )
    result = asyncio.run(predict(upload))
    assert "predicted_class_id" in result


def test_fastapi_app_error_branches(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.datastructures import UploadFile

    import cassava_leaf_disease.serving.app as app_mod

    app = app_mod.create_app()
    health = None
    predict = None
    for route in app.routes:
        if getattr(route, "path", None) == "/health":
            health = route.endpoint
        if getattr(route, "path", None) == "/predict":
            predict = route.endpoint
    assert health is not None and predict is not None
    assert health() == {"status": "ok"}

    # no checkpoint env -> 503
    monkeypatch.delenv("CASSAVA_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("CASSAVA_CHECKPOINT_S3_URI", raising=False)
    buf = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    upload = UploadFile(
        filename="x.png",
        file=BytesIO(buf.getvalue()),
        headers={"content-type": "image/png"},
    )
    with pytest.raises(app_mod.HTTPException) as exc:
        asyncio.run(predict(upload))
    assert exc.value.status_code == 503

    # wrong content type -> 415
    monkeypatch.setenv("CASSAVA_CHECKPOINT_PATH", str(tmp_path / "m.ckpt"))
    (tmp_path / "m.ckpt").write_bytes(b"fake")
    upload2 = UploadFile(
        filename="x.txt",
        file=BytesIO(b"123"),
        headers={"content-type": "text/plain"},
    )
    with pytest.raises(app_mod.HTTPException) as exc2:
        asyncio.run(predict(upload2))
    assert exc2.value.status_code == 415


def test_fastapi_app_invalid_image(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.serving.app as app_mod

    monkeypatch.setenv("CASSAVA_CHECKPOINT_PATH", str(tmp_path / "m.ckpt"))
    (tmp_path / "m.ckpt").write_bytes(b"fake")
    monkeypatch.setenv("CASSAVA_DEVICE", "cpu")

    app = app_mod.create_app()
    predict = None
    for route in app.routes:
        if getattr(route, "path", None) == "/predict":
            predict = route.endpoint
            break

    upload = UploadFile(
        filename="x.bin",
        file=BytesIO(b"invalid image data"),
        headers={"content-type": "image/png"},
    )
    with pytest.raises(app_mod.HTTPException) as exc:
        asyncio.run(predict(upload))  # type: ignore[misc]
    assert exc.value.status_code == 400


def test_fastapi_app_checkpoint_not_exists(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test app when checkpoint path doesn't exist."""
    import cassava_leaf_disease.serving.app as app_mod

    monkeypatch.setenv("CASSAVA_CHECKPOINT_PATH", str(tmp_path / "missing.ckpt"))
    monkeypatch.setenv("CASSAVA_DEVICE", "cpu")

    from omegaconf import OmegaConf

    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda *_a, **_k: OmegaConf.create(
            {"data": {"dataset": {"class_names": []}}, "augment": {}}
        ),
    )

    app = app_mod.create_app()
    predict = None
    for route in app.routes:
        if getattr(route, "path", None) == "/predict":
            predict = route.endpoint
            break

    buf = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    upload = UploadFile(
        filename="x.png",
        file=BytesIO(buf.getvalue()),
        headers={"content-type": "image/png"},
    )
    with pytest.raises(app_mod.HTTPException) as exc:
        asyncio.run(predict(upload))  # type: ignore[misc]
    assert exc.value.status_code == 503
    assert "does not exist" in exc.value.detail


def test_fastapi_app_s3_download_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test app when S3 download fails."""
    import cassava_leaf_disease.serving.app as app_mod

    monkeypatch.delenv("CASSAVA_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("CASSAVA_CHECKPOINT_S3_URI", "s3://b/x/best.ckpt")
    monkeypatch.setenv("CASSAVA_DEVICE", "cpu")

    def fake_download_checkpoint_from_s3_uri(**_k):
        return type("R", (), {"success": False, "path": None, "message": "download failed"})()

    monkeypatch.setattr(
        app_mod,
        "download_checkpoint_from_s3_uri",
        fake_download_checkpoint_from_s3_uri,
    )

    from omegaconf import OmegaConf

    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda *_a, **_k: OmegaConf.create(
            {"data": {"dataset": {"class_names": []}}, "augment": {}}
        ),
    )

    app = app_mod.create_app()
    predict = None
    for route in app.routes:
        if getattr(route, "path", None) == "/predict":
            predict = route.endpoint
            break

    buf = BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    upload = UploadFile(
        filename="x.png",
        file=BytesIO(buf.getvalue()),
        headers={"content-type": "image/png"},
    )
    with pytest.raises(app_mod.HTTPException) as exc:
        asyncio.run(predict(upload))  # type: ignore[misc]
    assert exc.value.status_code == 503
    assert "download failed" in exc.value.detail
