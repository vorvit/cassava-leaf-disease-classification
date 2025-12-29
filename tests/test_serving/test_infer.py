"""Tests for cassava_leaf_disease.serving.infer."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def test_serving_infer_outputs_json(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    # Create a dummy image
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"fake")

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    monkeypatch.setattr(infer_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        infer_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )
    monkeypatch.setattr(
        infer_mod, "dvc_pull", lambda *a, **k: SimpleNamespace(success=True, message="ok")
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(tmp_path / "outputs")),
        infer=SimpleNamespace(
            image_path=str(img_path), checkpoint_path=str(ckpt), device="cpu", top_k=3
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    result = infer_mod.infer(cfg)
    out = capsys.readouterr().out.strip()
    # Extract JSON from output (may have log messages before it)
    lines = out.split("\n")
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith("{"):
            json_line = line.strip()
            break
    assert json_line is not None, f"No JSON found in output: {out}"
    parsed = json.loads(json_line)
    assert parsed["predicted_class_id"] == result["predicted_class_id"]


def test_serving_infer_error_branches(tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(tmp_path / "outputs")),
        infer=SimpleNamespace(image_path=None, checkpoint_path="x.ckpt", device="cpu", top_k=1),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    with pytest.raises(SystemExit, match=r"infer\.image_path is required"):
        infer_mod.infer(cfg)


def test_serving_infer_image_not_found(tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(tmp_path / "outputs")),
        infer=SimpleNamespace(
            image_path="missing.jpg", checkpoint_path="x.ckpt", device="cpu", top_k=1
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    with pytest.raises(SystemExit, match="Image not found"):
        infer_mod.infer(cfg)


def test_serving_infer_checkpoint_not_found(tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    # Create empty outputs dir (no checkpoints)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(outputs_dir)),
        infer=SimpleNamespace(
            image_path=str(img_path),
            checkpoint_path="missing.ckpt",
            checkpoint_s3_uri=None,
            download=SimpleNamespace(enabled=False),
            device="cpu",
            top_k=1,
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    with pytest.raises(SystemExit, match="Checkpoint not found"):
        infer_mod.infer(cfg)


def test_serving_infer_downloads_checkpoint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    # Create empty outputs dir (no checkpoints to auto-discover)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    monkeypatch.setattr(
        infer_mod, "dvc_pull", lambda *a, **k: SimpleNamespace(success=True, message="ok")
    )

    downloaded = tmp_path / "best.ckpt"

    def fake_download_checkpoint_from_s3_uri(**_k):
        downloaded.write_bytes(b"fake")
        return SimpleNamespace(success=True, path=downloaded, message="ok")

    monkeypatch.setattr(
        infer_mod,
        "download_checkpoint_from_s3_uri",
        fake_download_checkpoint_from_s3_uri,
    )

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    monkeypatch.setattr(infer_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        infer_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(outputs_dir)),
        infer=SimpleNamespace(
            image_path=str(img_path),
            checkpoint_path=None,
            checkpoint_s3_uri="s3://b/k/best.ckpt",
            download=SimpleNamespace(enabled=True, dst_dir=str(tmp_path), overwrite=False),
            device="cpu",
            top_k=1,
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    result = infer_mod.infer(cfg)
    assert "predicted_class_id" in result


def test_serving_infer_resolve_device_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    monkeypatch.setattr(infer_mod.torch.cuda, "is_available", lambda: False)
    assert str(infer_mod._resolve_device("cuda")) == "cpu"
    assert str(infer_mod._resolve_device("auto")) == "cpu"
    assert str(infer_mod._resolve_device("cpu")) == "cpu"


def test_serving_infer_load_model_error_branches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    class DummyInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def load_state_dict(self, *_a, **_k):
            return ([], [])

    class Dummy:
        def __init__(self, *_a, **_k):
            self.eval_called = False
            self.to_called = False
            self.model = DummyInner()

        def eval(self):
            self.eval_called = True

        def to(self, _d):
            self.to_called = True

    monkeypatch.setattr(infer_mod, "CassavaClassifier", Dummy)

    # bad format (state_dict is not a dict)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": "nope"})
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))

    # unexpected keys
    class DummyUnexpectedInner(DummyInner):
        def load_state_dict(self, *_a, **_k):
            return ([], ["u"])

    class DummyUnexpected(Dummy):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.model = DummyUnexpectedInner()

    monkeypatch.setattr(infer_mod, "CassavaClassifier", DummyUnexpected)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": {}})
    with pytest.raises(ValueError, match="Unexpected keys"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))

    # missing keys
    class DummyMissingInner(DummyInner):
        def load_state_dict(self, *_a, **_k):
            return (["m"], [])

    class DummyMissing(Dummy):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.model = DummyMissingInner()

    monkeypatch.setattr(infer_mod, "CassavaClassifier", DummyMissing)
    with pytest.raises(ValueError, match="Missing keys"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))


def test_serving_infer_load_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    class DummyInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
            self.loaded = False

        def load_state_dict(self, *_a, **_k):
            self.loaded = True
            return ([], [])

    class Dummy(torch.nn.Module):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.model = DummyInner()
            self.eval_called = False
            self.to_called = False

        def eval(self):
            self.eval_called = True

        def to(self, _d):
            self.to_called = True

    monkeypatch.setattr(infer_mod, "CassavaClassifier", Dummy)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": {}})

    m = infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))
    assert m.eval_called is True
    assert m.to_called is True


def test_serving_infer_load_model_filters_model_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    loaded = {}

    class Inner(torch.nn.Module):
        def load_state_dict(self, sd, strict=False):
            loaded["keys"] = sorted(sd.keys())
            return ([], [])

    class Wrapper(torch.nn.Module):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.model = Inner()

        def eval(self):
            return None

        def to(self, _d):
            return None

    monkeypatch.setattr(infer_mod, "CassavaClassifier", Wrapper)

    def fake_torch_load(*_a, **_k):
        return {"state_dict": {"model.w": torch.zeros(()), "loss_fn.weight": torch.zeros(())}}

    monkeypatch.setattr(infer_mod.torch, "load", fake_torch_load)

    infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))
    assert loaded["keys"] == ["w"]


def test_serving_infer_s3_download_failure_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    # Create empty outputs dir (no checkpoints to auto-discover)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    monkeypatch.setattr(
        infer_mod, "dvc_pull", lambda *a, **k: SimpleNamespace(success=True, message="ok")
    )
    monkeypatch.setattr(
        infer_mod,
        "download_checkpoint_from_s3_uri",
        lambda **_k: SimpleNamespace(success=False, path=None, message="nope"),
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(outputs_dir)),
        infer=SimpleNamespace(
            image_path=str(img_path),
            checkpoint_path=None,
            checkpoint_s3_uri="s3://b/k/best.ckpt",
            device="cpu",
            top_k=1,
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    with pytest.raises(SystemExit, match="S3 download failed"):
        infer_mod.infer(cfg)


def test_serving_infer_load_model_wrapper_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    class Wrapper:
        def __init__(self, *_a, **_k):
            self.model = "not a module"

        def eval(self):
            return None

        def to(self, _d):
            return None

    monkeypatch.setattr(infer_mod, "CassavaClassifier", Wrapper)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": {}})
    with pytest.raises(ValueError, match="Unsupported model wrapper"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))


def test_serving_infer_cleanup_temp_file_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    # Create empty outputs dir (no checkpoints to auto-discover)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    monkeypatch.setattr(
        infer_mod, "dvc_pull", lambda *a, **k: SimpleNamespace(success=True, message="ok")
    )

    downloaded = tmp_path / "best.ckpt"

    def fake_download_checkpoint_from_s3_uri(**_k):
        downloaded.write_bytes(b"fake")
        return SimpleNamespace(success=True, path=downloaded, message="ok")

    monkeypatch.setattr(
        infer_mod,
        "download_checkpoint_from_s3_uri",
        fake_download_checkpoint_from_s3_uri,
    )

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    monkeypatch.setattr(infer_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        infer_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    # Force cleanup to fail.
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(outputs_dir)),
        infer=SimpleNamespace(
            image_path=str(img_path),
            checkpoint_path=None,
            checkpoint_s3_uri="s3://b/k/best.ckpt",
            device="cpu",
            top_k=1,
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    infer_mod.infer(cfg)
    out = capsys.readouterr().out
    assert "failed to cleanup" in out


def test_serving_infer_auto_discovers_latest_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Test that infer automatically discovers the latest checkpoint from outputs/."""
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    # Create outputs directory structure with multiple checkpoints
    outputs_dir = tmp_path / "outputs"
    run1_dir = outputs_dir / "2024-01-01" / "10-00-00" / "checkpoints"
    run1_dir.mkdir(parents=True)
    run1_ckpt = run1_dir / "best.ckpt"
    run1_ckpt.write_bytes(b"old checkpoint")

    run2_dir = outputs_dir / "2024-01-02" / "15-30-00" / "checkpoints"
    run2_dir.mkdir(parents=True)
    run2_ckpt = run2_dir / "best.ckpt"
    run2_ckpt.write_bytes(b"new checkpoint")

    monkeypatch.setattr(
        infer_mod, "dvc_pull", lambda *a, **k: SimpleNamespace(success=True, message="ok")
    )

    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((1, 5))

    monkeypatch.setattr(infer_mod, "_load_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        infer_mod,
        "build_transforms",
        lambda *a, **k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path), outputs_dir=str(outputs_dir)),
        infer=SimpleNamespace(
            image_path=str(img_path),
            checkpoint_path=None,  # Auto-discover
            checkpoint_s3_uri=None,
            device="cpu",
            top_k=1,
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    result = infer_mod.infer(cfg)
    assert "predicted_class_id" in result
    out = capsys.readouterr().out
    # Should mention auto-discovery
    assert "Auto-discovered latest checkpoint" in out
    # Should use the most recent checkpoint (run2) - check normalized path
    normalized_path = str(run2_ckpt).replace("\\", "/")
    assert normalized_path in out.replace(
        "\\", "/"
    ) or "2024-01-02/15-30-00/checkpoints/best.ckpt" in out.replace("\\", "/")
