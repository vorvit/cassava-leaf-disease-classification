"""Tests for cassava_leaf_disease.serving.infer."""

from __future__ import annotations

import importlib
import json
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
        paths=SimpleNamespace(data_dir=str(tmp_path)),
        infer=SimpleNamespace(
            image_path=str(img_path), checkpoint_path=str(ckpt), device="cpu", top_k=3
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a", "b", "c", "d", "e"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    result = infer_mod.infer(cfg)
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["predicted_class_id"] == result["predicted_class_id"]


def test_serving_infer_error_branches(tmp_path) -> None:
    import importlib

    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")
    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path)),
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
        paths=SimpleNamespace(data_dir=str(tmp_path)),
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

    cfg = SimpleNamespace(
        paths=SimpleNamespace(data_dir=str(tmp_path)),
        infer=SimpleNamespace(
            image_path=str(img_path), checkpoint_path="missing.ckpt", device="cpu", top_k=1
        ),
        data=SimpleNamespace(dataset=SimpleNamespace(class_names=["a"])),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    )
    with pytest.raises(SystemExit, match="Checkpoint not found"):
        infer_mod.infer(cfg)


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

    class Dummy:
        def __init__(self, *_a, **_k):
            self.eval_called = False
            self.to_called = False

        def load_state_dict(self, *_a, **_k):
            return ([], [])

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
    class DummyUnexpected(Dummy):
        def load_state_dict(self, *_a, **_k):
            return ([], ["u"])

    monkeypatch.setattr(infer_mod, "CassavaClassifier", DummyUnexpected)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": {}})
    with pytest.raises(ValueError, match="Unexpected keys"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))

    # missing keys
    class DummyMissing(Dummy):
        def load_state_dict(self, *_a, **_k):
            return (["m"], [])

    monkeypatch.setattr(infer_mod, "CassavaClassifier", DummyMissing)
    with pytest.raises(ValueError, match="Missing keys"):
        infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))


def test_serving_infer_load_model_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    infer_mod = importlib.import_module("cassava_leaf_disease.serving.infer")

    class Dummy(torch.nn.Module):
        def __init__(self, *_a, **_k):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
            self.eval_called = False
            self.to_called = False

        def load_state_dict(self, *_a, **_k):
            return ([], [])

        def eval(self):
            self.eval_called = True

        def to(self, _d):
            self.to_called = True

    monkeypatch.setattr(infer_mod, "CassavaClassifier", Dummy)
    monkeypatch.setattr(infer_mod.torch, "load", lambda *_a, **_k: {"state_dict": {}})

    m = infer_mod._load_model(SimpleNamespace(), tmp_path / "x.ckpt", torch.device("cpu"))
    assert m.eval_called is True
    assert m.to_called is True
