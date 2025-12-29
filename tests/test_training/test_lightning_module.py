"""Tests for cassava_leaf_disease.training.lightning_module."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_lightning_module_with_stubbed_timm(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            # Keep at least one trainable parameter so optimizer construction succeeds.
            self._head = torch.nn.Linear(1, 1)

        def forward(self, x):
            return torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)

    class FakeTimm:
        @staticmethod
        def create_model(_backbone, pretrained, num_classes, drop_rate):
            return DummyModel(int(num_classes))

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(
            loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0),
            lr=1e-3,
            weight_decay=0.0,
        ),
    )
    model = lightning_module.CassavaClassifier(cfg)
    batch = (torch.zeros((2, 3, 8, 8)), torch.tensor([0, 1]))
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    model.validation_step(batch, 0)
    opt = model.configure_optimizers()
    assert opt is not None


def test_lightning_module_loss_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(loss=SimpleNamespace(name="mse", label_smoothing=0.0)),
    )
    with pytest.raises(ValueError, match="Unsupported loss"):
        lightning_module.CassavaClassifier(cfg)


def test_lightning_module_no_loss_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=None,
    )
    model = lightning_module.CassavaClassifier(cfg)
    assert model.loss_fn is not None


def test_lightning_module_no_train_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(loss=None),
    )
    model = lightning_module.CassavaClassifier(cfg)
    assert model.loss_fn is not None
