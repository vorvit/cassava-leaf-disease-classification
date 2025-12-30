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
            self.classifier = torch.nn.Linear(1, 1)

        def forward(self, x):
            return torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)

        def get_classifier(self):
            return self.classifier

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


def test_lightning_module_scheduler_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0),
            scheduler=SimpleNamespace(name="cosine", t_max=1, eta_min=0.0),
            epochs=1,
        ),
    )

    model = lightning_module.CassavaClassifier(cfg)
    out = model.configure_optimizers()
    assert isinstance(out, dict)
    assert "optimizer" in out
    assert "lr_scheduler" in out


def test_lightning_module_scheduler_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0),
            scheduler=SimpleNamespace(
                name="plateau",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
                monitor="val/loss",
            ),
        ),
    )

    model = lightning_module.CassavaClassifier(cfg)
    out = model.configure_optimizers()
    assert isinstance(out, dict)
    assert out["lr_scheduler"]["monitor"] == "val/loss"


def test_lightning_module_freeze_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            self.backbone_param = torch.nn.Parameter(torch.zeros(()))
            self.classifier = torch.nn.Linear(1, 1)

        def forward(self, x):
            return torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)

        def get_classifier(self):
            return self.classifier

    class FakeTimm:
        @staticmethod
        def create_model(_backbone, pretrained, num_classes, drop_rate):
            return DummyModel(int(num_classes))

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(
            backbone="efficientnet_b0",
            pretrained=True,
            dropout=0.0,
            freeze_backbone=True,
            unfreeze_epoch=None,
        ),
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    assert model.model.backbone_param.requires_grad is False
    assert any(p.requires_grad for p in model.model.classifier.parameters())


def test_lightning_module_unfreeze_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            self.backbone_param = torch.nn.Parameter(torch.zeros(()))
            self.classifier = torch.nn.Linear(1, 1)

        def forward(self, x):
            return torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)

        def get_classifier(self):
            return self.classifier

    class FakeTimm:
        @staticmethod
        def create_model(_backbone, pretrained, num_classes, drop_rate):
            return DummyModel(int(num_classes))

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(
            backbone="efficientnet_b0",
            pretrained=True,
            dropout=0.0,
            freeze_backbone=True,
            unfreeze_epoch=0,
        ),
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    assert model.model.backbone_param.requires_grad is False
    model.on_train_epoch_start()
    assert model.model.backbone_param.requires_grad is True


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


def test_lightning_module_class_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )
    weights = torch.ones((5,), dtype=torch.float32)
    model = lightning_module.CassavaClassifier(cfg, class_weights=weights)
    assert model.loss_fn.weight is not None


def test_lightning_module_scheduler_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(backbone="resnet18", pretrained=False, dropout=0.0),
        train=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0),
            scheduler=SimpleNamespace(name="wtf"),
        ),
    )
    model = lightning_module.CassavaClassifier(cfg)
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        model.configure_optimizers()


def test_lightning_module_iter_head_params_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _iter_head_params fallback to classifier/fc/head attributes."""
    from cassava_leaf_disease.training import lightning_module

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            # No get_classifier, but has 'fc' attribute
            self.fc = torch.nn.Linear(1, num_classes)

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
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    # Test that _iter_head_params works with 'fc' fallback
    params = list(model._iter_head_params())
    assert len(params) > 0


def test_lightning_module_iter_head_params_classifier_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _iter_head_params with 'classifier' attribute."""
    from cassava_leaf_disease.training import lightning_module

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            self.classifier = torch.nn.Linear(1, num_classes)

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
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    params = list(model._iter_head_params())
    assert len(params) > 0


def test_lightning_module_on_train_epoch_start_no_unfreeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test on_train_epoch_start when unfreeze_epoch is None."""
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(
            backbone="resnet18", pretrained=False, dropout=0.0, unfreeze_epoch=None
        ),
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    model.on_train_epoch_start()  # Should return early


def test_lightning_module_on_train_epoch_start_invalid_unfreeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test on_train_epoch_start with invalid unfreeze_epoch."""
    from cassava_leaf_disease.training import lightning_module

    class FakeTimm:
        @staticmethod
        def create_model(*_a, **_k):
            return torch.nn.Linear(1, 5)

    monkeypatch.setitem(__import__("sys").modules, "timm", FakeTimm)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        model=SimpleNamespace(
            backbone="resnet18", pretrained=False, dropout=0.0, unfreeze_epoch="invalid"
        ),
        train=SimpleNamespace(loss=SimpleNamespace(name="cross_entropy", label_smoothing=0.0)),
    )

    model = lightning_module.CassavaClassifier(cfg)
    model.on_train_epoch_start()  # Should handle invalid value gracefully
