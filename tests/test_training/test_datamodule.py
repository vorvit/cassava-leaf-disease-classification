"""Tests for cassava_leaf_disease.training.datamodule."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


def test_datamodule_synthetic_setup_and_dataloaders() -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=42, batch_size=4, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv="missing.csv", images_dir="missing_dir"),
            split=SimpleNamespace(strategy="holdout", val_size=0.2, seed=42, folds=5, fold_index=0),
            synthetic=SimpleNamespace(
                enabled=True, fallback_if_missing=True, seed=1, train_size=8, val_size=4
            ),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert len(batch) == 2


def test_datamodule_holdout_and_parse_helpers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    # cover _parse_optional_int branches
    dm = CassavaDataModule(
        SimpleNamespace(train=SimpleNamespace(num_workers=0), data=None, augment=None)
    )
    assert dm._parse_optional_int(None) is None
    assert dm._parse_optional_int("null") is None
    assert dm._parse_optional_int("5") == 5
    assert dm._parse_optional_int(object()) is None

    # holdout path + limits
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    for name in [
        "a0.jpg",
        "a1.jpg",
        "b0.jpg",
        "b1.jpg",
        "c0.jpg",
        "c1.jpg",
        "d0.jpg",
        "d1.jpg",
        "e0.jpg",
        "e1.jpg",
    ]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images_dir / name)

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image_id,label\n"
        "a0.jpg,0\na1.jpg,0\n"
        "b0.jpg,1\nb1.jpg,1\n"
        "c0.jpg,2\nc1.jpg,2\n"
        "d0.jpg,3\nd1.jpg,3\n"
        "e0.jpg,4\ne1.jpg,4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        dm_mod,
        "build_transforms",
        lambda *_a, **_k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=1, batch_size=2, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv=str(train_csv), images_dir=str(images_dir)),
            split=SimpleNamespace(strategy="holdout", val_size=0.5, seed=1, folds=5, fold_index=0),
            synthetic=SimpleNamespace(enabled=False, fallback_if_missing=False),
            limits=SimpleNamespace(max_train_samples="3", max_val_samples="2"),
        ),
    )
    dm2 = CassavaDataModule(cfg)
    with pytest.raises(RuntimeError):
        dm2.train_dataloader()
    dm2.setup()
    batch = next(iter(dm2.train_dataloader()))
    assert len(batch) == 2


def test_datamodule_kfold(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    for name in [
        "a0.jpg",
        "a1.jpg",
        "a2.jpg",
        "b0.jpg",
        "b1.jpg",
        "b2.jpg",
        "c0.jpg",
        "c1.jpg",
        "c2.jpg",
        "d0.jpg",
        "d1.jpg",
        "d2.jpg",
        "e0.jpg",
        "e1.jpg",
        "e2.jpg",
    ]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images_dir / name)

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image_id,label\n"
        "a0.jpg,0\na1.jpg,0\na2.jpg,0\n"
        "b0.jpg,1\nb1.jpg,1\nb2.jpg,1\n"
        "c0.jpg,2\nc1.jpg,2\nc2.jpg,2\n"
        "d0.jpg,3\nd1.jpg,3\nd2.jpg,3\n"
        "e0.jpg,4\ne1.jpg,4\ne2.jpg,4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        dm_mod,
        "build_transforms",
        lambda *_a, **_k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=1, batch_size=2, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv=str(train_csv), images_dir=str(images_dir)),
            split=SimpleNamespace(strategy="kfold", val_size=0.2, seed=1, folds=3, fold_index=0),
            synthetic=SimpleNamespace(enabled=False, fallback_if_missing=False),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 2


def test_datamodule_resolve_num_workers_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod

    cfg = SimpleNamespace(
        train=SimpleNamespace(num_workers="auto"),
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        augment=SimpleNamespace(),
    )
    dm = dm_mod.CassavaDataModule(cfg)

    monkeypatch.setattr(dm_mod.sys, "platform", "linux")
    monkeypatch.setattr(dm_mod.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: False)
    assert dm._resolve_num_workers() == 7

    monkeypatch.setattr(dm_mod.sys, "platform", "win32")
    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: True)
    assert dm._resolve_num_workers() == 0

    # Test explicit num_workers on Linux (not Windows+CUDA)
    monkeypatch.setattr(dm_mod.sys, "platform", "linux")
    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: False)
    cfg2 = SimpleNamespace(
        train=SimpleNamespace(num_workers=4),
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        augment=SimpleNamespace(),
    )
    dm2 = dm_mod.CassavaDataModule(cfg2)
    assert dm2._resolve_num_workers() == 4


def test_datamodule_resolve_pin_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod

    cfg = SimpleNamespace(
        train=SimpleNamespace(num_workers=0),
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        augment=SimpleNamespace(),
    )
    dm = dm_mod.CassavaDataModule(cfg)

    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: True)
    assert dm._resolve_pin_memory() is True

    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: False)
    assert dm._resolve_pin_memory() is False


def test_datamodule_missing_columns(tmp_path) -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("bad,columns\n", encoding="utf-8")

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=1, batch_size=2, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv=str(train_csv), images_dir=str(tmp_path)),
            split=SimpleNamespace(strategy="holdout", val_size=0.2, seed=1, folds=5, fold_index=0),
            synthetic=SimpleNamespace(enabled=False, fallback_if_missing=False),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    with pytest.raises(ValueError, match="Expected columns"):
        dm.setup()


def test_datamodule_kfold_invalid_fold_index(tmp_path) -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image_id,label\na.jpg,0\n", encoding="utf-8")

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=1, batch_size=2, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv=str(train_csv), images_dir=str(tmp_path)),
            split=SimpleNamespace(strategy="kfold", val_size=0.2, seed=1, folds=3, fold_index=5),
            synthetic=SimpleNamespace(enabled=False, fallback_if_missing=False),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    with pytest.raises(ValueError, match="fold_index must be in"):
        dm.setup()


def test_datamodule_prepare_data() -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    cfg = SimpleNamespace(train=SimpleNamespace(num_workers=0), data=None, augment=None)
    dm = CassavaDataModule(cfg)
    # Should not crash
    dm.prepare_data()


def test_datamodule_no_limits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    for name in ["a0.jpg", "a1.jpg", "b0.jpg", "b1.jpg"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(images_dir / name)

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image_id,label\na0.jpg,0\na1.jpg,0\nb0.jpg,1\nb1.jpg,1\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        dm_mod,
        "build_transforms",
        lambda *_a, **_k: (lambda image: {"image": torch.zeros((3, 8, 8))}),
    )

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=1, batch_size=2, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=2),
            paths=SimpleNamespace(train_csv=str(train_csv), images_dir=str(images_dir)),
            split=SimpleNamespace(strategy="holdout", val_size=0.5, seed=1, folds=5, fold_index=0),
            synthetic=SimpleNamespace(enabled=False, fallback_if_missing=False),
            limits=None,
        ),
    )
    dm = CassavaDataModule(cfg)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 2


def test_datamodule_win32_cuda_num_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    import cassava_leaf_disease.training.datamodule as dm_mod

    cfg = SimpleNamespace(
        train=SimpleNamespace(num_workers="auto"),
        data=SimpleNamespace(dataset=SimpleNamespace(num_classes=5)),
        augment=SimpleNamespace(),
    )
    dm = dm_mod.CassavaDataModule(cfg)

    monkeypatch.setattr(dm_mod.sys, "platform", "win32")
    monkeypatch.setattr(dm_mod.torch.cuda, "is_available", lambda: True)
    assert dm._resolve_num_workers() == 0


def test_datamodule_val_dataloader_error_before_setup() -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=42, batch_size=4, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv="missing.csv", images_dir="missing_dir"),
            split=SimpleNamespace(strategy="holdout", val_size=0.2, seed=42, folds=5, fold_index=0),
            synthetic=SimpleNamespace(
                enabled=True, fallback_if_missing=True, seed=1, train_size=8, val_size=4
            ),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    with pytest.raises(RuntimeError, match="Call setup"):
        dm.val_dataloader()


def test_datamodule_val_dataloader(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training.datamodule import CassavaDataModule

    cfg = SimpleNamespace(
        train=SimpleNamespace(seed=42, batch_size=4, num_workers=0),
        augment=SimpleNamespace(image_size=8, mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        data=SimpleNamespace(
            dataset=SimpleNamespace(num_classes=5),
            paths=SimpleNamespace(train_csv="missing.csv", images_dir="missing_dir"),
            split=SimpleNamespace(strategy="holdout", val_size=0.2, seed=42, folds=5, fold_index=0),
            synthetic=SimpleNamespace(
                enabled=True, fallback_if_missing=True, seed=1, train_size=8, val_size=4
            ),
            limits=SimpleNamespace(max_train_samples=None, max_val_samples=None),
        ),
    )
    dm = CassavaDataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    assert len(batch) == 2
