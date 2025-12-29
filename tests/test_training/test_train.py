"""Tests for cassava_leaf_disease.training.train."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


def test_train_entrypoint_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassava_leaf_disease.training import train as train_mod

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": False,
                "artifacts": None,
            },
            "paths": {"data_dir": "data/cassava", "outputs_dir": "outputs"},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing_dir"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 42,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 4,
                    "val_size": 2,
                },
                "limits": {"max_train_samples": None, "max_val_samples": None},
            },
        }
    )

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    import pytorch_lightning as pl

    monkeypatch.setattr(pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )
    train_mod.train(cfg)


def test_train_mlflow_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    class MLFlowLogger:
        def __init__(self, **_k):
            self.run_id = "r1"

        def log_hyperparams(self, _h):
            return None

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr("pytorch_lightning.loggers.MLFlowLogger", MLFlowLogger, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": False,
                "artifacts": None,
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {
                "enabled": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "e",
                "run_name": "r",
            },
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_mlflow_exception(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    def fake_mlflow_logger(*_a, **_k):
        raise ImportError("mlflow missing")

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr("pytorch_lightning.loggers.MLFlowLogger", fake_mlflow_logger, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": False,
                "artifacts": None,
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {
                "enabled": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "e",
                "run_name": None,
            },
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_artifacts_branch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    # Fake MLFlow logger and experiment
    class Experiment:
        def __init__(self):
            self.artifacts = []
            self.params = {}

        def log_artifact(self, run_id, local_path, artifact_path=None):
            self.artifacts.append((run_id, local_path, artifact_path))

        def log_param(self, run_id, key, value):
            self.params[(run_id, key)] = value

    class MLFlowLogger:
        def __init__(self, **_k):
            self.run_id = "r1"
            self.experiment = Experiment()

        def log_hyperparams(self, _h):
            return None

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr("pytorch_lightning.loggers.MLFlowLogger", MLFlowLogger, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=False, message="no dvc"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    # Stub heavy model/datamodule
    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        lambda _cfg: object(),
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        lambda _cfg: object(),
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": True, "upload_checkpoint_to_s3": False},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {
                "enabled": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "e",
                "run_name": "r",
            },
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_artifacts_no_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = []

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": False,
                "artifacts": {"log_checkpoint_to_mlflow": True, "upload_checkpoint_to_s3": False},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_s3_upload_missing_creds(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("YC_ACCESS_KEY_ID", raising=False)

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": False, "upload_checkpoint_to_s3": True},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_mlflow_artifact_no_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class MLFlowLogger:
        def __init__(self, **_k):
            self.run_id = None
            self.experiment = None

        def log_hyperparams(self, _h):
            return None

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr("pytorch_lightning.loggers.MLFlowLogger", MLFlowLogger, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": True, "upload_checkpoint_to_s3": False},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {
                "enabled": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "e",
                "run_name": "r",
            },
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_mlflow_artifact_exception(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class Experiment:
        def log_artifact(self, run_id, local_path, artifact_path=None):
            raise RuntimeError("upload failed")

        def log_param(self, run_id, key, value):
            raise RuntimeError("log failed")

    class MLFlowLogger:
        def __init__(self, **_k):
            self.run_id = "r1"
            self.experiment = Experiment()

        def log_hyperparams(self, _h):
            return None

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr("pytorch_lightning.loggers.MLFlowLogger", MLFlowLogger, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": True, "upload_checkpoint_to_s3": False},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {
                "enabled": True,
                "tracking_uri": "http://127.0.0.1:8080",
                "experiment_name": "e",
                "run_name": "r",
            },
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_checkpoint_not_exists(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(tmp_path / "missing.ckpt")

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": True, "upload_checkpoint_to_s3": False},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_s3_upload_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    class MockS3Client:
        def upload_file(self, local_path, bucket, key):
            pass

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret")

    def fake_boto3_client(*args, **kwargs):
        return MockS3Client()

    import boto3

    monkeypatch.setattr(boto3, "client", fake_boto3_client)

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {
                    "log_checkpoint_to_mlflow": False,
                    "upload_checkpoint_to_s3": True,
                    "s3_bucket": "test",
                    "s3_prefix": "models",
                },
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_s3_upload_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    class MockS3Client:
        def upload_file(self, local_path, bucket, key):
            raise RuntimeError("S3 error")

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret")

    def fake_boto3_client(*args, **kwargs):
        return MockS3Client()

    import boto3

    monkeypatch.setattr(boto3, "client", fake_boto3_client)

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": False, "upload_checkpoint_to_s3": True},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)


def test_train_s3_upload_missing_deps(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from cassava_leaf_disease.training import train as train_mod

    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    class ModelCheckpoint:
        def __init__(self, **_k):
            self.best_model_path = str(ckpt)

    class DummyTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def fit(self, model=None, datamodule=None):
            return None

    monkeypatch.setattr("pytorch_lightning.Trainer", DummyTrainer, raising=False)
    monkeypatch.setattr(
        "pytorch_lightning.callbacks.ModelCheckpoint", ModelCheckpoint, raising=False
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.data.dvc_pull",
        lambda *a, **k: SimpleNamespace(success=True, message="ok"),
    )
    monkeypatch.setattr("cassava_leaf_disease.utils.git.get_git_commit_id", lambda: "c0")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    class DummyDataModule:
        def __init__(self, _cfg):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _cfg):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        "cassava_leaf_disease.training.datamodule.CassavaDataModule",
        DummyDataModule,
        raising=False,
    )
    monkeypatch.setattr(
        "cassava_leaf_disease.training.lightning_module.CassavaClassifier",
        DummyModel,
        raising=False,
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 1,
                "epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "log_every_n_steps": 1,
                "fast_dev_run": True,
                "save_checkpoints": True,
                "artifacts": {"log_checkpoint_to_mlflow": False, "upload_checkpoint_to_s3": True},
            },
            "paths": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
            "logger": {"enabled": False},
            "model": {"backbone": "resnet18", "pretrained": False, "dropout": 0.0},
            "augment": {"image_size": 8, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
            "data": {
                "dataset": {"num_classes": 5, "class_names": ["a", "b", "c", "d", "e"]},
                "paths": {"train_csv": "missing.csv", "images_dir": "missing"},
                "split": {
                    "strategy": "holdout",
                    "val_size": 0.2,
                    "seed": 1,
                    "folds": 5,
                    "fold_index": 0,
                },
                "synthetic": {
                    "enabled": True,
                    "fallback_if_missing": True,
                    "seed": 1,
                    "train_size": 2,
                    "val_size": 1,
                },
            },
        }
    )
    train_mod.train(cfg)
