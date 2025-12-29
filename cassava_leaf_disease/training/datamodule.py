"""PyTorch Lightning DataModule."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from cassava_leaf_disease.training.dataset import (
    CassavaCsvDataset,
    CassavaSample,
    SyntheticCassavaDataset,
)
from cassava_leaf_disease.training.transforms import build_transforms


@dataclass(frozen=True)
class DataPaths:
    train_csv: Path
    images_dir: Path


class CassavaDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.augment_cfg = cfg.augment

        self._train_dataset: Dataset[Any] | None = None
        self._val_dataset: Dataset[Any] | None = None

    @staticmethod
    def _parse_optional_int(value: object) -> int | None:
        """Parse optional int from Hydra/OmegaConf values."""
        if value is None:
            return None
        if isinstance(value, str) and value.lower() == "null":
            return None
        try:
            return int(value)  # type: ignore[call-overload]
        except Exception:
            return None

    def _resolve_num_workers(self) -> int:
        raw = getattr(self.cfg.train, "num_workers", 0)
        if isinstance(raw, str) and raw.lower() == "auto":
            # Windows multiprocessing often causes instability / high memory overhead
            # for torch DataLoader workers (especially in constrained environments).
            # For a predictable "it just runs" experience we default to 0 on Windows.
            if sys.platform.startswith("win"):
                return 0

            cpu_count = os.cpu_count() or 0
            num_workers = max(0, cpu_count - 1)
        else:
            num_workers = int(raw)

        # Windows + CUDA: keep single-process loading to avoid common worker crashes.
        if sys.platform.startswith("win") and torch.cuda.is_available():
            return 0

        return max(0, num_workers)

    def _resolve_pin_memory(self) -> bool:
        # Only useful when CUDA is available; otherwise it produces a warning.
        return bool(torch.cuda.is_available())

    def prepare_data(self) -> None:
        # DVC pull is handled at a higher level (CLI), to keep this module reusable.
        return

    def setup(self, stage: str | None = None) -> None:
        num_classes = int(self.data_cfg.dataset.num_classes)

        train_transform = build_transforms(self.augment_cfg, is_train=True)
        val_transform = build_transforms(self.augment_cfg, is_train=False)

        train_csv = Path(self.data_cfg.paths.train_csv)
        images_dir = Path(self.data_cfg.paths.images_dir)

        synthetic_cfg = getattr(self.data_cfg, "synthetic", None)
        synthetic_enabled = (
            bool(getattr(synthetic_cfg, "enabled", False)) if synthetic_cfg else False
        )
        fallback_if_missing = (
            bool(getattr(synthetic_cfg, "fallback_if_missing", True)) if synthetic_cfg else True
        )

        data_missing = not train_csv.exists() or not images_dir.exists()
        if synthetic_enabled or (fallback_if_missing and data_missing):
            image_size = int(self.augment_cfg.image_size)
            seed = (
                int(getattr(synthetic_cfg, "seed", self.cfg.train.seed))
                if synthetic_cfg
                else int(self.cfg.train.seed)
            )
            train_size = int(getattr(synthetic_cfg, "train_size", 256)) if synthetic_cfg else 256
            val_size = int(getattr(synthetic_cfg, "val_size", 64)) if synthetic_cfg else 64

            self._train_dataset = SyntheticCassavaDataset(
                size=train_size, num_classes=num_classes, image_size=image_size, seed=seed
            )
            self._val_dataset = SyntheticCassavaDataset(
                size=val_size, num_classes=num_classes, image_size=image_size, seed=seed + 1
            )
            return

        df = pd.read_csv(train_csv)
        if "image_id" not in df.columns or "label" not in df.columns:
            raise ValueError("Expected columns: image_id,label in train.csv")

        samples = [
            CassavaSample(image_id=str(row.image_id), label=int(row.label))
            for row in df.itertuples()
        ]
        labels = [s.label for s in samples]

        split_strategy = str(self.data_cfg.split.strategy)
        if split_strategy == "kfold":
            folds = int(self.data_cfg.split.folds)
            fold_index = int(self.data_cfg.split.fold_index)
            if not (0 <= fold_index < folds):
                raise ValueError("fold_index must be in [0, folds)")

            skf = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=int(self.data_cfg.split.seed),
            )
            all_indices = list(range(len(samples)))
            splits = list(skf.split(all_indices, labels))
            train_idx, val_idx = splits[fold_index]
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
        else:
            train_samples, val_samples = train_test_split(
                samples,
                test_size=float(self.data_cfg.split.val_size),
                random_state=int(self.data_cfg.split.seed),
                stratify=labels,
            )

        limits_cfg = getattr(self.data_cfg, "limits", None)
        if limits_cfg:
            max_train_raw = getattr(limits_cfg, "max_train_samples", None)
            max_val_raw = getattr(limits_cfg, "max_val_samples", None)
            max_train = self._parse_optional_int(max_train_raw)
            max_val = self._parse_optional_int(max_val_raw)
            if max_train is not None:
                train_samples = train_samples[:max_train]
            if max_val is not None:
                val_samples = val_samples[:max_val]

        self._train_dataset = CassavaCsvDataset(
            train_samples, images_dir=images_dir, transform=train_transform
        )
        self._val_dataset = CassavaCsvDataset(
            val_samples, images_dir=images_dir, transform=val_transform
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        num_workers = self._resolve_num_workers()
        return DataLoader(
            self._train_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self._resolve_pin_memory(),
            persistent_workers=(num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        num_workers = self._resolve_num_workers()
        return DataLoader(
            self._val_dataset,
            batch_size=int(self.cfg.train.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self._resolve_pin_memory(),
            persistent_workers=(num_workers > 0),
        )
