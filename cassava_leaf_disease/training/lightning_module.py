"""LightningModule for cassava classification."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class CassavaClassifier(pl.LightningModule):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        num_classes = int(cfg.data.dataset.num_classes)
        backbone = str(cfg.model.backbone)

        import timm

        self.model = timm.create_model(
            backbone,
            pretrained=bool(cfg.model.pretrained),
            num_classes=num_classes,
            drop_rate=float(cfg.model.dropout),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.f1_macro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/acc",
            self.acc(preds, y),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/f1_macro",
            self.f1_macro(preds, y),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.acc(preds, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val/f1_macro",
            self.f1_macro(preds, y),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.train.lr),
            weight_decay=float(self.cfg.train.weight_decay),
        )
        return optimizer
