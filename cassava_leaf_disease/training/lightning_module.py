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

        train_cfg = getattr(cfg, "train", None)
        loss_cfg = getattr(train_cfg, "loss", None) if train_cfg else None
        if loss_cfg:
            loss_name = str(getattr(loss_cfg, "name", "cross_entropy")).lower()
            label_smoothing = float(getattr(loss_cfg, "label_smoothing", 0.0))
        else:
            loss_name = "cross_entropy"
            label_smoothing = 0.0
        if loss_name != "cross_entropy":
            raise ValueError(f"Unsupported loss: {loss_name!r}. Supported: 'cross_entropy'")
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
        train_cfg = getattr(self.cfg, "train", None)
        lr = float(getattr(train_cfg, "lr", 0.0003)) if train_cfg else 0.0003
        weight_decay = float(getattr(train_cfg, "weight_decay", 0.0001)) if train_cfg else 0.0001
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer
