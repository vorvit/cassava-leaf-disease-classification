"""LightningModule for cassava classification."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class CassavaClassifier(pl.LightningModule):
    def __init__(self, cfg: Any, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        self._backbone_frozen = False
        num_classes = int(cfg.data.dataset.num_classes)
        backbone = str(cfg.model.backbone)

        import timm

        self.model = timm.create_model(
            backbone,
            pretrained=bool(cfg.model.pretrained),
            num_classes=num_classes,
            drop_rate=float(cfg.model.dropout),
        )

        if bool(getattr(cfg.model, "freeze_backbone", False)):
            self._freeze_backbone_keep_head_trainable()

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
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights.to(dtype=torch.float32),
                label_smoothing=label_smoothing,
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.f1_macro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        )

    def _iter_head_params(self) -> Iterator[torch.nn.Parameter]:
        # timm models commonly implement `get_classifier()`.
        # As fallback, try typical attribute names.
        getter = getattr(self.model, "get_classifier", None)
        if callable(getter):
            head = getter()
            if isinstance(head, torch.nn.Module):
                yield from head.parameters()
                return

        for name in ("classifier", "fc", "head"):
            head = getattr(self.model, name, None)
            if isinstance(head, torch.nn.Module):
                yield from head.parameters()
                return

    def _freeze_backbone_keep_head_trainable(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze head params so optimizer isn't empty and model can learn.
        for param in self._iter_head_params():
            param.requires_grad = True

        self._backbone_frozen = True

    def on_train_epoch_start(self) -> None:
        unfreeze_epoch_raw = getattr(self.cfg.model, "unfreeze_epoch", None)
        if unfreeze_epoch_raw in (None, "null"):
            return

        if not isinstance(unfreeze_epoch_raw, (int, str)):
            return
        try:
            unfreeze_epoch = int(unfreeze_epoch_raw)
        except Exception:
            return

        if self._backbone_frozen and int(self.current_epoch) == unfreeze_epoch:
            for param in self.model.parameters():
                param.requires_grad = True
            self._backbone_frozen = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/acc",
            self.acc(preds, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/f1_macro",
            self.f1_macro(preds, labels),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val/acc",
            self.acc(preds, labels),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/f1_macro",
            self.f1_macro(preds, labels),
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

        scheduler_cfg = getattr(train_cfg, "scheduler", None) if train_cfg else None
        name = str(getattr(scheduler_cfg, "name", "none")).lower() if scheduler_cfg else "none"
        if name in {"none", "null"}:
            return optimizer

        if name == "cosine":
            t_max = int(getattr(scheduler_cfg, "t_max", getattr(train_cfg, "epochs", 1)))
            eta_min = float(getattr(scheduler_cfg, "eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max(1, t_max),
                eta_min=eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if name in {"plateau", "reduce_on_plateau"}:
            factor = float(getattr(scheduler_cfg, "factor", 0.5))
            patience = int(getattr(scheduler_cfg, "patience", 1))
            min_lr = float(getattr(scheduler_cfg, "min_lr", 1e-6))
            monitor = str(getattr(scheduler_cfg, "monitor", "val/loss"))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": monitor},
            }

        raise ValueError(f"Unsupported scheduler: {name!r}. Supported: none|cosine|plateau")
