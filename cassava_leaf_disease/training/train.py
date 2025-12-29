"""Training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train(cfg: Any) -> None:
    """Run training using Hydra-composed config."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from cassava_leaf_disease.data import dvc_pull
    from cassava_leaf_disease.training.datamodule import CassavaDataModule
    from cassava_leaf_disease.training.lightning_module import CassavaClassifier

    pl.seed_everything(int(cfg.train.seed), workers=True)

    # Try to pull data (if tracked by DVC). On failure we can still run synthetic fallback.
    data_dir = str(cfg.paths.data_dir)
    pull_result = dvc_pull(targets=[data_dir])
    if not pull_result.success:
        print(f"[dvc] pull failed (continuing): {pull_result.message}")

    datamodule = CassavaDataModule(cfg)
    model = CassavaClassifier(cfg)

    outputs_dir = Path(str(cfg.paths.outputs_dir))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(save_dir=str(outputs_dir), name="runs")

    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        precision=str(cfg.train.precision),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        fast_dev_run=bool(cfg.train.fast_dev_run),
        default_root_dir=str(outputs_dir),
        logger=logger,
        callbacks=[ckpt],
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
