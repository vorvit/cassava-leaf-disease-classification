"""Training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train(cfg: Any) -> None:
    """Run training using Hydra-composed config."""
    import warnings

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    from cassava_leaf_disease.data import dvc_pull
    from cassava_leaf_disease.training.datamodule import CassavaDataModule
    from cassava_leaf_disease.training.lightning_module import CassavaClassifier
    from cassava_leaf_disease.utils.git import get_git_commit_id

    # Reduce noise in console logs for Task2 checks / local debugging.
    # These warnings are informative, but not actionable on Windows where
    # num_workers=0 is often required.
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pin_memory.*no accelerator is found.*",
        category=UserWarning,
    )

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

    loggers: list[object] = [CSVLogger(save_dir=str(outputs_dir), name="runs")]

    git_commit_id = get_git_commit_id()
    hparams = {
        "git_commit_id": git_commit_id,
        "train": dict(cfg.train),
        "model": dict(cfg.model),
        "augment": dict(cfg.augment),
        "data": {
            "dataset": dict(cfg.data.dataset),
            "split": dict(cfg.data.split),
            "paths": dict(cfg.data.paths),
            "synthetic": dict(cfg.data.synthetic),
        },
        "logger": dict(cfg.logger),
    }

    if bool(getattr(cfg.logger, "enabled", False)):
        try:
            from pytorch_lightning.loggers import MLFlowLogger

            mlflow_logger = MLFlowLogger(
                tracking_uri=str(cfg.logger.tracking_uri),
                experiment_name=str(cfg.logger.experiment_name),
                run_name=None
                if cfg.logger.run_name in (None, "null")
                else str(cfg.logger.run_name),
            )
            mlflow_logger.log_hyperparams(hparams)
            loggers.append(mlflow_logger)
        except Exception as exc:
            print(f"[mlflow] logger disabled (continuing): {exc}")

    callbacks: list[object] = []
    if bool(getattr(cfg.train, "save_checkpoints", False)):
        from pytorch_lightning.callbacks import ModelCheckpoint

        callbacks.append(
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best",
                save_weights_only=True,
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        precision=str(cfg.train.precision),
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        fast_dev_run=bool(cfg.train.fast_dev_run),
        default_root_dir=str(outputs_dir),
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
